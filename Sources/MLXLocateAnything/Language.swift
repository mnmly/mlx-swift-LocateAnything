import Foundation
import MLX
import MLXNN
import MLXLMCommon

// Ported from mlx_vlm/models/locateanything/language.py — a Qwen2 text decoder
// (Qwen2.5-3B-Instruct backbone) with two RoPE paths:
//   * standard nn.RoPE(offset:) for autoregressive decoding
//   * explicit position_ids cos/sin for Parallel Box Decoding (Phase 2)
// plus the MAGI block-attention mask used by PBD.

// MARK: - Explicit RoPE (position_ids path, GPT-NeoX rotate-half)

/// Returns (cos, sin) each [B, 1, L, headDim] from explicit position ids.
func ropeCosSin(positionIds: MLXArray, headDim: Int, base: Float) -> (MLXArray, MLXArray) {
    let half = headDim / 2
    let invFreq = MLX.pow(MLXArray(base), -MLXArray(0 ..< half).asType(.float32) / Float(half))
    // positionIds: [B, L] -> freqs [B, L, half]
    let freqs = positionIds.expandedDimensions(axis: -1).asType(.float32) * invFreq
    let emb = concatenated([freqs, freqs], axis: -1)         // [B, L, headDim]
    let cos = MLX.cos(emb).expandedDimensions(axis: 1)       // [B, 1, L, headDim]
    let sin = MLX.sin(emb).expandedDimensions(axis: 1)
    return (cos, sin)
}

/// Apply rotate-half RoPE. `x`: [B, heads, L, headDim]; cos/sin: [B, 1, L, headDim].
func applyRopeWithCosSin(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    let rotated = concatenated([-x2, x1], axis: -1)
    return (x * cos) + (rotated * sin)
}

// MARK: - MAGI block attention mask (PBD, Phase 2)

/// Additive mask [1, 1, q_len, kv_len] (0 / -inf) implementing the MAGI
/// multi-token-prediction block attention pattern.
func buildMagiBlockMask(kvLen: Int, qLen: Int, blockSize: Int) -> MLXArray {
    let qGlobalStart = kvLen - qLen
    let windowStartK = kvLen - blockSize
    let blockedK = windowStartK - 1

    let qIdx = MLXArray(0 ..< qLen).reshaped([qLen, 1])
    let kIdx = MLXArray(0 ..< kvLen).reshaped([1, kvLen])
    let gIdx = qIdx + qGlobalStart

    let inWindow = qIdx .>= (qLen - blockSize)
    let causal = logicalNot(inWindow) .&& (kIdx .<= gIdx)
    let winToPrefix = inWindow .&& (kIdx .< blockedK)
    let winToWindow = inWindow .&& (kIdx .>= windowStartK)
    let allowed = causal .|| winToPrefix .|| winToWindow

    let zero = MLXArray(Float(0))
    let neg = MLXArray(-Float.infinity)
    let mask = `where`(allowed, zero, neg)
    return mask.reshaped([1, 1, qLen, kvLen])
}

// MARK: - SwiGLU MLP

final class TextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(dim: Int, hiddenDim: Int) {
        self._gate = ModuleInfo(wrappedValue: Linear(dim, hiddenDim, bias: false), key: "gate_proj")
        self._up = ModuleInfo(wrappedValue: Linear(dim, hiddenDim, bias: false), key: "up_proj")
        self._down = ModuleInfo(wrappedValue: Linear(hiddenDim, dim, bias: false), key: "down_proj")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Attention

final class TextAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let ropeBase: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let rope: RoPE

    init(_ config: TextConfiguration) {
        let dim = config.hiddenSize
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = dim / config.numAttentionHeads
        self.scale = 1.0 / Float(headDim).squareRoot()
        self.ropeBase = config.ropeTheta

        self._qProj = ModuleInfo(wrappedValue: Linear(dim, nHeads * headDim, bias: true), key: "q_proj")
        self._kProj = ModuleInfo(wrappedValue: Linear(dim, nKVHeads * headDim, bias: true), key: "k_proj")
        self._vProj = ModuleInfo(wrappedValue: Linear(dim, nKVHeads * headDim, bias: true), key: "v_proj")
        self._oProj = ModuleInfo(wrappedValue: Linear(nHeads * headDim, dim, bias: false), key: "o_proj")
        self.rope = RoPE(dimensions: headDim, traditional: config.ropeTraditional, base: config.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        positionIds: MLXArray?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = qProj(x).reshaped(B, L, nHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        if let positionIds {
            let (cos, sin) = ropeCosSin(positionIds: positionIds, headDim: headDim, base: ropeBase)
            q = applyRopeWithCosSin(q, cos: cos, sin: sin)
            k = applyRopeWithCosSin(k, cos: cos, sin: sin)
            if let cache {
                (k, v) = cache.update(keys: k, values: v)
            }
        } else if let cache {
            q = rope(q, offset: cache.offset)
            k = rope(k, offset: cache.offset)
            (k, v) = cache.update(keys: k, values: v)
        } else {
            q = rope(q)
            k = rope(k)
        }

        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        out = out.transposed(0, 2, 1, 3).reshaped(B, L, nHeads * headDim)
        return oProj(out)
    }
}

// MARK: - Decoder layer

final class Qwen2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: TextAttention
    @ModuleInfo(key: "mlp") var mlp: TextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: TextConfiguration) {
        self._selfAttn = ModuleInfo(wrappedValue: TextAttention(config), key: "self_attn")
        self._mlp = ModuleInfo(wrappedValue: TextMLP(dim: config.hiddenSize, hiddenDim: config.intermediateSize), key: "mlp")
        self._inputLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps), key: "input_layernorm")
        self._postAttentionLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps), key: "post_attention_layernorm")
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        positionIds: MLXArray?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache, positionIds: positionIds)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - Inner model

final class Qwen2Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: TextConfiguration) {
        self._embedTokens = ModuleInfo(wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize), key: "embed_tokens")
        self._layers = ModuleInfo(wrappedValue: (0 ..< config.numHiddenLayers).map { _ in Qwen2DecoderLayer(config) }, key: "layers")
        self._norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps), key: "norm")
        super.init()
    }

    func callAsFunction(
        inputs: MLXArray?,
        inputsEmbeds: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: [KVCache]?,
        positionIds: MLXArray?
    ) -> MLXArray {
        var h = inputsEmbeds ?? embedTokens(inputs!)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], positionIds: positionIds)
        }
        return norm(h)
    }
}

// MARK: - Language model (tied lm head)

public final class LanguageModel: Module {
    let config: TextConfiguration
    @ModuleInfo(key: "model") var model: Qwen2Model

    init(_ config: TextConfiguration) {
        self.config = config
        self._model = ModuleInfo(wrappedValue: Qwen2Model(config), key: "model")
        super.init()
    }

    var headDim: Int { config.hiddenSize / config.numAttentionHeads }
    var kvHeads: [Int] { (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads } }

    func callAsFunction(
        inputs: MLXArray?,
        inputsEmbeds: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: [KVCache]?,
        positionIds: MLXArray?
    ) -> MLXArray {
        let out = model(
            inputs: inputs, inputsEmbeds: inputsEmbeds, mask: mask, cache: cache, positionIds: positionIds)
        // tie_word_embeddings == true: logits via the (quantized) embedding.
        return model.embedTokens.asLinear(out)
    }
}
