import Foundation
import MLX
import MLXNN
import MLXLMCommon

// Ported from mlx_vlm/models/locateanything/locateanything.py — the top-level
// composition: MoonViT vision tower -> LN/Linear/GELU/Linear connector ->
// InternVL-style scatter of visual tokens into the text sequence at
// <IMG_CONTEXT> positions -> Qwen2 decoder.

// MARK: - Connector

final class MultiModalProjector: Module {
    let inputDim: Int
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: LocateAnythingConfiguration) {
        let vitHidden = config.visionConfig.hiddenSize
        let merge = config.visionConfig.mergeKernelSize
        self.inputDim = vitHidden * merge[0] * merge[1]           // 1152 * 2 * 2 = 4608
        let llmHidden = config.textConfig.hiddenSize
        self._layerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: inputDim), key: "layer_norm")
        self._linear1 = ModuleInfo(wrappedValue: Linear(inputDim, llmHidden), key: "linear_1")
        self._linear2 = ModuleInfo(wrappedValue: Linear(llmHidden, llmHidden), key: "linear_2")
        super.init()
    }

    /// `imageFeatures`: [Σ N_merged, 4, vitHidden]. Returns [Σ N_merged, llmHidden].
    func callAsFunction(_ imageFeatures: MLXArray) -> MLXArray {
        var h = imageFeatures.reshaped([-1, inputDim])
        h = layerNorm(h)
        h = linear1(h)
        h = gelu(h)                      // nn.GELU() == exact (erf)
        h = linear2(h)
        return h
    }
}

// MARK: - Top-level model

public final class LocateAnythingModel: Module {
    public let config: LocateAnythingConfiguration
    let imageTokenIndex: Int

    @ModuleInfo(key: "vision_tower") var visionTower: VisionModel
    @ModuleInfo(key: "language_model") var languageModel: LanguageModel
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: MultiModalProjector

    public init(_ config: LocateAnythingConfiguration) {
        self.config = config
        self.imageTokenIndex = config.imageTokenIndex
        self._visionTower = ModuleInfo(wrappedValue: VisionModel(config: config.visionConfig), key: "vision_tower")
        self._languageModel = ModuleInfo(wrappedValue: LanguageModel(config.textConfig), key: "language_model")
        self._multiModalProjector = ModuleInfo(wrappedValue: MultiModalProjector(config), key: "multi_modal_projector")
        super.init()
    }

    /// One `KVCacheSimple` per decoder layer.
    public func newCache() -> [KVCache] {
        (0 ..< config.textConfig.numHiddenLayers).map { _ in KVCacheSimple() }
    }

    /// Plain token embedding (used per-step during decode).
    public func embed(_ ids: MLXArray) -> MLXArray {
        languageModel.model.embedTokens(ids)
    }

    /// Vision tower only (parity introspection). `pixelValues`: [P, C, patch, patch].
    public func visionForward(pixelValues: MLXArray, grids: [(Int, Int)]) -> MLXArray {
        visionTower(pixelValues.transposed(0, 2, 3, 1), grids: grids)
    }

    /// Connector only (parity introspection). `visionOut`: [ΣN, 4, vit].
    public func connectorForward(_ visionOut: MLXArray) -> MLXArray {
        multiModalProjector(visionOut)
    }

    /// Run the first `k` decoder layers on `inputsEmbeds` (no final norm/head).
    /// Bisection helper for localizing LM numerical divergence.
    public func debugHiddenAfterLayers(inputsEmbeds: MLXArray, k: Int) -> MLXArray {
        var h = inputsEmbeds
        for i in 0 ..< k {
            h = languageModel.model.layers[i](h, mask: .causal, cache: nil, positionIds: nil)
        }
        return h
    }

    /// Merge visual features into the text embeddings at <IMG_CONTEXT> positions.
    /// `pixelValues`: patchified pixels [P, C, patch, patch] (NCHW), or nil for text-only.
    /// `grids`: (h, w) patch grid per image.
    public func inputEmbeddings(
        inputIds: MLXArray, pixelValues: MLXArray?, grids: [(Int, Int)]
    ) -> MLXArray {
        var embeds = embed(inputIds)                                  // [1, L, D]
        guard let pixelValues else { return embeds }

        // NCHW -> NHWC for the MoonViT conv.
        let hidden = visionTower(pixelValues.transposed(0, 2, 3, 1), grids: grids)  // [ΣN, 4, vit]
        let feats = multiModalProjector(hidden)                       // [ΣN, D]

        let imageMask = inputIds .== imageTokenIndex                  // [1, L] bool
        let maskFlat = imageMask.reshaped([-1])                       // [L]
        let cumsum = MLX.cumsum(maskFlat.asType(.int32), axis: 0) - 1 // running feature index
        let featIdx = `where`(maskFlat, cumsum, MLXArray(Int32(0))).reshaped(inputIds.shape)
        let gathered = take(feats, featIdx, axis: 0)                  // [1, L, D]
        embeds = `where`(imageMask.expandedDimensions(axis: -1), gathered, embeds)
        return embeds
    }

    /// Run the Qwen2 decoder over precomputed embeddings; returns logits [1, L, V].
    public func languageLogits(
        inputsEmbeds: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: [KVCache]?,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        languageModel(
            inputs: nil, inputsEmbeds: inputsEmbeds, mask: mask, cache: cache, positionIds: positionIds)
    }

    /// General decoder forward over token ids OR embeddings (used by PBD).
    public func languageForward(
        inputs: MLXArray?,
        inputsEmbeds: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: [KVCache],
        positionIds: MLXArray?
    ) -> MLXArray {
        languageModel(inputs: inputs, inputsEmbeds: inputsEmbeds, mask: mask, cache: cache, positionIds: positionIds)
    }
}
