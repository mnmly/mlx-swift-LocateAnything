import Foundation
import MLX
import MLXNN

// Ported from mlx_vlm/models/locateanything/vision.py — the MoonViT (Kimi-VL
// lineage) vision tower.
//
// Two porting notes vs. the Python source:
//  1. The 2D axial RoPE in Python uses mx.complex64 arithmetic. mlx-swift's
//     complex support is limited, so we reformulate it with real cos/sin tables.
//     Pairing is adjacent (GPT-J style): pair j = (x[2j], x[2j+1]); frequencies
//     interleave x-axis and y-axis (x0, y0, x1, y1, ...).
//  2. Patch grid shapes (h, w) per image are kept host-side as `[(Int, Int)]`
//     (the `_grid_shapes` from the processor) and drive every reshape / split.

let visionLayerNormEps: Float = 1e-5

// MARK: - Bicubic resize (pos-emb interpolation)

/// Bicubic resize of an NHWC tensor to an exact target size, using the exact
/// PyTorch-style kernel (see MetalBicubic) so the pos-emb interpolation matches
/// the Python reference bit-for-bit.
func bicubicResizeNHWC(_ x: MLXArray, _ outH: Int, _ outW: Int) -> MLXArray {
    let inH = x.dim(1)
    let inW = x.dim(2)
    if inH == outH && inW == outW { return x }
    let nchw = x.transposed(0, 3, 1, 2)                       // [B, C, H, W]
    let interp = MetalBicubic.interpolate(nchw, outH: outH, outW: outW)
    return interp.transposed(0, 2, 3, 1)                      // [B, outH, outW, C]
}

// MARK: - Learnable 2D interpolated position embedding

final class Learnable2DInterpPosEmb: Module {
    let height: Int
    let width: Int
    let dim: Int
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(height: Int, width: Int, dim: Int) {
        self.height = height
        self.width = width
        self.dim = dim
        self._weight = ParameterInfo(wrappedValue: MLXArray.ones([height, width, dim]), key: "weight")
        super.init()
    }

    /// `x`: [Σ h*w, dim]. Adds a (possibly interpolated) position embedding per image.
    func callAsFunction(_ x: MLXArray, grids: [(Int, Int)]) -> MLXArray {
        var posEmbs: [MLXArray] = []
        posEmbs.reserveCapacity(grids.count)
        for (h, w) in grids {
            if h == height && w == width {
                posEmbs.append(weight.reshaped([h * w, dim]))
            } else {
                // weight is [H, W, C] (NHWC-ready); add batch, resize, drop batch.
                let resized = bicubicResizeNHWC(weight.expandedDimensions(axis: 0), h, w)
                posEmbs.append(resized.reshaped([h * w, dim]))
            }
        }
        let pos = concatenated(posEmbs, axis: 0).asType(x.dtype)
        return x + pos
    }
}

// MARK: - Patch embedding

final class PatchEmbed: Module {
    let patchSize: Int
    let embedDim: Int
    @ModuleInfo(key: "proj") var proj: Conv2d
    @ModuleInfo(key: "pos_emb") var posEmb: Learnable2DInterpPosEmb

    init(config: VisionConfiguration) {
        self.patchSize = config.patchSize
        self.embedDim = config.embedDim
        self._proj = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: config.numChannels,
                outputChannels: config.embedDim,
                kernelSize: .init(config.patchSize),
                stride: .init(config.patchSize),
                bias: true),
            key: "proj")
        self._posEmb = ModuleInfo(
            wrappedValue: Learnable2DInterpPosEmb(
                height: config.initPosEmbHeight,
                width: config.initPosEmbWidth,
                dim: config.embedDim),
            key: "pos_emb")
        super.init()
    }

    /// `hidden`: patchified pixels in NHWC layout [P, patch, patch, C].
    func callAsFunction(_ hidden: MLXArray, grids: [(Int, Int)]) -> MLXArray {
        // Conv over each 14×14 patch collapses it to a single [1,1,embedDim].
        var h = proj(hidden)                       // [P, 1, 1, embedDim]
        h = h.reshaped([h.dim(0), embedDim])       // [P, embedDim]
        h = posEmb(h, grids: grids)
        return h
    }
}

// MARK: - 2D axial RoPE (real reformulation)

/// Build cos/sin tables of shape [Σ h*w, headDim/2] for the concatenated images.
func visionRopeCosSin(grids: [(Int, Int)], headDim: Int, theta: Float = 10000) -> (MLXArray, MLXArray) {
    // dim_range = arange(0, headDim, 4)[:headDim//4]
    let quarter = headDim / 4
    let dimRange = MLXArray(stride(from: 0, to: 4 * quarter, by: 4).map { Float($0) })  // [quarter]
    let freqs = 1.0 / MLX.pow(MLXArray(theta), dimRange / Float(headDim))                // [quarter]

    var cosList: [MLXArray] = []
    var sinList: [MLXArray] = []
    for (h, w) in grids {
        let n = h * w
        let flat = MLXArray(0 ..< n).asType(.float32)
        let xpos = flat % Float(w)                 // column index
        let ypos = MLX.floor(flat / Float(w))      // row index
        let xfreqs = MLX.outer(xpos, freqs)        // [n, quarter]
        let yfreqs = MLX.outer(ypos, freqs)        // [n, quarter]
        // Interleave x/y along last axis -> [n, headDim/2]: (x0,y0,x1,y1,...).
        let cosI = stacked([MLX.cos(xfreqs), MLX.cos(yfreqs)], axis: -1).reshaped([n, headDim / 2])
        let sinI = stacked([MLX.sin(xfreqs), MLX.sin(yfreqs)], axis: -1).reshaped([n, headDim / 2])
        cosList.append(cosI)
        sinList.append(sinI)
    }
    return (concatenated(cosList, axis: 0), concatenated(sinList, axis: 0))
}

/// Apply 2D RoPE to `x` of shape [seq, heads, headDim]. cos/sin: [seq, headDim/2].
func applyVisionRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let seq = x.dim(0)
    let heads = x.dim(1)
    let hd = x.dim(2)
    let half = hd / 2
    let xf = x.asType(.float32).reshaped([seq, heads, half, 2])
    let a = xf[.ellipsis, 0]                        // [seq, heads, half]
    let b = xf[.ellipsis, 1]
    let cosE = cos.reshaped([seq, 1, half])
    let sinE = sin.reshaped([seq, 1, half])
    let o0 = a * cosE - b * sinE
    let o1 = a * sinE + b * cosE
    let out = stacked([o0, o1], axis: -1).reshaped([seq, heads, hd])
    return out.asType(x.dtype)
}

// MARK: - Block-diagonal attention mask (multi-image)

/// Boolean mask [1, 1, seq, seq]; true where two patches belong to the same image.
func visionBlockMask(grids: [(Int, Int)], seqLen: Int) -> MLXArray {
    var blockIds = [Int32](repeating: 0, count: seqLen)
    var idx = 0
    for (i, g) in grids.enumerated() {
        let len = g.0 * g.1
        for _ in 0 ..< len { blockIds[idx] = Int32(i); idx += 1 }
    }
    let ids = MLXArray(blockIds, [seqLen])
    let mask = ids.expandedDimensions(axis: 1) .== ids.expandedDimensions(axis: 0)  // [seq, seq]
    return mask.reshaped([1, 1, seqLen, seqLen])
}

// MARK: - Attention

final class VisionAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    @ModuleInfo(key: "wqkv") var wqkv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(dim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = 1.0 / Float(dim / numHeads).squareRoot()
        self._wqkv = ModuleInfo(wrappedValue: Linear(dim, dim * 3, bias: true), key: "wqkv")
        self._wo = ModuleInfo(wrappedValue: Linear(dim, dim, bias: true), key: "wo")
        super.init()
    }

    /// `x`: [seq, dim]. cos/sin: [seq, headDim/2]. mask: optional [1,1,seq,seq] bool.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?) -> MLXArray {
        let seq = x.dim(0)
        let qkv = wqkv(x).reshaped([seq, 3, numHeads, headDim])
        var q = qkv[0..., 0]                         // [seq, heads, headDim]
        var k = qkv[0..., 1]
        let v = qkv[0..., 2]

        q = applyVisionRope(q, cos: cos, sin: sin)
        k = applyVisionRope(k, cos: cos, sin: sin)

        // -> [1, heads, seq, headDim]
        let qh = q.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let kh = k.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let vh = v.transposed(1, 0, 2).expandedDimensions(axis: 0)

        var out = MLXFast.scaledDotProductAttention(
            queries: qh, keys: kh, values: vh, scale: scale, mask: mask)
        out = out.transposed(0, 2, 1, 3).reshaped([seq, numHeads * headDim])
        return wo(out)
    }
}

// MARK: - MLP

final class VisionMLP: Module {
    @ModuleInfo(key: "fc0") var fc0: Linear
    @ModuleInfo(key: "fc1") var fc1: Linear

    init(dim: Int, hiddenDim: Int) {
        self._fc0 = ModuleInfo(wrappedValue: Linear(dim, hiddenDim, bias: true), key: "fc0")
        self._fc1 = ModuleInfo(wrappedValue: Linear(hiddenDim, dim, bias: true), key: "fc1")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // nn.GELU(approx="precise") == tanh approximation.
        fc1(geluApproximate(fc0(x)))
    }
}

// MARK: - Transformer block

final class VisionBlock: Module {
    @ModuleInfo(key: "norm0") var norm0: LayerNorm
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: VisionAttention
    @ModuleInfo(key: "mlp") var mlp: VisionMLP

    init(config: VisionConfiguration) {
        self._norm0 = ModuleInfo(wrappedValue: LayerNorm(dimensions: config.embedDim, eps: visionLayerNormEps), key: "norm0")
        self._norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: config.embedDim, eps: visionLayerNormEps), key: "norm1")
        self._attn = ModuleInfo(wrappedValue: VisionAttention(dim: config.embedDim, numHeads: config.numHeads), key: "attn")
        self._mlp = ModuleInfo(wrappedValue: VisionMLP(dim: config.embedDim, hiddenDim: config.intermediateSize), key: "mlp")
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + attn(norm0(x), cos: cos, sin: sin, mask: mask)
        h = h + mlp(norm1(h))
        return h
    }
}

// MARK: - Patch merger

/// Per-image 2×2 merge. `x`: [Σ h*w, dim] -> [Σ (h/2)*(w/2), 4, dim].
func patchMerger(_ x: MLXArray, grids: [(Int, Int)], mergeKernel: [Int]) -> MLXArray {
    let dModel = x.dim(-1)
    let kh = mergeKernel[0]
    let kw = mergeKernel[1]
    var offset = 0
    var outputs: [MLXArray] = []
    for (h, w) in grids {
        let len = h * w
        let seq = x[offset ..< (offset + len)]
        offset += len
        let nh = h / kh
        let nw = w / kw
        var r = seq.reshaped([nh, kh, nw, kw, dModel])
        r = r.transposed(0, 2, 1, 3, 4)
        r = r.reshaped([nh * nw, kh * kw, dModel])
        outputs.append(r)
    }
    return concatenated(outputs, axis: 0)
}

// MARK: - Vision tower

public final class VisionModel: Module {
    let config: VisionConfiguration
    let headDim: Int
    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
    @ModuleInfo(key: "blocks") var blocks: [VisionBlock]
    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: LayerNorm

    init(config: VisionConfiguration) {
        self.config = config
        self.headDim = config.embedDim / config.numHeads
        self._patchEmbed = ModuleInfo(wrappedValue: PatchEmbed(config: config), key: "patch_embed")
        self._blocks = ModuleInfo(
            wrappedValue: (0 ..< config.depth).map { _ in VisionBlock(config: config) },
            key: "blocks")
        self._finalLayerNorm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: visionLayerNormEps),
            key: "final_layernorm")
        super.init()
    }

    /// `hidden`: patchified pixels NHWC [P, patch, patch, C]. `grids`: (h, w) per image.
    /// Returns merged patches [Σ (h/2)*(w/2), 4, embedDim].
    public func callAsFunction(_ hidden: MLXArray, grids: [(Int, Int)]) -> MLXArray {
        var h = patchEmbed(hidden, grids: grids)
        let (cos, sin) = visionRopeCosSin(grids: grids, headDim: headDim)
        let mask: MLXArray? = grids.count > 1 ? visionBlockMask(grids: grids, seqLen: h.dim(0)) : nil

        for blk in blocks {
            h = blk(h, cos: cos, sin: sin, mask: mask)
        }
        h = finalLayerNorm(h)
        return patchMerger(h, grids: grids, mergeKernel: config.mergeKernelSize)
    }
}
