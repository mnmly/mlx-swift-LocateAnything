import Foundation
import MLX
import MLXNN

// Weight loading + selective quantization.
//
// The mlx-community/LocateAnything-3B-8bit checkpoint stores keys already in the
// final module-tree layout (vision_tower.*, language_model.model.*,
// multi_modal_projector.*), the connector patch conv already in OHWI, and no
// lm_head (tied). Only the language model carries .scales/.biases, so we quantize
// exactly the modules that have a matching .scales weight — mirroring
// MLXLMCommon.loadWeights.

public enum LocateAnythingWeights {

    static func isConvOHWI(_ shape: [Int]) -> Bool {
        guard shape.count == 4 else { return false }
        let out = shape[0], kH = shape[1], kW = shape[2]
        return out >= kH && out >= kW && kH == kW
    }

    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (k, v) in weights {
            if k.contains("position_ids") { continue }
            if k.contains("rotary_emb.inv_freq") { continue }
            if k == "language_model.lm_head.weight" { continue }   // tied
            if k.hasSuffix("patch_embed.proj.weight") && !isConvOHWI(v.shape) {
                // PyTorch OIHW -> MLX OHWI (mlx-community weights are already OHWI).
                out[k] = v.transposed(0, 2, 3, 1)
                continue
            }
            out[k] = v
        }
        return out
    }

    /// Load all *.safetensors under `directory` into `model`, quantizing LM modules.
    public static func load(
        model: LocateAnythingModel,
        directory: URL,
        config: LocateAnythingConfiguration
    ) throws {
        let files = try FileManager.default
            .contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        guard !files.isEmpty else {
            throw LocateAnythingError.weightsNotFound(directory.path)
        }

        var weights: [String: MLXArray] = [:]
        for file in files {
            let (w, _) = try loadArraysAndMetadata(url: file)
            for (k, v) in w { weights[k] = v }
        }
        let sanitized = sanitize(weights)

        if let q = config.quantization {
            quantize(model: model) { path, _ in
                if sanitized["\(path).scales"] != nil {
                    return (groupSize: q.groupSize, bits: q.bits, mode: .affine)
                }
                return nil
            }
        }

        try model.update(
            parameters: ModuleParameters.unflattened(sanitized),
            verify: [.noUnusedKeys])
        eval(model)
    }
}
