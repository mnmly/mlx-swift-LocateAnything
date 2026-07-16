import Foundation
import MLX

// Per-stage numerical parity against fixtures dumped from the Python mlx-vlm
// reference. Each stage is fed the REFERENCE input for that stage (bisection),
// so an error localizes to the stage that produced it rather than compounding.

public struct ParityStage: Sendable {
    public let name: String
    public let relError: Float     // mean|a-b| / mean|b|
    public let maxAbs: Float
    public let refMean: Float
    public let shape: [Int]
}

public enum LocateAnythingParity {

    static func stats(_ a: MLXArray, _ b: MLXArray) -> (rel: Float, maxAbs: Float, refMean: Float) {
        let af = a.asType(.float32)
        let bf = b.asType(.float32)
        let diff = MLX.abs(af - bf)
        MLX.eval(diff)
        let maxAbs = diff.max().item(Float.self)
        let refMean = MLX.abs(bf).mean().item(Float.self)
        let rel = diff.mean().item(Float.self) / Swift.max(refMean, 1e-9)
        return (rel, maxAbs, refMean)
    }

    /// Run vision / connector / prefill-logits parity against a fixtures file.
    /// `computeDType` controls the pixel/vision compute precision; the Python
    /// oracle ran with float32 pixel_values (weights bf16 promote to float32).
    public static func run(
        session: LocateAnythingSession,
        fixtures: URL,
        computeDType: DType = .float32
    ) throws -> [ParityStage] {
        let f = try loadArrays(url: fixtures)
        guard let pixelValues = f["pixel_values"],
              let grid = f["grid"],
              let visionOutRef = f["vision_out"],
              let featsRef = f["feats"],
              let inputIds = f["input_ids"],
              let lastLogitsRef = f["last_logits"]
        else { throw LocateAnythingError.weightsNotFound(fixtures.path) }

        // grid: [numImages, 2] int32 -> [(Int, Int)]
        MLX.eval(grid)
        let gridInts = grid.asType(.int32).asArray(Int32.self)
        var grids: [(Int, Int)] = []
        for i in stride(from: 0, to: gridInts.count, by: 2) {
            grids.append((Int(gridInts[i]), Int(gridInts[i + 1])))
        }

        var stages: [ParityStage] = []

        // Stage 1: vision tower (input = reference pixel_values).
        let pv = pixelValues.asType(computeDType)
        let visionOut = session.model.visionForward(pixelValues: pv, grids: grids)
        MLX.eval(visionOut)
        let s1 = stats(visionOut, visionOutRef)
        stages.append(ParityStage(name: "vision_out", relError: s1.rel, maxAbs: s1.maxAbs, refMean: s1.refMean, shape: visionOut.shape))

        // Stage 2: connector (input = reference vision_out, isolates connector).
        let feats = session.model.connectorForward(visionOutRef.asType(computeDType))
        MLX.eval(feats)
        let s2 = stats(feats, featsRef)
        stages.append(ParityStage(name: "connector_feats", relError: s2.rel, maxAbs: s2.maxAbs, refMean: s2.refMean, shape: feats.shape))

        // Stage 3: prefill logits (input = reference input_ids + pixel_values).
        let cache = session.model.newCache()
        let embeds = session.model.inputEmbeddings(inputIds: inputIds, pixelValues: pv, grids: grids)
        let logits = session.model.languageLogits(inputsEmbeds: embeds, mask: .causal, cache: cache)
        let l = logits.dim(1)
        let lastLogits = logits[0..., (l - 1) ..< l, 0...].reshaped([1, -1])
        MLX.eval(lastLogits)
        let s3 = stats(lastLogits, lastLogitsRef)
        stages.append(ParityStage(name: "prefill_last_logits", relError: s3.rel, maxAbs: s3.maxAbs, refMean: s3.refMean, shape: lastLogits.shape))

        // Stage 3b: argmax agreement of the first token.
        let refArg = argMax(lastLogitsRef, axis: -1).item(Int.self)
        let swiftArg = argMax(lastLogits, axis: -1).item(Int.self)
        stages.append(ParityStage(
            name: "first_token_argmax(ref=\(refArg),swift=\(swiftArg))",
            relError: refArg == swiftArg ? 0 : 1, maxAbs: 0, refMean: 0, shape: []))

        return stages
    }

    /// Isolate the language model: feed reference merged embeddings and compare logits.
    public static func lmIso(session: LocateAnythingSession, fixtures: URL) throws -> [ParityStage] {
        let f = try loadArrays(url: fixtures)
        guard let embeds = f["merged_embeds"], let logitsRef = f["lm_last_logits"] else {
            throw LocateAnythingError.weightsNotFound(fixtures.path)
        }
        let cache = session.model.newCache()
        let logits = session.model.languageLogits(inputsEmbeds: embeds, mask: .causal, cache: cache)
        let l = logits.dim(1)
        let last = logits[0..., (l - 1) ..< l, 0...].reshaped([1, -1])
        MLX.eval(last)
        let s = stats(last, logitsRef)
        let refArg = argMax(logitsRef, axis: -1).item(Int.self)
        let swiftArg = argMax(last, axis: -1).item(Int.self)
        return [
            ParityStage(name: "lm_logits_from_ref_embeds", relError: s.rel, maxAbs: s.maxAbs, refMean: s.refMean, shape: last.shape),
            ParityStage(name: "lm_argmax(ref=\(refArg),swift=\(swiftArg))", relError: refArg == swiftArg ? 0 : 1, maxAbs: 0, refMean: 0, shape: []),
        ]
    }

    /// Bisect the decoder: compare hidden state after k layers (k = 1,9,18,27,36).
    public static func lmBisect(session: LocateAnythingSession, embeds embedsURL: URL, layers layersURL: URL) throws -> [ParityStage] {
        let e = try loadArrays(url: embedsURL)
        let ls = try loadArrays(url: layersURL)
        guard let embeds = e["merged_embeds"] else { throw LocateAnythingError.weightsNotFound(embedsURL.path) }
        var stages: [ParityStage] = []
        for k in [1, 9, 18, 27, 36] {
            guard let ref = ls["h\(k)"] else { continue }
            let h = session.model.debugHiddenAfterLayers(inputsEmbeds: embeds, k: k)
            MLX.eval(h)
            let s = stats(h, ref)
            stages.append(ParityStage(name: "hidden_after_\(k)_layers", relError: s.rel, maxAbs: s.maxAbs, refMean: s.refMean, shape: h.shape))
        }
        return stages
    }

    /// Isolate the vision sub-components: pos-emb bicubic interpolation and 2D rope.
    public static func visionIso(fixtures: URL, gridH: Int = 72, gridW: Int = 138) throws -> [ParityStage] {
        let f = try loadArrays(url: fixtures)
        guard let raw = f["pos_emb_raw"], let peRef = f["pos_emb_interp"],
              let cosRef = f["rope_cos"], let sinRef = f["rope_sin"]
        else { throw LocateAnythingError.weightsNotFound(fixtures.path) }

        var stages: [ParityStage] = []

        // Pos-emb bicubic interpolation.
        let resized = bicubicResizeNHWC(raw.expandedDimensions(axis: 0), gridH, gridW)
            .reshaped([gridH * gridW, raw.dim(2)])
        MLX.eval(resized)
        let sp = stats(resized, peRef)
        stages.append(ParityStage(name: "pos_emb_interp", relError: sp.rel, maxAbs: sp.maxAbs, refMean: sp.refMean, shape: resized.shape))

        // 2D rope cos/sin.
        let (cos, sin) = visionRopeCosSin(grids: [(gridH, gridW)], headDim: 72)
        MLX.eval(cos, sin)
        let sc = stats(cos, cosRef)
        let ss = stats(sin, sinRef)
        stages.append(ParityStage(name: "rope_cos", relError: sc.rel, maxAbs: sc.maxAbs, refMean: sc.refMean, shape: cos.shape))
        stages.append(ParityStage(name: "rope_sin", relError: ss.rel, maxAbs: ss.maxAbs, refMean: ss.refMean, shape: sin.shape))

        return stages
    }
}
