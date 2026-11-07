import CoreGraphics
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import Tokenizers

// High-level pipeline mirroring locateanything_worker.py: predict(image,
// question, visualPromptBox?) -> detected boxes. Phase 1 uses standard
// autoregressive (greedy) decoding of the box grammar; Parallel Box Decoding is
// added in Phase 2.

/// One detection: a bounding box (or point) with an optional referring-expression label.
public struct DetectedBox: Sendable {
    /// The referring phrase this detection belongs to (e.g. "car"), if the model emitted one.
    public let label: String?
    /// Normalized `[x1, y1, x2, y2]` (or `[x, y]` for a point) in `0...1`.
    public let box: [Float]
    /// Pixel coordinates in the original image (`[x1, y1, x2, y2]` or `[x, y]`).
    public let boxPixels: [Float]
    /// `true` when this is a point rather than a box.
    public let isPoint: Bool
}

/// Sampling controls for decoding. Defaults to greedy (deterministic: same image +
/// prompt → same boxes).
///
/// `temperature`, `topP` and `topK` only affect the ``PBDMode/ar`` path — the
/// Parallel Box Decoding modes select coordinates by threshold, deterministically.
/// `repetitionPenalty` applies to every mode.
public struct SamplingParameters: Sendable {
    /// `0` = greedy (argmax). Higher = flatter distribution / more random.
    public var temperature: Float
    /// Nucleus sampling: keep the smallest set of tokens summing to `topP`. `1` = off.
    public var topP: Float
    /// Keep only the `topK` most-likely tokens. `0` = off.
    public var topK: Int
    /// CTRL-style penalty on every token already in the sequence (prompt included),
    /// as in NVIDIA's `generate_utils.apply_repetition_penalty`. `1` = off (default).
    ///
    /// NVIDIA's reference settings use `1.1` alongside `temperature: 0.7`. With
    /// greedy decoding it nudges legitimately repeated coordinates (e.g. a
    /// full-frame `<0><0><998><998>` box), so it is off by default.
    public var repetitionPenalty: Float

    /// - Parameters:
    ///   - temperature: `0` for greedy decoding.
    ///   - topP: nucleus threshold; `1` disables it.
    ///   - topK: top-k cutoff; `0` disables it.
    ///   - repetitionPenalty: penalty on already-seen tokens; `1` disables it.
    public init(temperature: Float = 0, topP: Float = 1.0, topK: Int = 0, repetitionPenalty: Float = 1.0) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
    }

    /// Deterministic argmax decoding.
    public static let greedy = SamplingParameters()

    public var isGreedy: Bool { temperature <= 0 }
}

/// The result of a ``LocateAnythingSession/predict(image:question:maxNewTokens:mode:visualPromptBox:visualPromptBoxFormat:)`` call.
public struct LocateAnythingResult: Sendable {
    /// The raw decoded box grammar, e.g. `<ref>car</ref><box><0><111><74><213></box>`.
    public let text: String
    /// The parsed detections.
    public let boxes: [DetectedBox]
    /// Number of tokens generated.
    public let tokenCount: Int
}

/// The single library-side driver both the CLI and the SwiftUI demo consume.
///
/// `@unchecked Sendable` under a **single-writer** contract: at most one caller
/// drives `predict(...)` at a time. The CLI is single-threaded; the GUI runs it
/// on one `Task.detached` while the UI only reads results afterward. Sharing a
/// session across two concurrent `predict` callers is unsupported — build a
/// serializing wrapper for that case. Inputs are materialized (`CGImage`), never
/// file paths, so a sandboxed frontend works identically to an unsandboxed CLI.
public final class LocateAnythingSession: @unchecked Sendable {
    public let model: LocateAnythingModel
    public let config: LocateAnythingConfiguration
    let tokenizer: any MLXLMCommon.Tokenizer
    let mergeLength: Int

    init(model: LocateAnythingModel, config: LocateAnythingConfiguration, tokenizer: any MLXLMCommon.Tokenizer) {
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.mergeLength = config.visionConfig.mergeKernelSize[0] * config.visionConfig.mergeKernelSize[1]
    }

    // MARK: - Loading

    public static let defaultRepoID = "mlx-community/LocateAnything-3B-8bit"

    /// Download the model snapshot into the Hugging Face cache (`~/.cache/huggingface`),
    /// returning its local dir. `progress` reports 0…1 on the main actor.
    public static func download(
        repoId: String = defaultRepoID,
        progress: (@MainActor @Sendable (Double) -> Void)? = nil
    ) async throws -> URL {
        guard let repo = Repo.ID(rawValue: repoId) else {
            throw LocateAnythingError.invalidRepoID(repoId)
        }
        let client = HubClient()
        let handler: (@MainActor @Sendable (Progress) -> Void)?
        if let progress {
            handler = { @MainActor @Sendable p in progress(p.fractionCompleted) }
        } else {
            handler = nil
        }
        return try await client.downloadSnapshot(of: repo, revision: "main", progressHandler: handler)
    }

    /// Locate an already-downloaded snapshot in the HF cache, if present.
    /// Honors `HF_HUB_CACHE` / `HF_HOME`, else `~/.cache/huggingface/hub`.
    public static func cachedDirectory(repoId: String = defaultRepoID) -> URL? {
        let fm = FileManager.default
        let env = ProcessInfo.processInfo.environment
        let hubRoot: URL
        if let hc = env["HF_HUB_CACHE"] {
            hubRoot = URL(fileURLWithPath: hc)
        } else if let home = env["HF_HOME"] {
            hubRoot = URL(fileURLWithPath: home).appendingPathComponent("hub")
        } else {
            hubRoot = fm.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache/huggingface/hub")
        }
        let folder = "models--" + repoId.replacingOccurrences(of: "/", with: "--")
        let snapshots = hubRoot.appendingPathComponent(folder).appendingPathComponent("snapshots")
        guard let entries = try? fm.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey])
        else { return nil }
        // Newest snapshot that actually contains config.json.
        let valid = entries.filter { fm.fileExists(atPath: $0.appendingPathComponent("config.json").path) }
        return valid.sorted {
            let a = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
            let b = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
            return a > b
        }.first
    }

    /// Load model + tokenizer from a local snapshot directory (contains
    /// `config.json`, `model.safetensors`, and tokenizer files). Synthesizes a
    /// `tokenizer.json` in Swift if the checkpoint ships only `vocab.json`/`merges.txt`.
    /// - Parameter directory: the model snapshot folder.
    public static func load(directory: URL) async throws -> LocateAnythingSession {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(LocateAnythingConfiguration.self, from: configData)

        let model = LocateAnythingModel(config)
        try LocateAnythingWeights.load(model: model, directory: directory, config: config)

        // Legacy checkpoints ship vocab.json + merges.txt without tokenizer.json;
        // synthesize it in Swift so swift-transformers can load it.
        try LocateAnythingTokenizerSetup.ensureTokenizerJSON(in: directory)
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: directory)
        return LocateAnythingSession(model: model, config: config, tokenizer: tokenizer)
    }

    /// Convenience: download (if needed) then load.
    /// - Parameter repoId: the Hugging Face repo to fetch (default: the 8-bit release).
    public static func fromPretrained(repoId: String = defaultRepoID) async throws -> LocateAnythingSession {
        let dir = try await download(repoId: repoId)
        return try await load(directory: dir)
    }

    // MARK: - Inference

    /// Ground the instruction in the image and return the detected boxes.
    /// - Parameters:
    ///   - image: the input image (already decoded — see ``LocateAnythingImageProcessor/baked(_:)``).
    ///   - question: a LocateAnything instruction, e.g. `"Locate all the instances that matches the following description: car."`.
    ///   - maxNewTokens: generation cap.
    ///   - mode: decoding strategy — ``PBDMode/hybrid`` (default) uses Parallel Box Decoding.
    ///   - visualPromptBox: optional box to crop as a visual prompt (released weights don't support this yet).
    ///   - visualPromptBoxFormat: how `visualPromptBox` coordinates are expressed.
    public func predict(
        image: CGImage,
        question: String,
        maxNewTokens: Int = 512,
        mode: PBDMode = .hybrid,
        sampling: SamplingParameters = .greedy,
        visualPromptBox: [Float]? = nil,
        visualPromptBoxFormat: BoxFormat = .normalized1000
    ) throws -> LocateAnythingResult {
        var processed: [ProcessedImage] = [try LocateAnythingImageProcessor.preprocess(image)]
        var numImages = 1
        if let vp = visualPromptBox {
            let crop = cropVisualPrompt(image, box: vp, format: visualPromptBoxFormat)
            processed.append(try LocateAnythingImageProcessor.preprocess(crop))
            numImages = 2
        }

        let (pixelValues, grids) = LocateAnythingImageProcessor.batch(processed)
        let pv = pixelValues.asType(.bfloat16)

        let prompt = LocateAnythingPrompt.chatML(question: question, numImages: numImages)
        let expanded = LocateAnythingPrompt.expandPlaceholders(prompt, grids: grids, mergeLength: mergeLength)
        let ids = tokenizer.encode(text: expanded, addSpecialTokens: false)

        return runGeneration(
            ids: ids, pixelValues: pv, grids: grids,
            displayWidth: image.width, displayHeight: image.height,
            maxNewTokens: maxNewTokens, mode: mode, sampling: sampling)
    }

    /// Ground an instruction across sampled video frames.
    ///
    /// The frontend extracts the frames (see ``LocateAnythingVideo/plan(totalFrames:videoFPS:)``)
    /// and passes them as materialized images. Frames are globally downscaled to the
    /// reference pixel budget, preprocessed per-frame, and grounded together.
    /// - Parameters:
    ///   - frames: the sampled frames, in order.
    ///   - timestamps: each frame's time in seconds (used in the prompt).
    ///   - sampledFPS: effective sample rate, reported in the prompt.
    ///   - displayWidth: width used to map boxes to pixels (defaults to the frame's).
    ///   - displayHeight: height used to map boxes to pixels (defaults to the frame's).
    public func predictVideo(
        frames: [CGImage],
        timestamps: [Double],
        sampledFPS: Float,
        displayWidth: Int? = nil,
        displayHeight: Int? = nil,
        question: String,
        maxNewTokens: Int = 512,
        mode: PBDMode = .hybrid,
        sampling: SamplingParameters = .greedy
    ) throws -> LocateAnythingResult {
        guard let first = frames.first else { throw LocateAnythingError.imageDecodeFailed }

        // Global downscale so all frames together fit the reference pixel budget.
        let scale = LocateAnythingVideo.frameScale(
            frameCount: frames.count, width: first.width, height: first.height)

        var prepped: [ProcessedImage] = []
        prepped.reserveCapacity(frames.count)
        for f in frames {
            var frame = f
            if scale < 1.0 {
                let w = max(28, Int((Double(f.width) * scale).rounded()))
                let h = max(28, Int((Double(f.height) * scale).rounded()))
                frame = try LocateAnythingImageProcessor.resize(f, width: w, height: h)
            }
            prepped.append(try LocateAnythingImageProcessor.preprocess(frame))
        }

        let (pixelValues, grids) = LocateAnythingImageProcessor.batch(prepped)
        let pv = pixelValues.asType(.bfloat16)

        let prompt = LocateAnythingPrompt.videoChatML(
            question: question, frames: prepped, timestamps: timestamps,
            sampledFPS: sampledFPS, mergeLength: mergeLength)
        let ids = tokenizer.encode(text: prompt, addSpecialTokens: false)

        return runGeneration(
            ids: ids, pixelValues: pv, grids: grids,
            displayWidth: displayWidth ?? first.width, displayHeight: displayHeight ?? first.height,
            maxNewTokens: maxNewTokens, mode: mode, sampling: sampling)
    }

    /// Shared prefill + decode + parse for the image and video paths.
    private func runGeneration(
        ids: [Int], pixelValues: MLXArray, grids: [(Int, Int)],
        displayWidth: Int, displayHeight: Int,
        maxNewTokens: Int, mode: PBDMode, sampling: SamplingParameters
    ) -> LocateAnythingResult {
        let inputIds = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        let cache = model.newCache()
        let embeds = model.inputEmbeddings(inputIds: inputIds, pixelValues: pixelValues, grids: grids)

        let generated: [Int]
        if mode == .ar {
            generated = generateAR(ids: ids, embeds: embeds, cache: cache, maxNewTokens: maxNewTokens, sampling: sampling)
        } else {
            let decoder = PBDDecoder(model: model, mode: mode, repetitionPenalty: sampling.repetitionPenalty)
            generated = decoder.generate(
                prompt: ids, inputsEmbeds: embeds, cache: cache, maxTokens: maxNewTokens)
        }

        let text = tokenizer.decode(tokenIds: generated, skipSpecialTokens: false)
        let boxes = parseBoxes(generated, imageWidth: displayWidth, imageHeight: displayHeight)
        return LocateAnythingResult(text: text, boxes: boxes, tokenCount: generated.count)
    }

    /// Standard autoregressive decoding (Phase 1 baseline / PBD fallback).
    /// Greedy by default; `sampling` enables temperature / top-p / top-k.
    private func generateAR(
        ids: [Int], embeds: MLXArray, cache: [KVCache], maxNewTokens: Int,
        sampling: SamplingParameters
    ) -> [Int] {
        let sampler: any LogitSampler = sampling.isGreedy
            ? ArgMaxSampler()
            : TopPSampler(temperature: sampling.temperature, topP: sampling.topP, topK: sampling.topK)

        var seen = SeenTokens(ids)
        let penalty = sampling.repetitionPenalty

        let prefillLogits = model.languageLogits(
            inputsEmbeds: embeds, mask: ids.count > 1 ? .causal : .none, cache: cache)
        var nextId = sampleLast(prefillLogits, sampler, seen: seen, penalty: penalty)
        let eos = Set(config.eosTokenIds)
        var generated: [Int] = []
        generated.reserveCapacity(maxNewTokens)
        for _ in 0 ..< maxNewTokens {
            if eos.contains(nextId) { break }
            generated.append(nextId)
            seen.insert(nextId)
            let emb1 = model.embed(MLXArray([Int32(nextId)], [1, 1]))
            let logits = model.languageLogits(inputsEmbeds: emb1, mask: .none, cache: cache)
            nextId = sampleLast(logits, sampler, seen: seen, penalty: penalty)
        }
        return generated
    }

    private func sampleLast(
        _ logits: MLXArray, _ sampler: any LogitSampler, seen: SeenTokens, penalty: Float
    ) -> Int {
        let l = logits.dim(1)
        let last = logits[0..., (l - 1) ..< l, 0...].reshaped([1, -1])   // [1, V]
        return sampler.sample(logits: applyRepetitionPenalty(last, seen: seen, penalty: penalty)).item(Int.self)
    }

    // MARK: - Helpers

    public enum BoxFormat: Sendable {
        case normalized1000, normalized1, pixel
    }

    private func cropVisualPrompt(_ image: CGImage, box: [Float], format: BoxFormat) -> CGImage {
        guard box.count == 4 else { return image }
        let w = Float(image.width)
        let h = Float(image.height)
        var x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3]
        switch format {
        case .normalized1000:
            x1 = x1 / 1000 * w; y1 = y1 / 1000 * h; x2 = x2 / 1000 * w; y2 = y2 / 1000 * h
        case .normalized1:
            x1 *= w; y1 *= h; x2 *= w; y2 *= h
        case .pixel:
            break
        }
        let rect = CGRect(x: CGFloat(x1), y: CGFloat(y1),
                          width: CGFloat(max(1, x2 - x1)), height: CGFloat(max(1, y2 - y1)))
        return image.cropping(to: rect) ?? image
    }

    private func parseBoxes(_ tokens: [Int], imageWidth: Int, imageHeight: Int) -> [DetectedBox] {
        let boxStart = config.boxStartTokenId
        let boxEnd = config.boxEndTokenId
        let coordStart = config.coordStartTokenId
        let coordEnd = config.coordEndTokenId
        let refStart = config.refStartTokenId
        let refEnd = config.refEndTokenId
        let fw = Float(imageWidth)
        let fh = Float(imageHeight)

        var boxes: [DetectedBox] = []
        var pendingLabel: String? = nil
        var i = 0
        while i < tokens.count {
            let t = tokens[i]
            if t == refStart {
                var j = i + 1
                var labelTokens: [Int] = []
                while j < tokens.count && tokens[j] != refEnd { labelTokens.append(tokens[j]); j += 1 }
                pendingLabel = tokenizer.decode(tokenIds: labelTokens, skipSpecialTokens: true)
                i = (j < tokens.count) ? j + 1 : j
            } else if t == boxStart {
                var j = i + 1
                var coords: [Int] = []
                while j < tokens.count && tokens[j] != boxEnd {
                    let c = tokens[j]
                    if c >= coordStart && c <= coordEnd { coords.append(c - coordStart) }
                    j += 1
                }
                if coords.count >= 4 {
                    let n = coords.prefix(4).map { Float($0) / 1000.0 }
                    boxes.append(DetectedBox(
                        label: pendingLabel,
                        box: Array(n),
                        boxPixels: [n[0] * fw, n[1] * fh, n[2] * fw, n[3] * fh],
                        isPoint: false))
                } else if coords.count == 2 {
                    let n = coords.map { Float($0) / 1000.0 }
                    boxes.append(DetectedBox(
                        label: pendingLabel,
                        box: n,
                        boxPixels: [n[0] * fw, n[1] * fh],
                        isPoint: true))
                }
                pendingLabel = nil
                i = (j < tokens.count) ? j + 1 : j
            } else {
                i += 1
            }
        }
        return boxes
    }
}
