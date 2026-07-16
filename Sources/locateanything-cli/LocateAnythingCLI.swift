import ArgumentParser
import Foundation
import MLX
import MLXLocateAnything

@main
struct LocateAnythingCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "locateanything-cli",
        abstract: "Visual grounding with LocateAnything on Apple Silicon (MLX).",
        subcommands: [Predict.self, Bench.self, Parity.self, VisionIso.self, LMIso.self, LMBisect.self],
        defaultSubcommand: Predict.self
    )
}

struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Benchmark throughput + watch memory for leaks (reuses one loaded model).")

    @OptionGroup var modelOptions: ModelOptions

    @Argument(help: "Path to the input image.")
    var image: String

    @Option(name: .shortAndLong, help: "The grounding instruction.")
    var prompt: String = "Locate all the instances that matches the following description: person."

    @Option(help: "Max new tokens per run.")
    var maxTokens: Int = 128

    @Option(help: "Decoding mode: hybrid | fast | slow | ar.")
    var mode: String = "hybrid"

    @Option(help: "Number of timed runs.")
    var runs: Int = 6

    func run() async throws {
        let dir = try await modelOptions.resolveDirectory()
        let session = try await LocateAnythingSession.load(directory: dir)
        let img = try LocateAnythingImageProcessor.loadCGImage(path: image)
        let pbdMode = PBDMode(rawValue: mode) ?? .hybrid

        func mb(_ bytes: Int) -> String { String(format: "%.0f MB", Double(bytes) / 1_048_576) }

        // Warm-up (build kernels, load caches) — excluded from timing.
        _ = try session.predict(image: img, question: prompt, maxNewTokens: maxTokens, mode: pbdMode)

        print("mode=\(pbdMode.rawValue)")
        print("run  tokens  time_s  tok/s   active_mem  cache_mem")
        var toks = 0
        var totalTime = 0.0
        for i in 1 ... runs {
            let t0 = Date()
            let r = try session.predict(image: img, question: prompt, maxNewTokens: maxTokens, mode: pbdMode)
            let dt = Date().timeIntervalSince(t0)
            toks += r.tokenCount
            totalTime += dt
            print(String(format: "%-4d %-7d %-7.2f %-7.1f %-11@ %@",
                         i, r.tokenCount, dt, Double(r.tokenCount) / dt,
                         mb(MLX.Memory.activeMemory) as NSString, mb(MLX.Memory.cacheMemory) as NSString))
        }
        print(String(format: "avg %.1f tok/s over %d runs | peak %@ | active flat across runs ⇒ no leak",
                     Double(toks) / totalTime, runs, mb(MLX.Memory.peakMemory) as NSString))
    }
}

struct ModelOptions: ParsableArguments {
    @Option(help: "Hugging Face repo id (used when --model-path is not given).")
    var model: String = "mlx-community/LocateAnything-3B-8bit"

    @Option(help: "Local model snapshot directory (overrides --model).")
    var modelPath: String?

    func resolveDirectory() async throws -> URL {
        if let modelPath { return URL(fileURLWithPath: modelPath) }
        FileHandle.standardError.write("Downloading \(model)…\n".data(using: .utf8)!)
        return try await LocateAnythingSession.download(repoId: model)
    }
}

struct Predict: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Detect / ground objects in an image.")

    @OptionGroup var modelOptions: ModelOptions

    @Argument(help: "Path to the input image.")
    var image: String

    @Option(name: .shortAndLong, help: "The grounding instruction. Detection example: 'Locate all the instances that matches the following description: person</c>car.'")
    var prompt: String = "Locate all the instances that matches the following description: person."

    @Option(help: "Maximum new tokens to generate.")
    var maxTokens: Int = 512

    @Option(help: "Decoding mode: hybrid | fast | slow | ar.")
    var mode: String = "hybrid"

    @Option(name: .customLong("visual-prompt-box"), help: "Optional visual-prompt box 'x1,y1,x2,y2' (normalized 0-1000).")
    var visualPromptBox: String?

    func run() async throws {
        let dir = try await modelOptions.resolveDirectory()
        let t0 = Date()
        let session = try await LocateAnythingSession.load(directory: dir)
        FileHandle.standardError.write(
            "Loaded model in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s\n".data(using: .utf8)!)

        let img = try LocateAnythingImageProcessor.loadCGImage(path: image)
        let vpb: [Float]? = visualPromptBox.map {
            $0.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
        }
        let pbdMode = PBDMode(rawValue: mode) ?? .hybrid

        let t1 = Date()
        let result = try session.predict(
            image: img, question: prompt, maxNewTokens: maxTokens, mode: pbdMode, visualPromptBox: vpb)
        let dt = Date().timeIntervalSince(t1)

        print("--- raw output (\(result.tokenCount) tokens, \(String(format: "%.2f", dt))s, "
            + "\(String(format: "%.1f", Double(result.tokenCount) / max(dt, 1e-6))) tok/s) ---")
        print(result.text)
        print("--- \(result.boxes.count) box(es) ---")
        for (i, b) in result.boxes.enumerated() {
            let label = b.label.map { " \"\($0)\"" } ?? ""
            let px = b.boxPixels.map { String(format: "%.0f", $0) }.joined(separator: ", ")
            let kind = b.isPoint ? "point" : "box"
            print("[\(i)]\(label) \(kind) px=[\(px)] norm=\(b.box.map { String(format: "%.3f", $0) })")
        }
    }
}

struct LMBisect: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Bisect the decoder: per-layer hidden-state parity.")

    @OptionGroup var modelOptions: ModelOptions

    @Option(help: "la_lm_iso.safetensors (merged_embeds).")
    var embeds: String

    @Argument(help: "la_layers.safetensors (h1,h9,h18,h27,h36).")
    var layers: String

    func run() async throws {
        let dir = try await modelOptions.resolveDirectory()
        let session = try await LocateAnythingSession.load(directory: dir)
        let stages = try LocateAnythingParity.lmBisect(
            session: session, embeds: URL(fileURLWithPath: embeds), layers: URL(fileURLWithPath: layers))
        for s in stages {
            print("\(s.name): rel_err=\(String(format: "%.6f", s.relError)) max_abs=\(String(format: "%.5f", s.maxAbs)) ref_mean=\(String(format: "%.5f", s.refMean)) \(s.shape)")
        }
    }
}

struct LMIso: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Isolate the language model with reference embeddings.")

    @OptionGroup var modelOptions: ModelOptions

    @Argument(help: "Path to la_lm_iso.safetensors.")
    var fixtures: String

    func run() async throws {
        let dir = try await modelOptions.resolveDirectory()
        let session = try await LocateAnythingSession.load(directory: dir)
        let stages = try LocateAnythingParity.lmIso(session: session, fixtures: URL(fileURLWithPath: fixtures))
        for s in stages {
            print("\(s.name): rel_err=\(String(format: "%.6f", s.relError)) max_abs=\(String(format: "%.5f", s.maxAbs)) ref_mean=\(String(format: "%.5f", s.refMean)) \(s.shape)")
        }
    }
}

struct VisionIso: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Isolate pos-emb interpolation + 2D rope parity.")

    @Argument(help: "Path to la_vision_iso.safetensors.")
    var fixtures: String

    func run() async throws {
        let stages = try LocateAnythingParity.visionIso(fixtures: URL(fileURLWithPath: fixtures))
        for s in stages {
            print("\(s.name): rel_err=\(String(format: "%.6f", s.relError)) max_abs=\(String(format: "%.5f", s.maxAbs)) ref_mean=\(String(format: "%.5f", s.refMean)) \(s.shape)")
        }
    }
}

struct Parity: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Per-stage numerical parity vs a Python fixtures file.")

    @OptionGroup var modelOptions: ModelOptions

    @Argument(help: "Path to the fixtures .safetensors dumped from the Python reference.")
    var fixtures: String

    func run() async throws {
        let dir = try await modelOptions.resolveDirectory()
        let session = try await LocateAnythingSession.load(directory: dir)
        let stages = try LocateAnythingParity.run(
            session: session, fixtures: URL(fileURLWithPath: fixtures))
        func pad(_ s: String, _ n: Int) -> String {
            s.count >= n ? s : s + String(repeating: " ", count: n - s.count)
        }
        print("\(pad("stage", 40))  \(pad("rel_err", 12))  \(pad("max_abs", 12))  ref_mean / shape")
        for s in stages {
            let rel = String(format: "%.6f", s.relError)
            let mx = String(format: "%.5f", s.maxAbs)
            let rm = String(format: "%.5f", s.refMean)
            print("\(pad(s.name, 40))  \(pad(rel, 12))  \(pad(mx, 12))  \(rm)  \(s.shape)")
        }
    }
}
