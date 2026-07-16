import AVFoundation
import AppKit
import CoreGraphics
import Foundation
import MLXLocateAnything
import Observation
import UniformTypeIdentifiers

// Frontend view-model. It owns ONLY presentation concerns (model gate, current
// image + prompt + mode, MainActor hops). All compute goes through the library's
// `LocateAnythingSession` — the same shared driver `locateanything-cli` consumes —
// so the GUI can never drift from the CLI (swift-cli-gui-shared-driver).

/// CGImage is an immutable, thread-safe CoreFoundation type; box it to cross the
/// detached-task boundary without tripping Sendable checking.
private struct SendableImage: @unchecked Sendable { let image: CGImage }

/// Extracted video frames crossing back from the detached extraction task.
private struct SendableVideo: @unchecked Sendable {
    let frames: [CGImage]
    let timestamps: [Double]
    let sampledFPS: Float
    let width: Int
    let height: Int
}

@MainActor
@Observable
final class GroundingEngine {

    enum ModelState: Equatable {
        case needsModel
        case downloading(Double)      // 0…1
        case loading
        case ready
        case failed(String)
    }

    /// A downloadable quantization of the model.
    struct ModelVariant: Identifiable, Hashable {
        let id: String        // Hugging Face repo id
        let title: String
        let size: String
        static let all: [ModelVariant] = [
            .init(id: "mlx-community/LocateAnything-3B-4bit", title: "4-bit", size: "~1.8 GB"),
            .init(id: "mlx-community/LocateAnything-3B-8bit", title: "8-bit", size: "~3.3 GB"),
            .init(id: "mlx-community/LocateAnything-3B-bf16", title: "bf16", size: "~6.2 GB"),
        ]
    }

    private(set) var modelState: ModelState = .needsModel
    private(set) var session: LocateAnythingSession?

    /// The variant to download / load (bound by the gate's picker).
    var selectedRepoID: String = LocateAnythingSession.defaultRepoID

    /// The repo id of the currently-loaded model (nil for a hand-picked local dir).
    private(set) var loadedRepoID: String?

    /// Display name of the loaded model, for the persistent badge.
    var loadedTitle: String? {
        guard session != nil else { return nil }
        guard let loadedRepoID else { return "Local model" }
        return ModelVariant.all.first { $0.id == loadedRepoID }?.title ?? loadedRepoID
    }

    // Inputs (bound by the UI).
    var task: LocateAnythingTask = .detection
    var query: String = "person"
    var mode: PBDMode = .hybrid

    // Sampling (AR path only; 0 temperature = greedy / deterministic).
    var temperature: Float = 0
    var topP: Float = 1.0
    var topK: Int = 0

    /// The instruction sent to the model, built from the task + description.
    var instruction: String { task.instruction(query) }

    // Current image + last result.
    private(set) var image: CGImage?
    private(set) var imageName: String = ""

    // Video: sampled frames + their timestamps (empty for a still image).
    private(set) var frames: [CGImage] = []
    private(set) var frameTimestamps: [Double] = []
    private(set) var sampledFPS: Float = 0
    private(set) var videoSize: CGSize = .zero
    private(set) var videoURL: URL?
    private(set) var isLoadingMedia = false
    /// Frames to sample from a video. Cost scales with total image tokens, so this
    /// is the main speed lever (4 frames ≈ 15s, 10 ≈ 85s on 1080p).
    var maxFrames: Int = 8
    /// Which sampled frame the canvas shows (scrubber).
    var currentFrameIndex: Int = 0

    var isVideo: Bool { !frames.isEmpty }
    /// The frame currently on screen — a still image, or the scrubbed video frame.
    var displayImage: CGImage? {
        if isVideo { return frames[min(max(currentFrameIndex, 0), frames.count - 1)] }
        return image
    }
    var hasMedia: Bool { displayImage != nil }

    private(set) var isRunning = false
    private(set) var result: LocateAnythingResult?
    private(set) var lastError: String?
    private(set) var lastDuration: Double = 0

    private var runTask: Task<Void, Never>?

    func isCached(_ repoId: String) -> Bool { LocateAnythingSession.cachedDirectory(repoId: repoId) != nil }
    var selectedIsCached: Bool { isCached(selectedRepoID) }

    // MARK: - Model acquisition

    func loadCachedIfAvailable() {
        guard session == nil, case .needsModel = modelState else { return }
        // Prefer the selected variant; else load any variant already downloaded.
        if let dir = LocateAnythingSession.cachedDirectory(repoId: selectedRepoID) {
            load(directory: dir, repoId: selectedRepoID)
            return
        }
        for v in ModelVariant.all where isCached(v.id) {
            selectedRepoID = v.id
            load(directory: LocateAnythingSession.cachedDirectory(repoId: v.id)!, repoId: v.id)
            return
        }
    }

    /// Download the selected variant into `~/.cache/huggingface`, then load.
    /// If it's already cached, this loads it without re-downloading.
    func acquireSelected() {
        guard case .needsModel = modelState else { return }
        let repo = selectedRepoID
        if let dir = LocateAnythingSession.cachedDirectory(repoId: repo) {
            load(directory: dir, repoId: repo)
            return
        }
        modelState = .downloading(0)
        Task {
            do {
                let dir = try await LocateAnythingSession.download(repoId: repo) { @MainActor frac in
                    if case .downloading = self.modelState { self.modelState = .downloading(frac) }
                }
                load(directory: dir, repoId: repo)
            } catch {
                modelState = .failed("Download failed: \(error.localizedDescription)")
            }
        }
    }

    func load(directory: URL, repoId: String? = nil) {
        modelState = .loading
        Task {
            do {
                let loaded = try await Task.detached(priority: .userInitiated) {
                    try await LocateAnythingSession.load(directory: directory)
                }.value
                self.session = loaded
                self.loadedRepoID = repoId
                self.modelState = .ready
            } catch {
                self.modelState = .failed("Load failed: \(error.localizedDescription)")
            }
        }
    }

    /// Unload the current model and return to the gate to pick a different
    /// variant. Does not delete anything from the cache.
    func switchModel() {
        cancel()
        session = nil
        loadedRepoID = nil
        result = nil
        lastError = nil
        modelState = .needsModel
    }

    func retry() {
        session = nil
        loadedRepoID = nil
        modelState = .needsModel
        loadCachedIfAvailable()
    }

    /// Reveal the loaded model's snapshot folder in Finder.
    func revealModelInFinder() {
        guard let loadedRepoID,
              let dir = LocateAnythingSession.cachedDirectory(repoId: loadedRepoID) else { return }
        NSWorkspace.shared.activateFileViewerSelecting([dir])
    }

    func chooseLocalModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Load"
        panel.message = "Choose the model snapshot directory (contains config.json)"
        panel.directoryURL = LocateAnythingSession.cachedDirectory()?.deletingLastPathComponent()
        if panel.runModal() == .OK, let url = panel.url { load(directory: url) }
    }

    // MARK: - Image

    func openImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .heic, .tiff, .image]
        panel.allowsMultipleSelection = false
        panel.prompt = "Open"
        if panel.runModal() == .OK, let url = panel.url { setImage(url: url) }
    }

    func openVideo() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie, .video, .mpeg4Movie, .quickTimeMovie]
        panel.allowsMultipleSelection = false
        panel.prompt = "Open"
        panel.message = "Choose a video — frames are sampled at 2 fps (max 64)"
        if panel.runModal() == .OK, let url = panel.url { setVideo(url: url) }
    }

    /// Route a dropped/opened URL to the image or video path by content type.
    func setMedia(url: URL) {
        let isMovie = (try? url.resourceValues(forKeys: [.contentTypeKey]))?
            .contentType?.conforms(to: .movie) ?? false
        if isMovie { setVideo(url: url) } else { setImage(url: url) }
    }

    func setImage(url: URL) {
        guard let cg = try? LocateAnythingImageProcessor.loadCGImage(path: url.path) else {
            lastError = "Could not read \(url.lastPathComponent)"
            return
        }
        clearMedia()
        image = cg          // already baked (memory-backed, EXIF-oriented) by loadCGImage
        imageName = url.lastPathComponent
    }

    /// Sample + extract video frames (2 fps, max 64) via AVFoundation, off the main actor.
    func setVideo(url: URL) {
        isLoadingMedia = true
        lastError = nil
        let cap = maxFrames
        Task {
            do {
                // Shared library extractor — the CLI uses the same one.
                let box = try await LocateAnythingVideo.extractFrames(url: url, frameCap: cap)
                clearMedia()
                self.frames = box.frames
                self.frameTimestamps = box.timestamps
                self.sampledFPS = box.sampledFPS
                self.videoSize = CGSize(width: box.width, height: box.height)
                self.videoURL = url
                self.imageName = url.lastPathComponent
                self.isLoadingMedia = false
            } catch {
                self.lastError = "Couldn't read video: \(error.localizedDescription)"
                self.isLoadingMedia = false
            }
        }
    }

    private func clearMedia() {
        image = nil
        frames = []
        frameTimestamps = []
        sampledFPS = 0
        videoSize = .zero
        videoURL = nil
        currentFrameIndex = 0
        result = nil
        lastError = nil
    }

    /// Re-sample the current video after the frame cap changes.
    func resampleVideo() {
        guard let videoURL else { return }
        setVideo(url: videoURL)
    }

    /// For drag-and-drop: bake the pixels immediately (the CGImage may decode
    /// lazily and outlive the drop grant otherwise — swift-cli-gui-shared-driver).
    func setImage(_ cg: CGImage, name: String) {
        clearMedia()
        image = LocateAnythingImageProcessor.baked(cg)
        imageName = name
    }

    func loadBundledSample(_ name: String) {
        let ext = ["png", "jpg", "jpeg"].first { Bundle.main.url(forResource: name, withExtension: $0) != nil }
        guard let ext, let url = Bundle.main.url(forResource: name, withExtension: ext) else { return }
        setImage(url: url)
    }

    // MARK: - Inference (off the main actor)

    func run() {
        guard let session, hasMedia, !isRunning else { return }
        runTask?.cancel()
        isRunning = true
        lastError = nil
        let q = instruction
        let m = mode
        let s = SamplingParameters(temperature: temperature, topP: topP, topK: topK)
        let start = Date()

        let video: SendableVideo? = isVideo
            ? SendableVideo(
                frames: frames, timestamps: frameTimestamps, sampledFPS: sampledFPS,
                width: Int(videoSize.width), height: Int(videoSize.height))
            : nil
        let still: SendableImage? = isVideo ? nil : image.map { SendableImage(image: $0) }

        runTask = Task {
            do {
                let res = try await Task.detached(priority: .userInitiated) { () -> LocateAnythingResult in
                    if let video {
                        return try session.predictVideo(
                            frames: video.frames, timestamps: video.timestamps,
                            sampledFPS: video.sampledFPS,
                            displayWidth: video.width, displayHeight: video.height,
                            question: q, maxNewTokens: 512, mode: m, sampling: s)
                    }
                    guard let still else { throw LocateAnythingError.imageDecodeFailed }
                    return try session.predict(
                        image: still.image, question: q, maxNewTokens: 512, mode: m, sampling: s)
                }.value
                if Task.isCancelled { return }
                self.result = res
                self.lastDuration = Date().timeIntervalSince(start)
                self.isRunning = false
            } catch {
                if Task.isCancelled { return }
                self.lastError = error.localizedDescription
                self.isRunning = false
            }
        }
    }

    func cancel() {
        runTask?.cancel()
        isRunning = false
    }
}
