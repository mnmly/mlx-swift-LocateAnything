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

    /// The instruction sent to the model, built from the task + description.
    var instruction: String { task.instruction(query) }

    // Current image + last result.
    private(set) var image: CGImage?
    private(set) var imageName: String = ""
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

    func setImage(url: URL) {
        guard let cg = try? LocateAnythingImageProcessor.loadCGImage(path: url.path) else {
            lastError = "Could not read \(url.lastPathComponent)"
            return
        }
        image = cg          // already baked (memory-backed) by loadCGImage
        imageName = url.lastPathComponent
        result = nil
        lastError = nil
    }

    /// For drag-and-drop: bake the pixels immediately (the CGImage may decode
    /// lazily and outlive the drop grant otherwise — swift-cli-gui-shared-driver).
    func setImage(_ cg: CGImage, name: String) {
        image = LocateAnythingImageProcessor.baked(cg)
        imageName = name
        result = nil
        lastError = nil
    }

    func loadBundledSample(_ name: String) {
        let ext = ["png", "jpg", "jpeg"].first { Bundle.main.url(forResource: name, withExtension: $0) != nil }
        guard let ext, let url = Bundle.main.url(forResource: name, withExtension: ext) else { return }
        setImage(url: url)
    }

    // MARK: - Inference (off the main actor)

    func run() {
        guard let session, let image, !isRunning else { return }
        runTask?.cancel()
        isRunning = true
        lastError = nil
        let q = instruction
        let m = mode
        let imgBox = SendableImage(image: image)
        let start = Date()
        runTask = Task {
            do {
                let res = try await Task.detached(priority: .userInitiated) {
                    try session.predict(image: imgBox.image, question: q, maxNewTokens: 512, mode: m)
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
