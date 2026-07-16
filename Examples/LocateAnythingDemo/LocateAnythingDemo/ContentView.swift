import MLXLocateAnything
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(GroundingEngine.self) private var engine
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var dropTargeted = false
    @State private var showModeInfo = false
    @State private var showRaw = false

    private var isReady: Bool {
        if case .ready = engine.modelState { return true }
        return false
    }

    var body: some View {
        ZStack {
            canvas
                .ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer(minLength: 0)
                if isReady { controlBar.transition(.move(edge: .bottom).combined(with: .opacity)) }
            }

            if isReady {
                VStack {
                    HStack { Spacer(); modelBadge }
                    Spacer()
                }
                .padding(16)
                .transition(.opacity)
            }

            if !isReady {
                Color.black.opacity(0.28).ignoresSafeArea()
                    .transition(.opacity)
                ModelGate().transition(reduceMotion ? .opacity : .scale(scale: 0.96).combined(with: .opacity))
            }
        }
        .animation(reduceMotion ? .easeInOut(duration: 0.2) : Theme.spring, value: engine.modelState)
        .animation(Theme.spring, value: engine.isRunning)
        .onAppear { engine.loadCachedIfAvailable() }
        .onChange(of: engine.task) { old, new in
            // Points are 2-coord; Parallel Box Decoding atomically emits 4-coord
            // boxes, so pointing is only reliable under autoregressive decoding.
            if new == .pointing {
                engine.mode = .ar
            } else if old == .pointing && engine.mode == .ar {
                engine.mode = .hybrid
            }
        }
        .onDrop(of: [.fileURL, .image], isTargeted: $dropTargeted) { handleDrop($0) }
    }

    // MARK: - Canvas

    @ViewBuilder
    private var canvas: some View {
        ZStack {
            LinearGradient(
                colors: [Color(nsColor: .windowBackgroundColor), Color(nsColor: .underPageBackgroundColor)],
                startPoint: .top, endPoint: .bottom)

            if let image = engine.image {
                BoxOverlayView(image: image, boxes: engine.result?.boxes ?? [])
                    .padding(24)
                    .padding(.bottom, isReady ? 96 : 24)

                if engine.isRunning { runningOverlay }
            } else {
                EmptyCanvas(targeted: dropTargeted)
                    .padding(.bottom, isReady ? 96 : 0)
            }
        }
        .overlay(alignment: .top) {
            if dropTargeted {
                RoundedRectangle(cornerRadius: 20)
                    .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 3, dash: [8, 6]))
                    .padding(10)
                    .transition(.opacity)
            }
        }
        .animation(Theme.spring, value: dropTargeted)
    }

    private var runningOverlay: some View {
        VStack(spacing: 10) {
            ProgressView().controlSize(.large)
            Text("Locating…")
                .font(.callout.weight(.medium))
                .foregroundStyle(.secondary)
        }
        .padding(22)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.2), radius: 20, y: 8)
        .transition(.scale(scale: 0.9).combined(with: .opacity))
    }

    // MARK: - Model badge (persistent weights control)

    private var modelBadge: some View {
        Menu {
            Button("Switch model…", systemImage: "arrow.triangle.2.circlepath") { engine.switchModel() }
            Button("Reveal in Finder", systemImage: "folder") { engine.revealModelInFinder() }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "cpu").imageScale(.small)
                Text(engine.loadedTitle ?? "Model").font(.caption.weight(.semibold))
                Image(systemName: "chevron.down").imageScale(.small).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 11)
            .padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(.white.opacity(0.08)))
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Loaded model — switch variant or reveal in Finder")
    }

    // MARK: - Control bar

    private var controlBar: some View {
        @Bindable var engine = engine
        return VStack(spacing: 12) {
            // Task-specific example queries.
            let examples = TaskExamples.for(engine.task)
            if engine.task.usesDescription && !examples.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(examples, id: \.self) { ex in
                            Button(ex) { engine.query = ex }
                                .buttonStyle(ChipButtonStyle())
                        }
                    }
                    .padding(.horizontal, 2)
                }
                .transition(.opacity)
            }

            HStack(spacing: 12) {
                Picker("Task", selection: $engine.task) {
                    ForEach(LocateAnythingTask.allCases, id: \.self) { Text($0.title).tag($0) }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(width: 118)
                .help("Grounding task — sets the instruction template")

                // Description field (disabled for OCR, which needs no input).
                TextField(engine.task.descriptionHint, text: $engine.query, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1 ... 3)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 9)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
                    .disabled(!engine.task.usesDescription)
                    .opacity(engine.task.usesDescription ? 1 : 0.5)
                    .onSubmit { engine.run() }

                HStack(spacing: 2) {
                    Picker("Mode", selection: $engine.mode) {
                        ForEach(PBDMode.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .frame(width: 96)

                    Button { showModeInfo.toggle() } label: {
                        Image(systemName: "info.circle").imageScale(.medium).foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("What do these modes mean?")
                    .popover(isPresented: $showModeInfo, arrowEdge: .top) { ModeInfoView() }
                }

                Button {
                    engine.openImage()
                } label: {
                    Image(systemName: "photo.on.rectangle").imageScale(.large)
                }
                .buttonStyle(PressableButtonStyle())
                .help("Open an image")

                Button {
                    engine.isRunning ? engine.cancel() : engine.run()
                } label: {
                    Label(engine.isRunning ? "Stop" : "Locate",
                          systemImage: engine.isRunning ? "stop.fill" : "scope")
                        .font(.body.weight(.semibold))
                        .frame(minWidth: 96)
                }
                .buttonStyle(PrimaryButtonStyle(role: engine.isRunning ? .stop : .go))
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(engine.image == nil)
            }

            if let err = engine.lastError {
                Label(err, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .transition(.opacity)
            } else if let result = engine.result {
                resultSummary(result)
            }
        }
        .padding(14)
        .background(.ultraThinMaterial)
        .overlay(alignment: .top) { Divider().opacity(0.6) }
        .clipShape(RoundedRectangle(cornerRadius: 18))
        .padding(12)
        .shadow(color: .black.opacity(0.16), radius: 22, y: 10)
        .animation(reduceMotion ? .easeInOut(duration: 0.2) : Theme.spring, value: engine.task)
    }

    private func resultSummary(_ result: LocateAnythingResult) -> some View {
        HStack(spacing: 10) {
            Image(systemName: result.boxes.isEmpty ? "questionmark.circle" : "checkmark.circle.fill")
                .foregroundStyle(result.boxes.isEmpty ? Color.secondary : Color.green)
            Text("\(result.boxes.count) detection\(result.boxes.count == 1 ? "" : "s")")
                .font(.caption.weight(.medium))
            if engine.lastDuration > 0 {
                Text("· \(String(format: "%.2fs", engine.lastDuration)) · \(result.tokenCount) tok · "
                    + "\(String(format: "%.0f", Double(result.tokenCount) / max(engine.lastDuration, 1e-6))) tok/s")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            Spacer()
            if !result.text.isEmpty {
                Button { showRaw.toggle() } label: {
                    Label("Raw", systemImage: "curlybraces").font(.caption)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .popover(isPresented: $showRaw, arrowEdge: .bottom) { RawOutputView(text: result.text) }
            }
        }
        .transition(.opacity)
    }

    // MARK: - Drop

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                Task { @MainActor in engine.setImage(url: url) }
            }
            return true
        }
        return false
    }
}

// MARK: - Empty canvas

private struct EmptyCanvas: View {
    let targeted: Bool
    @Environment(GroundingEngine.self) private var engine

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "scope")
                .font(.system(size: 52, weight: .regular))
                .foregroundStyle(.secondary)
                .symbolEffect(.pulse, options: .repeating, isActive: targeted)
            Text("Drop an image to ground")
                .font(.title3.weight(.semibold))
                .tracking(-0.2)
            Text("or open one, then describe what to locate")
                .font(.callout)
                .foregroundStyle(.secondary)
            HStack(spacing: 8) {
                Button("Open Image…") { engine.openImage() }
                    .buttonStyle(PrimaryButtonStyle(role: .go))
                ForEach(SampleImage.all) { s in
                    Button(s.title) { engine.loadBundledSample(s.resource) }
                        .buttonStyle(ChipButtonStyle())
                }
            }
            .padding(.top, 4)
        }
        .padding(40)
    }
}

// MARK: - Model gate

private struct ModelGate: View {
    @Environment(GroundingEngine.self) private var engine

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "square.and.arrow.down.on.square")
                .font(.system(size: 40))
                .foregroundStyle(.tint)

            Text("LocateAnything")
                .font(.title.weight(.bold))
                .tracking(-0.4)
            Text("Fast visual grounding · MLX · Apple Silicon")
                .font(.callout)
                .foregroundStyle(.secondary)

            content
                .frame(maxWidth: 340)
        }
        .padding(32)
        .frame(width: 420)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 22))
        .overlay(RoundedRectangle(cornerRadius: 22).strokeBorder(.white.opacity(0.12)))
        .shadow(color: .black.opacity(0.3), radius: 40, y: 18)
    }

    @ViewBuilder
    private var content: some View {
        switch engine.modelState {
        case .needsModel:
            @Bindable var engine = engine
            VStack(spacing: 14) {
                Picker("Precision", selection: $engine.selectedRepoID) {
                    ForEach(GroundingEngine.ModelVariant.all) { v in Text(v.title).tag(v.id) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                if let v = GroundingEngine.ModelVariant.all.first(where: { $0.id == engine.selectedRepoID }) {
                    Text(engine.isCached(v.id)
                         ? "\(v.size) · already in ~/.cache/huggingface"
                         : "\(v.size) · downloads to ~/.cache/huggingface")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Button {
                    engine.acquireSelected()
                } label: {
                    Label(engine.selectedIsCached ? "Load model" : "Download model",
                          systemImage: engine.selectedIsCached ? "checkmark.circle.fill" : "arrow.down.circle.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(PrimaryButtonStyle(role: .go))

                Button("Choose local snapshot…") { engine.chooseLocalModel() }
                    .buttonStyle(.link)
            }
        case .downloading(let frac):
            VStack(spacing: 10) {
                ProgressView(value: frac) { Text("Downloading…") } currentValueLabel: {
                    Text("\(Int(frac * 100)) %").font(.caption).monospacedDigit()
                }
                .progressViewStyle(.linear)
            }
        case .loading:
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text("Loading weights…").foregroundStyle(.secondary)
            }
        case .ready:
            EmptyView()
        case .failed(let message):
            VStack(spacing: 12) {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                Button("Try again") { engine.retry() }
                    .buttonStyle(PrimaryButtonStyle(role: .go))
            }
        }
    }
}

// MARK: - Button styles (apple-design: instant press feedback)

struct PressableButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(8)
            .contentShape(Rectangle())
            .scaleEffect(configuration.isPressed ? 0.94 : 1)
            .animation(.spring(response: 0.2, dampingFraction: 1.0), value: configuration.isPressed)
    }
}

struct PrimaryButtonStyle: ButtonStyle {
    enum Role { case go, stop }
    var role: Role = .go

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(.horizontal, 16)
            .padding(.vertical, 9)
            .background(role == .stop ? Color.red : Color.accentColor, in: RoundedRectangle(cornerRadius: 10))
            .foregroundStyle(.white)
            .brightness(configuration.isPressed ? -0.06 : 0)
            .scaleEffect(configuration.isPressed ? 0.97 : 1)
            .animation(.spring(response: 0.2, dampingFraction: 1.0), value: configuration.isPressed)
    }
}

struct ChipButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.caption.weight(.medium))
            .padding(.horizontal, 11)
            .padding(.vertical, 6)
            .background(.quaternary, in: Capsule())
            .overlay(Capsule().strokeBorder(.white.opacity(0.06)))
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
            .animation(.spring(response: 0.2, dampingFraction: 1.0), value: configuration.isPressed)
    }
}

// MARK: - Presets

enum TaskExamples {
    static func `for`(_ task: LocateAnythingTask) -> [String] {
        switch task {
        case .detection: return ["person", "car", "person, car, bus", "traffic light"]
        case .grounding: return ["people wearing red", "the person on the left", "the moving vehicle"]
        case .gui: return ["the search button", "the menu icon", "the close button"]
        case .pointing: return ["the traffic light", "the tallest building", "the crosswalk"]
        case .ocr, .custom: return []
        }
    }
}

struct SampleImage: Identifiable {
    let id = UUID()
    let title: String
    let resource: String
    static let all: [SampleImage] = [
        .init(title: "Smart City", resource: "Smart_City"),
    ]
}

// MARK: - Mode explainer

private struct ModeInfoView: View {
    private let modes: [(String, String)] = [
        ("Hybrid", "Parallel Box Decoding with an autoregressive fallback. Predicts each box atomically in one forward pass, then drops to token-by-token if a box comes out malformed and resumes after it. Best quality + speed. Default."),
        ("Fast", "Pure Parallel Box Decoding — always accepts the atomic box block (even malformed ones). Fewest forward passes."),
        ("Slow", "The box grammar decoded autoregressively (no speculative block). A slower reference path."),
        ("AR", "Plain greedy token-by-token decoding — the baseline, no Parallel Box Decoding."),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Decoding modes")
                .font(.headline)
            ForEach(modes, id: \.0) { name, desc in
                VStack(alignment: .leading, spacing: 2) {
                    Text(name).font(.subheadline.weight(.semibold))
                    Text(desc)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(16)
        .frame(width: 320)
    }
}

private struct RawOutputView: View {
    let text: String
    var body: some View {
        ScrollView {
            Text(text)
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
        }
        .frame(width: 400, height: 170)
    }
}
