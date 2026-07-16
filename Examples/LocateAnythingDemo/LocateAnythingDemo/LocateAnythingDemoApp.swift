import SwiftUI

@main
struct LocateAnythingDemoApp: App {
    @State private var engine = GroundingEngine()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(engine)
                .frame(minWidth: 960, minHeight: 660)
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentMinSize)
    }
}
