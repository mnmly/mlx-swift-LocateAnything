import Foundation
import XCTest

@testable import MLXLocateAnything

/// The shared-driver contract: `Session.load` + `predict` works end-to-end.
/// Both the CLI and the SwiftUI demo inherit correctness from this.
///
/// Gated on env vars (skips when unset); MUST run via xcodebuild (MLX Metal):
///
///   LA_MODEL_PATH=<snapshot dir> LA_TEST_IMAGE=<image> \
///     xcodebuild -scheme mlx-swift-LocateAnything-Package \
///       -destination 'platform=macOS' test -only-testing:MLXLocateAnythingTests/SessionTests
final class SessionTests: XCTestCase {
    func testLoadAndPredict() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["LA_MODEL_PATH"],
              let imagePath = ProcessInfo.processInfo.environment["LA_TEST_IMAGE"]
        else {
            throw XCTSkip("Set LA_MODEL_PATH and LA_TEST_IMAGE to run the Session test.")
        }

        let session = try await LocateAnythingSession.load(directory: URL(fileURLWithPath: modelPath))
        let image = try LocateAnythingImageProcessor.loadCGImage(path: imagePath)

        let result = try session.predict(
            image: image,
            question: "Locate all the instances that matches the following description: car.",
            maxNewTokens: 128,
            mode: .hybrid)

        // The model should emit the box grammar and parse at least one box.
        XCTAssertFalse(result.text.isEmpty)
        XCTAssertGreaterThan(result.tokenCount, 0)
        // Every parsed box carries a normalized rect in 0…1.
        for box in result.boxes {
            for v in box.box { XCTAssert(v >= -0.01 && v <= 1.01, "coord out of range: \(v)") }
        }
    }
}
