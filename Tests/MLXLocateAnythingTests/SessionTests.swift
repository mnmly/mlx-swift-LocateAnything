import Foundation
import XCTest

@testable import MLXLocateAnything

/// The shared-driver contract: `Session.load` + `predict` works end-to-end.
/// Both the CLI and the SwiftUI demo inherit correctness from this.
///
/// Gated on env vars (skips when unset); MUST run via xcodebuild (MLX Metal).
/// xcodebuild only forwards variables prefixed with `TEST_RUNNER_`:
///
///   TEST_RUNNER_LA_MODEL_PATH=<snapshot dir> TEST_RUNNER_LA_TEST_IMAGE=<image> \
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

    /// Regression: with several categories, Parallel Box Decoding used to accept a
    /// block whose row 0 opened the next `<ref>` as a box, re-emitting the previous
    /// category's last box until the token budget ran out. PBD must terminate and
    /// agree with the autoregressive (`slow`) decode.
    ///
    /// Reproduced with COCO val2017 `000000039769.jpg` (two cats, couch, remotes).
    func testPBDMultiCategoryMatchesSlowAndTerminates() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["LA_MODEL_PATH"],
              let imagePath = ProcessInfo.processInfo.environment["LA_TEST_IMAGE"]
        else {
            throw XCTSkip("Set LA_MODEL_PATH and LA_TEST_IMAGE to run the Session test.")
        }

        let session = try await LocateAnythingSession.load(directory: URL(fileURLWithPath: modelPath))
        let image = try LocateAnythingImageProcessor.loadCGImage(path: imagePath)
        let question = "Locate all the instances that matches the following description: cat</c>couch</c>remote."
        let budget = 512

        let slow = try session.predict(image: image, question: question, maxNewTokens: budget, mode: .slow)
        XCTAssertLessThan(slow.tokenCount, budget, "slow decode should terminate on its own")

        for mode in [PBDMode.fast, .hybrid] {
            let r = try session.predict(image: image, question: question, maxNewTokens: budget, mode: mode)
            XCTAssertLessThan(r.tokenCount, budget, "\(mode) ran to the token budget (repetition loop)")
            XCTAssertEqual(r.boxes.count, slow.boxes.count, "\(mode) box count differs from slow")
            XCTAssertEqual(r.boxes.map(\.label), slow.boxes.map(\.label), "\(mode) labels differ from slow")
        }
    }
}
