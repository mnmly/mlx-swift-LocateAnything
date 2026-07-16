import Foundation
import XCTest

@testable import MLXLocateAnything

/// End-to-end numerical parity against Python-reference fixtures.
///
/// Gated on env vars (skips when unset) and MUST run via xcodebuild — `swift
/// test` can't load MLX's Metal library:
///
///   LA_MODEL_PATH=<snapshot dir> LA_FIXTURES=<la_fixtures.safetensors> \
///     xcodebuild -scheme mlx-swift-LocateAnything-Package \
///       -destination 'platform=macOS' test -only-testing:MLXLocateAnythingTests/ParityTests
final class ParityTests: XCTestCase {
    func testStageParity() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["LA_MODEL_PATH"],
              let fixtures = ProcessInfo.processInfo.environment["LA_FIXTURES"]
        else {
            throw XCTSkip("Set LA_MODEL_PATH and LA_FIXTURES to run parity.")
        }

        let session = try await LocateAnythingSession.load(directory: URL(fileURLWithPath: modelPath))
        let stages = try LocateAnythingParity.run(
            session: session, fixtures: URL(fileURLWithPath: fixtures))

        func rel(_ name: String) -> Float? { stages.first { $0.name.hasPrefix(name) }?.relError }

        // Vision runs bf16 weights; exact pos-emb kernel -> sub-percent.
        XCTAssertLessThan(try XCTUnwrap(rel("vision_out")), 0.02, "vision_out drift too high")
        // Connector fed reference vision_out -> essentially exact.
        XCTAssertLessThan(try XCTUnwrap(rel("connector_feats")), 1e-3, "connector not exact")
        // The 8-bit LM logits are noisy, but the greedy decision must match.
        XCTAssertEqual(try XCTUnwrap(rel("first_token_argmax")), 0, "first-token argmax disagrees")
    }
}
