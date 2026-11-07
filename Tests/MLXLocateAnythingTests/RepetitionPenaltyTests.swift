import MLX
import XCTest

@testable import MLXLocateAnything

/// `applyRepetitionPenalty` must match NVIDIA's `apply_repetition_penalty`:
/// seen tokens' positive logits are divided, non-positive multiplied, every row.
///
/// Needs MLX Metal — run via xcodebuild:
///
///   xcodebuild -scheme mlx-swift-LocateAnything-Package -destination 'platform=macOS' \
///     test -only-testing:MLXLocateAnythingTests/RepetitionPenaltyTests
final class RepetitionPenaltyTests: XCTestCase {
    func testPenalizesSeenTokensInEveryRow() {
        let logits = MLXArray([2, -2, 1, 4, 0, -1] as [Float], [2, 3])
        // Token 7 is out of vocab range and must be ignored.
        let out = applyRepetitionPenalty(logits, seen: SeenTokens([0, 1, 7]), penalty: 2)
        XCTAssertEqual(out.asArray(Float.self), [1, -4, 1, 2, 0, -1])
    }

    func testPenaltyOfOneIsIdentity() {
        let logits = MLXArray([2, -2, 1] as [Float], [1, 3])
        let out = applyRepetitionPenalty(logits, seen: SeenTokens([0, 1, 2]), penalty: 1)
        XCTAssertEqual(out.asArray(Float.self), [2, -2, 1])
    }
}
