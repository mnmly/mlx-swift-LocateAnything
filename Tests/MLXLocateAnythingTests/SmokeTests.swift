import Foundation
import XCTest

@testable import MLXLocateAnything

/// Pure (no-MLX) tests — safe under `swift test`.
final class ConfigTests: XCTestCase {
    func testDecodeConfig() throws {
        let json = """
        {
          "model_type": "locateanything",
          "image_token_index": 151665,
          "box_start_token_id": 151668,
          "box_end_token_id": 151669,
          "coord_start_token_id": 151677,
          "coord_end_token_id": 152677,
          "ref_start_token_id": 151672,
          "ref_end_token_id": 151673,
          "none_token_id": 4064,
          "eos_token_id": 151645,
          "quantization": { "group_size": 64, "bits": 8, "mode": "affine" },
          "text_config": {
            "model_type": "qwen2", "hidden_size": 2048, "num_hidden_layers": 36,
            "num_attention_heads": 16, "num_key_value_heads": 2, "vocab_size": 152681,
            "rope_theta": 1000000.0, "tie_word_embeddings": true, "block_size": 6,
            "text_mask_token_id": 151676, "null_token_id": 152678, "switch_token_id": 152679
          },
          "vision_config": {
            "model_type": "moonvit", "hidden_size": 1152, "num_hidden_layers": 27,
            "num_attention_heads": 16, "intermediate_size": 4304, "patch_size": 14,
            "merge_kernel_size": [2, 2]
          }
        }
        """.data(using: .utf8)!

        let c = try JSONDecoder().decode(LocateAnythingConfiguration.self, from: json)
        XCTAssertEqual(c.modelType, "locateanything")
        XCTAssertEqual(c.imageTokenIndex, 151665)
        XCTAssertEqual(c.coordStartTokenId, 151677)
        XCTAssertEqual(c.coordEndTokenId, 152677)
        XCTAssertEqual(c.eosTokenIds, [151645])
        XCTAssertEqual(c.quantization?.groupSize, 64)
        XCTAssertEqual(c.quantization?.bits, 8)
        // text
        XCTAssertEqual(c.textConfig.numHiddenLayers, 36)
        XCTAssertEqual(c.textConfig.numKeyValueHeads, 2)
        XCTAssertEqual(c.textConfig.headDim, 128)
        XCTAssertTrue(c.textConfig.tieWordEmbeddings)
        // vision derived aliases
        XCTAssertEqual(c.visionConfig.depth, 27)
        XCTAssertEqual(c.visionConfig.embedDim, 1152)
        XCTAssertEqual(c.visionConfig.spatialMergeSize, 2)
        XCTAssertEqual(c.visionConfig.headDim, 72)
    }
}

final class PromptTests: XCTestCase {
    func testChatMLSingleImage() {
        let p = LocateAnythingPrompt.chatML(question: "Locate all cars.", numImages: 1)
        XCTAssertTrue(p.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"))
        XCTAssertTrue(p.contains("<|im_start|>user\n<image-1>Locate all cars.<|im_end|>"))
        XCTAssertTrue(p.hasSuffix("<|im_start|>assistant\n"))
    }

    func testChatMLVisualPromptOrdering() {
        // Main image before text; visual prompt after (matches the worker).
        let p = LocateAnythingPrompt.chatML(question: "Detect this.", numImages: 2)
        XCTAssertTrue(p.contains("<image-1>Detect this.<image-2>"))
    }

    func testPlaceholderExpansion() {
        // grid 4x4 = 16 patches, merge 4 -> 4 image tokens.
        let prompt = "<image-1>find x"
        let expanded = LocateAnythingPrompt.expandPlaceholders(prompt, grids: [(4, 4)], mergeLength: 4)
        let count = expanded.components(separatedBy: LocateAnythingPrompt.imageToken).count - 1
        XCTAssertEqual(count, 4)
        XCTAssertTrue(expanded.hasPrefix(LocateAnythingPrompt.imageStartToken))
        XCTAssertTrue(expanded.contains(LocateAnythingPrompt.imageEndToken + "find x"))
    }
}
