import Foundation

// The mlx-community/LocateAnything-3B-8bit repo ships a legacy Qwen2 tokenizer
// (vocab.json + merges.txt + tokenizer_config.json) with no consolidated
// `tokenizer.json`, which swift-transformers requires. Rather than shelling out
// to Python `transformers`, synthesize a standard Qwen2 byte-level-BPE
// tokenizer.json in Swift, once, when it is missing.

enum LocateAnythingTokenizerSetup {

    // Qwen2 / GPT-2 style byte-level pre-tokenizer split pattern.
    private static let preTokenizerRegex =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    /// Ensure `directory/tokenizer.json` exists, synthesizing it from the legacy
    /// files if necessary. No-op when it is already present.
    static func ensureTokenizerJSON(in directory: URL) throws {
        let target = directory.appendingPathComponent("tokenizer.json")
        let fm = FileManager.default
        if fm.fileExists(atPath: target.path) { return }

        let vocabURL = directory.appendingPathComponent("vocab.json")
        let mergesURL = directory.appendingPathComponent("merges.txt")
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        guard fm.fileExists(atPath: vocabURL.path), fm.fileExists(atPath: mergesURL.path) else {
            // Nothing to synthesize from; let the loader surface its own error.
            return
        }

        let vocab = try JSONSerialization.jsonObject(with: Data(contentsOf: vocabURL)) as? [String: Any] ?? [:]

        var merges: [String] = []
        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        for rawLine in mergesText.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = String(rawLine)
            if line.hasPrefix("#") { continue }   // "#version: ..." header
            merges.append(line)
        }

        var addedTokens: [[String: Any]] = []
        if let cfgData = try? Data(contentsOf: configURL),
           let cfg = try? JSONSerialization.jsonObject(with: cfgData) as? [String: Any],
           let decoder = cfg["added_tokens_decoder"] as? [String: Any] {
            for (idString, value) in decoder {
                guard let id = Int(idString), let info = value as? [String: Any] else { continue }
                addedTokens.append([
                    "id": id,
                    "content": info["content"] as? String ?? "",
                    "single_word": info["single_word"] as? Bool ?? false,
                    "lstrip": info["lstrip"] as? Bool ?? false,
                    "rstrip": info["rstrip"] as? Bool ?? false,
                    "normalized": info["normalized"] as? Bool ?? false,
                    "special": info["special"] as? Bool ?? true,
                ])
            }
            addedTokens.sort { ($0["id"] as? Int ?? 0) < ($1["id"] as? Int ?? 0) }
        }

        let tokenizerJSON: [String: Any] = [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": addedTokens,
            "normalizer": ["type": "NFC"],
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    ["type": "Split",
                     "pattern": ["Regex": preTokenizerRegex],
                     "behavior": "Isolated",
                     "invert": false],
                    ["type": "ByteLevel",
                     "add_prefix_space": false,
                     "trim_offsets": true,
                     "use_regex": false],
                ],
            ],
            "post_processor": [
                "type": "TemplateProcessing",
                "single": [["Sequence": ["id": "A", "type_id": 0]]],
                "pair": [["Sequence": ["id": "A", "type_id": 0]], ["Sequence": ["id": "B", "type_id": 1]]],
                "special_tokens": [String: Any](),
            ],
            "decoder": [
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true,
            ],
            "model": [
                "type": "BPE",
                "dropout": NSNull(),
                "unk_token": NSNull(),
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocab,
                "merges": merges,
            ],
        ]

        let data = try JSONSerialization.data(withJSONObject: tokenizerJSON, options: [])
        try data.write(to: target)
    }
}
