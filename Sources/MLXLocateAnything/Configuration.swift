import Foundation

// Ported from mlx_vlm/models/locateanything/config.py
//
// The mlx-community/LocateAnything-3B-8bit config.json nests `text_config` and
// `vision_config` as real sub-objects, so we decode them directly (no overlay).
//
// Convention (matches other mlx-swift ports): snake_case JSON keys mapped via
// CodingKeys; fields that may be missing use a private optional + computed getter
// so struct defaults survive Codable decoding.

// MARK: - Quantization

public struct QuantizationConfig: Codable, Sendable {
    public let groupSize: Int
    public let bits: Int
    public let mode: String?

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
    }
}

// MARK: - Vision (MoonViT)

public struct VisionConfiguration: Codable, Sendable {
    private let _modelType: String?
    public var modelType: String { _modelType ?? "moonvit" }

    private let _hiddenSize: Int?
    public var hiddenSize: Int { _hiddenSize ?? 1152 }

    private let _numHiddenLayers: Int?
    public var numHiddenLayers: Int { _numHiddenLayers ?? 27 }

    private let _numAttentionHeads: Int?
    public var numAttentionHeads: Int { _numAttentionHeads ?? 16 }

    private let _intermediateSize: Int?
    public var intermediateSize: Int { _intermediateSize ?? 4304 }

    private let _patchSize: Int?
    public var patchSize: Int { _patchSize ?? 14 }

    private let _initPosEmbHeight: Int?
    public var initPosEmbHeight: Int { _initPosEmbHeight ?? 64 }

    private let _initPosEmbWidth: Int?
    public var initPosEmbWidth: Int { _initPosEmbWidth ?? 64 }

    private let _numChannels: Int?
    public var numChannels: Int { _numChannels ?? 3 }

    private let _mergeKernelSize: [Int]?
    public var mergeKernelSize: [Int] { _mergeKernelSize ?? [2, 2] }

    // Derived aliases (Python __post_init__).
    public var depth: Int { numHiddenLayers }
    public var numHeads: Int { numAttentionHeads }
    public var embedDim: Int { hiddenSize }
    public var spatialMergeSize: Int { mergeKernelSize[0] }
    public var headDim: Int { hiddenSize / numAttentionHeads }

    enum CodingKeys: String, CodingKey {
        case _modelType = "model_type"
        case _hiddenSize = "hidden_size"
        case _numHiddenLayers = "num_hidden_layers"
        case _numAttentionHeads = "num_attention_heads"
        case _intermediateSize = "intermediate_size"
        case _patchSize = "patch_size"
        case _initPosEmbHeight = "init_pos_emb_height"
        case _initPosEmbWidth = "init_pos_emb_width"
        case _numChannels = "num_channels"
        case _mergeKernelSize = "merge_kernel_size"
    }
}

// MARK: - Text (Qwen2)

public struct TextConfiguration: Codable, Sendable {
    private let _modelType: String?
    public var modelType: String { _modelType ?? "qwen2" }

    private let _hiddenSize: Int?
    public var hiddenSize: Int { _hiddenSize ?? 2048 }

    private let _numHiddenLayers: Int?
    public var numHiddenLayers: Int { _numHiddenLayers ?? 36 }

    private let _intermediateSize: Int?
    public var intermediateSize: Int { _intermediateSize ?? 11008 }

    private let _numAttentionHeads: Int?
    public var numAttentionHeads: Int { _numAttentionHeads ?? 16 }

    private let _numKeyValueHeads: Int?
    public var numKeyValueHeads: Int { _numKeyValueHeads ?? numAttentionHeads }

    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? 152681 }

    private let _rmsNormEps: Float?
    public var rmsNormEps: Float { _rmsNormEps ?? 1e-6 }

    private let _ropeTheta: Float?
    public var ropeTheta: Float { _ropeTheta ?? 1_000_000.0 }

    private let _ropeTraditional: Bool?
    public var ropeTraditional: Bool { _ropeTraditional ?? false }

    private let _maxPositionEmbeddings: Int?
    public var maxPositionEmbeddings: Int { _maxPositionEmbeddings ?? 32768 }

    private let _tieWordEmbeddings: Bool?
    public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? true }

    private let _blockSize: Int?
    public var blockSize: Int { _blockSize ?? 6 }

    private let _causalAttn: Bool?
    public var causalAttn: Bool { _causalAttn ?? false }

    private let _textMaskTokenId: Int?
    public var textMaskTokenId: Int { _textMaskTokenId ?? 151676 }

    private let _nullTokenId: Int?
    public var nullTokenId: Int { _nullTokenId ?? 152678 }

    private let _switchTokenId: Int?
    public var switchTokenId: Int { _switchTokenId ?? 152679 }

    public var headDim: Int { hiddenSize / numAttentionHeads }

    enum CodingKeys: String, CodingKey {
        case _modelType = "model_type"
        case _hiddenSize = "hidden_size"
        case _numHiddenLayers = "num_hidden_layers"
        case _intermediateSize = "intermediate_size"
        case _numAttentionHeads = "num_attention_heads"
        case _numKeyValueHeads = "num_key_value_heads"
        case _vocabSize = "vocab_size"
        case _rmsNormEps = "rms_norm_eps"
        case _ropeTheta = "rope_theta"
        case _ropeTraditional = "rope_traditional"
        case _maxPositionEmbeddings = "max_position_embeddings"
        case _tieWordEmbeddings = "tie_word_embeddings"
        case _blockSize = "block_size"
        case _causalAttn = "causal_attn"
        case _textMaskTokenId = "text_mask_token_id"
        case _nullTokenId = "null_token_id"
        case _switchTokenId = "switch_token_id"
    }
}

// MARK: - Top level

public struct LocateAnythingConfiguration: Decodable, Sendable {
    public let textConfig: TextConfiguration
    public let visionConfig: VisionConfiguration

    private let _modelType: String?
    public var modelType: String { _modelType ?? "locateanything" }

    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int { _imageTokenIndex ?? 151665 }

    private let _boxStartTokenId: Int?
    public var boxStartTokenId: Int { _boxStartTokenId ?? 151668 }

    private let _boxEndTokenId: Int?
    public var boxEndTokenId: Int { _boxEndTokenId ?? 151669 }

    private let _coordStartTokenId: Int?
    public var coordStartTokenId: Int { _coordStartTokenId ?? 151677 }

    private let _coordEndTokenId: Int?
    public var coordEndTokenId: Int { _coordEndTokenId ?? 152677 }

    private let _refStartTokenId: Int?
    public var refStartTokenId: Int { _refStartTokenId ?? 151672 }

    private let _refEndTokenId: Int?
    public var refEndTokenId: Int { _refEndTokenId ?? 151673 }

    private let _noneTokenId: Int?
    public var noneTokenId: Int { _noneTokenId ?? 4064 }

    private let _mlpConnectorLayers: Int?
    public var mlpConnectorLayers: Int { _mlpConnectorLayers ?? 2 }

    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? 152681 }

    private let _nFutureTokens: Int?
    public var nFutureTokens: Int { _nFutureTokens ?? 6 }

    // eos_token_id may be a single Int or a list; we normalize to [Int].
    public let eosTokenIds: [Int]

    public let quantization: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case _modelType = "model_type"
        case _imageTokenIndex = "image_token_index"
        case _boxStartTokenId = "box_start_token_id"
        case _boxEndTokenId = "box_end_token_id"
        case _coordStartTokenId = "coord_start_token_id"
        case _coordEndTokenId = "coord_end_token_id"
        case _refStartTokenId = "ref_start_token_id"
        case _refEndTokenId = "ref_end_token_id"
        case _noneTokenId = "none_token_id"
        case _mlpConnectorLayers = "mlp_connector_layers"
        case _vocabSize = "vocab_size"
        case _nFutureTokens = "n_future_tokens"
        case eosTokenId = "eos_token_id"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfig = try c.decode(TextConfiguration.self, forKey: .textConfig)
        visionConfig = try c.decode(VisionConfiguration.self, forKey: .visionConfig)
        _modelType = try c.decodeIfPresent(String.self, forKey: ._modelType)
        _imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
        _boxStartTokenId = try c.decodeIfPresent(Int.self, forKey: ._boxStartTokenId)
        _boxEndTokenId = try c.decodeIfPresent(Int.self, forKey: ._boxEndTokenId)
        _coordStartTokenId = try c.decodeIfPresent(Int.self, forKey: ._coordStartTokenId)
        _coordEndTokenId = try c.decodeIfPresent(Int.self, forKey: ._coordEndTokenId)
        _refStartTokenId = try c.decodeIfPresent(Int.self, forKey: ._refStartTokenId)
        _refEndTokenId = try c.decodeIfPresent(Int.self, forKey: ._refEndTokenId)
        _noneTokenId = try c.decodeIfPresent(Int.self, forKey: ._noneTokenId)
        _mlpConnectorLayers = try c.decodeIfPresent(Int.self, forKey: ._mlpConnectorLayers)
        _vocabSize = try c.decodeIfPresent(Int.self, forKey: ._vocabSize)
        _nFutureTokens = try c.decodeIfPresent(Int.self, forKey: ._nFutureTokens)
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)

        // eos_token_id: Int OR [Int] OR absent.
        if let single = try? c.decode(Int.self, forKey: .eosTokenId) {
            eosTokenIds = [single]
        } else if let list = try? c.decode([Int].self, forKey: .eosTokenId) {
            eosTokenIds = list
        } else {
            eosTokenIds = [151645]
        }
    }
}
