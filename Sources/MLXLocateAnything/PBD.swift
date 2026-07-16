import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Ported from mlx_vlm/models/locateanything/pbd.py — Parallel Box Decoding.
//
// Each step speculatively appends a "bridge" token + (block_size-1) mask tokens
// with decremented position ids and a MAGI block-attention mask, reads the
// last-B logits (one full box/point/ref in a single forward), then trims the
// speculative block from the KV cache. Top-k threshold sampling decides whether
// the block is a valid box, a referring phrase, or a fallback; hybrid mode drops
// to autoregressive decoding on a malformed box and resumes MTP after the box.

/// Decoding strategy for ``LocateAnythingSession/predict(image:question:maxNewTokens:mode:visualPromptBox:visualPromptBoxFormat:)``.
public enum PBDMode: String, Sendable, CaseIterable {
    /// Pure Parallel Box Decoding — force-accepts each speculative box block.
    case fast
    /// Plain autoregressive decoding driven by the PBD grammar.
    case slow
    /// Parallel Box Decoding with an autoregressive fallback for malformed boxes (default).
    case hybrid
    /// Standard token-by-token greedy decoding (baseline, no PBD).
    case ar
}

struct PBDTokenIDs {
    let boxStart, boxEnd, coordStart, coordEnd: Int
    let refStart, refEnd, noneId, nullId, switchId, defaultMask, imEnd: Int

    init(_ c: LocateAnythingConfiguration) {
        boxStart = c.boxStartTokenId
        boxEnd = c.boxEndTokenId
        coordStart = c.coordStartTokenId
        coordEnd = c.coordEndTokenId
        refStart = c.refStartTokenId
        refEnd = c.refEndTokenId
        noneId = c.noneTokenId
        nullId = c.textConfig.nullTokenId
        switchId = c.textConfig.switchTokenId
        defaultMask = c.textConfig.textMaskTokenId
        imEnd = c.eosTokenIds.first ?? 151645
    }
}

private enum BoxFrame { case empty, legal, illegal }

private struct BlockPattern {
    let type: String
    let tokens: [Int]
    let needSwitchToAR: Bool
    let isTerminal: Bool
}

public final class PBDDecoder {
    let model: LocateAnythingModel
    let ids: PBDTokenIDs
    let mode: PBDMode
    let blockSize: Int

    public init(model: LocateAnythingModel, mode: PBDMode) {
        self.model = model
        self.ids = PBDTokenIDs(model.config)
        self.mode = mode
        self.blockSize = model.config.textConfig.blockSize
        precondition(blockSize == 6, "PBD assumes block_size == 6")
    }

    // MARK: - CPU-side sampling helpers (operate on materialized probs)

    /// Top-k ids + probs per row of `sub` [R, V], via device argsort.
    private func topK(_ sub: MLXArray, _ keepK: Int) -> (ids: [[Int]], probs: [[Float]]) {
        let order = argSort(-sub, axis: -1)[0..., ..<keepK]      // [R, keepK]
        let topProbs = takeAlong(sub, order, axis: -1)           // [R, keepK]
        MLX.eval(order, topProbs)
        let r = sub.dim(0)
        let oFlat = order.asType(.int32).asArray(Int32.self)
        let pFlat = topProbs.asType(.float32).asArray(Float.self)
        var ids: [[Int]] = []
        var probs: [[Float]] = []
        ids.reserveCapacity(r)
        probs.reserveCapacity(r)
        for i in 0 ..< r {
            ids.append((0 ..< keepK).map { Int(oFlat[i * keepK + $0]) })
            probs.append((0 ..< keepK).map { pFlat[i * keepK + $0] })
        }
        return (ids, probs)
    }

    private func isValidBoxFrame(_ probs: MLXArray, startThresh: Float, endThresh: Float) -> BoxFrame {
        func p(_ r: Int, _ c: Int) -> Float { probs[r, c].item(Float.self) }
        if p(0, ids.boxStart) >= startThresh {
            if p(1, ids.noneId) > 0.2 && p(2, ids.boxEnd) > 0.2
                && p(3, ids.nullId) > 0.1 && p(4, ids.nullId) > 0.1 {
                return .empty
            }
        }
        let pStart = p(0, ids.boxStart)
        if pStart < p(0, ids.imEnd) || pStart < p(0, ids.nullId) { return .illegal }
        let endScore = p(5, ids.boxEnd) + p(5, ids.nullId) + p(5, ids.imEnd)
        return endScore >= endThresh ? .legal : .illegal
    }

    private func decodeBboxAvg(_ probs: MLXArray, keepK: Int = 5, startThresh: Float = 0.7, endThresh: Float = 0.2) -> [Int]? {
        switch isValidBoxFrame(probs, startThresh: startThresh, endThresh: endThresh) {
        case .empty:
            return [ids.boxStart, ids.noneId, ids.boxEnd, ids.nullId, ids.nullId, ids.nullId]
        case .illegal:
            return nil
        case .legal:
            break
        }
        let sub = probs[1 ..< 5]                                  // [4, V]
        let (tids, tprobs) = topK(sub, keepK)
        var final: [Int] = []
        for i in 0 ..< 4 {
            let valid = zip(tids[i], tprobs[i]).filter { ids.coordStart <= $0.0 && $0.0 <= ids.coordEnd }
            guard let (firstId, firstP) = valid.first else { return nil }
            if mode == .hybrid {
                let validIds = valid.map { $0.0 }
                let abnormal = firstP < 0.9 && validIds.count > 1
                    && (validIds.max()! - validIds.min()!) > 60
                final.append(abnormal ? 0 : firstId)
            } else {
                final.append(firstId)
            }
        }
        return [ids.boxStart] + final + [ids.boxEnd]
    }

    private func decodeRef(_ probs: MLXArray, keepK: Int = 5, startThresh: Float = 0.6) -> [Int]? {
        if probs[0, ids.refStart].item(Float.self) < startThresh { return nil }
        let sub = probs[1...]                                     // [B-1, V]
        let (tids, _) = topK(sub, keepK)
        var final: [Int] = []
        for row in tids {
            guard let first = row.first(where: { !(ids.coordStart <= $0 && $0 <= ids.coordEnd) }) else { return nil }
            final.append(first)
        }
        return [ids.refStart] + final
    }

    private func sampleBlock(_ blockLogits: MLXArray, keepK: Int = 5) -> [Int] {
        let probs = softmax(blockLogits.asType(.float32), axis: -1)  // [B, V]
        MLX.eval(probs)
        let x0 = argMax(probs, axis: -1).asType(.int32).asArray(Int32.self).map { Int($0) }
        if let box = decodeBboxAvg(probs, keepK: keepK) { return box }
        if let ref = decodeRef(probs, keepK: keepK) { return ref }
        return x0
    }

    private func handlePattern(_ x0: [Int]) -> BlockPattern {
        if x0[0] == ids.nullId || x0[0] == ids.imEnd {
            return BlockPattern(type: "im_end", tokens: [ids.imEnd], needSwitchToAR: false, isTerminal: true)
        }
        if x0.count >= 2 && x0[0] == ids.boxStart && x0[1] == ids.noneId {
            return BlockPattern(type: "empty_box", tokens: [ids.boxStart, ids.noneId, ids.boxEnd], needSwitchToAR: false, isTerminal: false)
        }
        if x0[0] == ids.boxStart {
            var coordIx = 1
            for coord in x0[1 ..< min(5, x0.count)] {
                if ids.coordStart <= coord && coord <= ids.coordEnd { coordIx += 1 } else { break }
            }
            if coordIx == 5 && x0.count > 5 && x0[5] == ids.boxEnd {
                return BlockPattern(type: "coord_box", tokens: x0, needSwitchToAR: false, isTerminal: false)
            }
            if coordIx == 3 && x0.count > 3 && x0[3] == ids.boxEnd {
                return BlockPattern(type: "point_box", tokens: Array(x0[0 ..< 4]), needSwitchToAR: false, isTerminal: false)
            }
            if mode == .fast {
                return BlockPattern(type: "coord_box", tokens: x0, needSwitchToAR: false, isTerminal: false)
            }
            return BlockPattern(type: "error_box", tokens: Array(x0[0 ..< coordIx]), needSwitchToAR: true, isTerminal: false)
        }
        var tokens = x0
        if let idx = tokens.firstIndex(of: ids.nullId) { tokens = Array(tokens[0 ..< idx]) }
        if tokens.count >= 2 && tokens[tokens.count - 1] == ids.refEnd && tokens[tokens.count - 2] == ids.refEnd {
            tokens = Array(tokens[0 ..< tokens.count - 1])
        }
        return BlockPattern(type: "ref_object", tokens: tokens, needSwitchToAR: false, isTerminal: false)
    }

    private func consumeBlock(_ blockLogits: MLXArray) -> (String, [Int]) {
        let x0 = sampleBlock(blockLogits)
        let pat = handlePattern(x0)
        return (pat.type, pat.tokens)
    }

    // MARK: - Forwards

    private func forwardMTP(_ generated: [Int], _ cache: [KVCache]) -> MLXArray {
        let b = blockSize
        let acc = cache[0].offset
        let tail = Array(generated[acc...])
        var window = tail
        window.append(generated[generated.count - 1])
        window.append(contentsOf: Array(repeating: ids.defaultMask, count: b - 1))
        let qLen = window.count
        let kvLen = acc + qLen

        var positions = Array(acc ..< (acc + qLen))
        for i in 0 ..< b { positions[positions.count - 1 - i] -= 1 }
        let positionIds = MLXArray(positions.map { Int32($0) }, [1, qLen])

        let mask = buildMagiBlockMask(kvLen: kvLen, qLen: qLen, blockSize: b)
        let inputs = MLXArray(window.map { Int32($0) }, [1, qLen])
        let logits = model.languageForward(
            inputs: inputs, inputsEmbeds: nil, mask: .array(mask), cache: cache, positionIds: positionIds)
        let blockLogits = logits[0..., (qLen - b) ..< qLen, 0...].reshaped([b, -1])
        MLX.eval(blockLogits)
        for c in cache { _ = c.trim(b) }
        return blockLogits
    }

    private func mtpPrefill(_ inputsEmbeds: MLXArray, _ cache: [KVCache]) -> MLXArray {
        let b = blockSize
        let l = inputsEmbeds.dim(1)
        let d = inputsEmbeds.dim(2)
        let bridge = inputsEmbeds[0..., (l - 1) ..< l, 0...]                 // [1, 1, D]
        let maskEmbed = model.embed(MLXArray([Int32(ids.defaultMask)], [1, 1]))  // [1, 1, D]
        let maskBlock = broadcast(maskEmbed, to: [1, b - 1, d])
        let window = concatenated([inputsEmbeds, bridge, maskBlock], axis: 1)   // [1, L+B, D]
        let qLen = window.dim(1)
        let kvLen = qLen

        var positions = Array(0 ..< qLen)
        for i in 0 ..< b { positions[positions.count - 1 - i] -= 1 }
        let positionIds = MLXArray(positions.map { Int32($0) }, [1, qLen])

        let mask = buildMagiBlockMask(kvLen: kvLen, qLen: qLen, blockSize: b)
        let logits = model.languageForward(
            inputs: nil, inputsEmbeds: window, mask: .array(mask), cache: cache, positionIds: positionIds)
        let blockLogits = logits[0..., (qLen - b) ..< qLen, 0...].reshaped([b, -1])
        MLX.eval(blockLogits)
        for c in cache { _ = c.trim(b) }
        return blockLogits
    }

    private func forwardAR(_ generated: [Int], _ cache: [KVCache]) -> MLXArray {
        let acc = cache[0].offset
        let tail = Array(generated[acc...])
        let inputs = MLXArray(tail.map { Int32($0) }, [1, tail.count])
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = tail.count > 1 ? .causal : .none
        let logits = model.languageForward(
            inputs: inputs, inputsEmbeds: nil, mask: mask, cache: cache, positionIds: nil)
        let n = logits.dim(1)
        return logits[0..., (n - 1) ..< n, 0...].reshaped([-1])
    }

    private func sampleAR(_ logits: MLXArray) -> (String, Int) {
        let token = argMax(logits, axis: -1).item(Int.self)
        if mode == .hybrid {
            if token == ids.boxEnd { return ("box_end_ar", token) }
            if (ids.coordStart <= token && token <= ids.coordEnd) || token == ids.noneId {
                return ("coord_ar", token)
            }
            return ("im_end", token)
        }
        return (token == ids.imEnd ? "im_end" : "continue_ar", token)
    }

    // MARK: - Generate

    /// Returns the newly generated token ids (excluding the prompt).
    public func generate(prompt: [Int], inputsEmbeds: MLXArray, cache: [KVCache], maxTokens: Int) -> [Int] {
        var generated = prompt
        let promptLen = prompt.count
        var useMTP = (mode == .fast || mode == .hybrid)

        if useMTP {
            let block = mtpPrefill(inputsEmbeds, cache)
            let (type, tokens) = consumeBlock(block)
            generated.append(contentsOf: tokens)
            if type == "im_end" { return trimmed(generated, promptLen, maxTokens) }
            if mode == .hybrid && type == "error_box" { useMTP = false }
        } else {
            // slow prefill
            let logits = model.languageForward(
                inputs: nil, inputsEmbeds: inputsEmbeds, mask: .causal, cache: cache, positionIds: nil)
            let n = logits.dim(1)
            let last = logits[0..., (n - 1) ..< n, 0...].reshaped([-1])
            let (type, token) = sampleAR(last)
            generated.append(token)
            if type == "im_end" { return trimmed(generated, promptLen, maxTokens) }
        }

        while generated.count < promptLen + maxTokens {
            if useMTP {
                let block = forwardMTP(generated, cache)
                let (type, tokens) = consumeBlock(block)
                generated.append(contentsOf: tokens)
                if type == "im_end" { break }
                if mode == .hybrid && type == "error_box" { useMTP = false }
            } else {
                let logits = forwardAR(generated, cache)
                let (type, token) = sampleAR(logits)
                generated.append(token)
                if type == "im_end" { break }
                if mode == .hybrid && type == "box_end_ar" { useMTP = true }
            }
        }

        return trimmed(generated, promptLen, maxTokens)
    }

    private func trimmed(_ generated: [Int], _ promptLen: Int, _ maxTokens: Int) -> [Int] {
        let end = min(generated.count, promptLen + maxTokens)
        guard promptLen < end else { return [] }
        return Array(generated[promptLen ..< end])
    }
}
