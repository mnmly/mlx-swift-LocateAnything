import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import MLX

#if canImport(UniformTypeIdentifiers)
import UniformTypeIdentifiers
#endif

// Ported from mlx_vlm/models/locateanything/image_processing_locateanything.py
// and processing_locateanything.py.
//
// Image: dynamic-resolution resize (in_token_limit sqrt-downscale, then
// ceil-to-(merge*patch) padding), normalize to [-1, 1], patchify to
// [P, 3, patch, patch]. Text: ChatML prompt with <image-N> placeholders expanded
// to <img><IMG_CONTEXT>×N</img>.

public struct ProcessedImage {
    public let pixelValues: MLXArray   // [P, 3, patch, patch]
    public let gridH: Int
    public let gridW: Int
    public var grid: (Int, Int) { (gridH, gridW) }
    public var numPatches: Int { gridH * gridW }
}

/// Dynamic-resolution image preprocessing: resize to `(merge×patch)` multiples,
/// normalize to `[-1, 1]`, and patchify into `[P, 3, patch, patch]`.
public enum LocateAnythingImageProcessor {
    public static let patchSize = 14
    public static let inTokenLimit = 25600
    public static let mergeKernel = [2, 2]
    public static let imageMean: Float = 0.5
    public static let imageStd: Float = 0.5

    /// Compute (intermediate, target) sizes following the Python two-step logic.
    static func plannedSizes(width w: Int, height h: Int) -> (interW: Int, interH: Int, targetW: Int, targetH: Int) {
        var nw = w
        var nh = h
        if (w / patchSize) * (h / patchSize) > inTokenLimit {
            let scale = (Double(inTokenLimit) / Double((w / patchSize) * (h / patchSize))).squareRoot()
            nw = Int(Double(w) * scale)
            nh = Int(Double(h) * scale)
        }
        let padW = mergeKernel[1] * patchSize   // 28
        let padH = mergeKernel[0] * patchSize   // 28
        let tw = Int(ceil(Double(nw) / Double(padW))) * padW
        let th = Int(ceil(Double(nh) / Double(padH))) * padH
        return (nw, nh, tw, th)
    }

    public static func preprocess(_ image: CGImage) throws -> ProcessedImage {
        let plan = plannedSizes(width: image.width, height: image.height)
        if plan.targetW / patchSize >= 512 || plan.targetH / patchSize >= 512 {
            throw LocateAnythingError.imageTooLarge(plan.targetW, plan.targetH)
        }

        // Two-step resize matching Python (only downscale when over token limit).
        var current = image
        if plan.interW != image.width || plan.interH != image.height {
            current = try resize(current, width: plan.interW, height: plan.interH)
        }
        if plan.targetW != current.width || plan.targetH != current.height {
            current = try resize(current, width: plan.targetW, height: plan.targetH)
        }

        // Extract sRGB float [H, W, 3] in [0, 1].
        let hwc = try floatRGB(current)                       // [H, W, 3]
        var x = hwc.transposed(2, 0, 1)                        // [3, H, W]
        x = (x - imageMean) / imageStd                         // -> [-1, 1]

        let c = 3
        let h = plan.targetH
        let w = plan.targetW
        let p = patchSize
        let gh = h / p
        let gw = w / p
        let patches = x.reshaped([c, gh, p, gw, p])
            .transposed(1, 3, 0, 2, 4)
            .reshaped([gh * gw, c, p, p])
        return ProcessedImage(pixelValues: patches, gridH: gh, gridW: gw)
    }

    /// Concatenate multiple processed images into one pixel_values tensor + grids.
    public static func batch(_ images: [ProcessedImage]) -> (pixelValues: MLXArray, grids: [(Int, Int)]) {
        let pv = concatenated(images.map { $0.pixelValues }, axis: 0)
        return (pv, images.map { $0.grid })
    }

    // MARK: CoreGraphics helpers

    static func resize(_ image: CGImage, width: Int, height: Int) throws -> CGImage {
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: 0, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw LocateAnythingError.imageDecodeFailed }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let out = ctx.makeImage() else { throw LocateAnythingError.imageDecodeFailed }
        return out
    }

    /// Render into an RGBA8 sRGB bitmap and return [H, W, 3] float32 in [0, 1].
    static func floatRGB(_ image: CGImage) throws -> MLXArray {
        let w = image.width
        let h = image.height
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        var buffer = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = buffer.withUnsafeMutableBytes({ ptr -> CGContext? in
            CGContext(
                data: ptr.baseAddress, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: w * 4, space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        }) else { throw LocateAnythingError.imageDecodeFailed }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

        // Drop alpha, scale to [0, 1].
        var rgb = [Float](repeating: 0, count: w * h * 3)
        for i in 0 ..< (w * h) {
            rgb[i * 3 + 0] = Float(buffer[i * 4 + 0]) / 255.0
            rgb[i * 3 + 1] = Float(buffer[i * 4 + 1]) / 255.0
            rgb[i * 3 + 2] = Float(buffer[i * 4 + 2]) / 255.0
        }
        return MLXArray(rgb, [h, w, 3])
    }

    public static func loadCGImage(path: String) throws -> CGImage {
        let url = URL(fileURLWithPath: path)
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw LocateAnythingError.imageDecodeFailed }
        var orientation = CGImagePropertyOrientation.up
        if let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
           let raw = props[kCGImagePropertyOrientation] as? UInt32,
           let o = CGImagePropertyOrientation(rawValue: raw) {
            orientation = o
        }
        return baked(img, orientation: orientation)
    }

    /// Force-decode into a memory-backed bitmap with no file backing, applying the
    /// EXIF `orientation` so both display and the model see upright pixels.
    ///
    /// `CGImageSource` decodes lazily on first pixel access and does *not* apply
    /// EXIF orientation; a sandboxed GUI must bake the pixels (upright) while access
    /// is valid, then hand the result to the session (never a path).
    public static func baked(_ image: CGImage, orientation: CGImagePropertyOrientation = .up) -> CGImage {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        if orientation != .up {
            let ci = CIImage(cgImage: image).oriented(orientation)
            let ctx = CIContext(options: [.outputColorSpace: cs, .workingColorSpace: cs])
            if let out = ctx.createCGImage(ci, from: ci.extent, format: .RGBA8, colorSpace: cs) {
                return out
            }
        }
        guard let ctx = CGContext(
            data: nil, width: image.width, height: image.height, bitsPerComponent: 8,
            bytesPerRow: 0, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return image }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return ctx.makeImage() ?? image
    }
}

// MARK: - Text / prompt

/// Builds the model's ChatML prompt and expands `<image-N>` placeholders into the
/// per-image `<img><IMG_CONTEXT>×N</img>` spans the model expects.
public enum LocateAnythingPrompt {
    public static let imageToken = "<IMG_CONTEXT>"
    public static let imageStartToken = "<img>"
    public static let imageEndToken = "</img>"
    public static let systemPrompt = "You are a helpful assistant."

    /// Build the ChatML prompt. Matches locateanything_worker._build_messages:
    /// main image (`<image-1>`) precedes the question; any additional
    /// (visual-prompt) images follow it.
    public static func chatML(question: String, numImages: Int) -> String {
        var userContent = ""
        if numImages >= 1 { userContent += "<image-1>" }
        userContent += question
        if numImages >= 2 {
            for i in 2 ... numImages { userContent += "<image-\(i)>" }
        }
        return "<|im_start|>system\n\(systemPrompt)<|im_end|>\n"
            + "<|im_start|>user\n\(userContent)<|im_end|>\n"
            + "<|im_start|>assistant\n"
    }

    /// Build the ChatML prompt for a video. Mirrors the reference expansion of
    /// `<video-N>`: `"The first video sampled with X fps: Frame-1-0.00s: <img>…</img>Frame-2-…"`,
    /// followed by the question.
    public static func videoChatML(
        question: String, frames: [ProcessedImage], timestamps: [Double],
        sampledFPS: Float, mergeLength: Int
    ) -> String {
        var content = String(format: "The first video sampled with %.2f fps: ", sampledFPS)
        for (i, frame) in frames.enumerated() {
            let count = frame.numPatches / mergeLength
            let ts = i < timestamps.count ? timestamps[i] : Double(i)
            content += String(format: "Frame-%d-%.2fs: ", i + 1, ts)
            content += imageStartToken + String(repeating: imageToken, count: count) + imageEndToken
        }
        content += question
        return "<|im_start|>system\n\(systemPrompt)<|im_end|>\n"
            + "<|im_start|>user\n\(content)<|im_end|>\n"
            + "<|im_start|>assistant\n"
    }

    /// Expand each <image-N> to <img><IMG_CONTEXT>×(patches/mergeLength)</img>.
    public static func expandPlaceholders(_ prompt: String, grids: [(Int, Int)], mergeLength: Int) -> String {
        var result = prompt
        for (i, g) in grids.enumerated() {
            let count = (g.0 * g.1) / mergeLength
            let replacement = imageStartToken
                + String(repeating: imageToken, count: count)
                + imageEndToken
            result = result.replacingOccurrences(of: "<image-\(i + 1)>", with: replacement)
        }
        return result
    }
}

/// The grounding tasks LocateAnything supports, with the exact instruction
/// templates used by the reference (matches the NVIDIA HF Space).
public enum LocateAnythingTask: String, Sendable, CaseIterable {
    case detection
    case grounding
    case ocr
    case gui
    case pointing
    case custom

    public var title: String {
        switch self {
        case .detection: return "Detection"
        case .grounding: return "Grounding"
        case .ocr: return "OCR"
        case .gui: return "GUI"
        case .pointing: return "Pointing"
        case .custom: return "Custom"
        }
    }

    /// Whether the task consumes the description field.
    public var usesDescription: Bool { self != .ocr }

    /// Placeholder / hint for the description field.
    public var descriptionHint: String {
        switch self {
        case .detection: return "categories, comma-separated (person, car, bicycle)"
        case .grounding: return "phrase to ground (people wearing red shirts)"
        case .gui: return "UI element (the search button)"
        case .pointing: return "what to point at (the traffic light)"
        case .ocr: return "no input needed"
        case .custom: return "full instruction"
        }
    }

    /// Build the instruction. Detection joins comma-separated categories with
    /// `</c>`; grounding / GUI / pointing use the phrase verbatim; OCR is fixed;
    /// custom passes the text through unchanged.
    public func instruction(_ description: String) -> String {
        let text = description.trimmingCharacters(in: .whitespacesAndNewlines)
        switch self {
        case .detection:
            let cats = text.split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
                .joined(separator: "</c>")
            return "Locate all the instances that matches the following description: \(cats)."
        case .grounding:
            return "Locate all the instances that match the following description: \(text)."
        case .ocr:
            return "Detect all the text in box format."
        case .gui:
            return "Locate the region that matches the following description: \(text)."
        case .pointing:
            return "Point to: \(text)."
        case .custom:
            return text
        }
    }
}

public enum LocateAnythingError: Error, CustomStringConvertible {
    case imageDecodeFailed
    case imageTooLarge(Int, Int)
    case invalidRepoID(String)
    case weightsNotFound(String)
    case tokenizerMissingImageToken

    public var description: String {
        switch self {
        case .imageDecodeFailed: return "Failed to decode/resize image."
        case .imageTooLarge(let w, let h): return "Image grid exceeds position-embedding limit (target \(w)x\(h), max 512 patches/side)."
        case .invalidRepoID(let id): return "Invalid Hugging Face repo id: \(id)"
        case .weightsNotFound(let p): return "No .safetensors weights found in \(p)"
        case .tokenizerMissingImageToken: return "Tokenizer does not define <IMG_CONTEXT>."
        }
    }
}
