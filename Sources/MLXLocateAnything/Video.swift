import AVFoundation
import CoreGraphics
import Foundation

// Video frame sampling + prompt, ported from the reference
// processing_locateanything.py (eaglevl). Frames are sampled at `targetFPS`
// (capped at `maxFrames`), globally downscaled to a total-pixel budget, then each
// frame is preprocessed like an image and grounded together (the vision tower
// attends within each frame via a block-diagonal mask).
//
// The frontend (AVFoundation) extracts the CGImage frames for the planned indices
// and hands them to ``LocateAnythingSession/predictVideo(frames:timestamps:sampledFPS:displayWidth:displayHeight:question:maxNewTokens:mode:sampling:)``.

public enum LocateAnythingVideo {
    /// Target sampling rate (frames/second).
    public static let targetFPS: Double = 2.0
    /// Maximum number of sampled frames.
    public static let maxFrames = 64
    /// Total pixel budget across all sampled frames (reference: 32000·28·28·0.9).
    public static let videoTotalPixels = Int(32000.0 * 28.0 * 28.0 * 0.9)

    public struct SamplePlan: Sendable {
        /// Original frame indices to extract.
        public let indices: [Int]
        /// Timestamp (seconds) of each sampled frame.
        public let timestamps: [Double]
        /// Effective sampled frame-rate reported in the prompt.
        public let sampledFPS: Float
    }

    /// Decide which frames to sample: resample to `targetFPS`, cap at `frameCap`,
    /// then take evenly-spaced indices (`linspace`), mirroring the reference.
    /// - Parameter frameCap: max frames to sample (reference default: 64). Lower it
    ///   to trade temporal coverage for speed — cost scales with total image tokens.
    public static func plan(totalFrames: Int, videoFPS: Double, frameCap: Int = maxFrames) -> SamplePlan {
        let total = max(totalFrames, 1)
        var n = Int((Double(total) / max(videoFPS, 1e-6) * targetFPS).rounded())
        n = max(1, min(n, min(max(frameCap, 1), total)))

        let indices: [Int]
        if n >= total {
            indices = Array(0 ..< total)
        } else if n == 1 {
            indices = [0]
        } else {
            indices = (0 ..< n).map { Int((Double($0) * Double(total - 1) / Double(n - 1)).rounded()) }
        }
        let sampledFPS = Float(Double(n) / Double(total) * videoFPS)
        let timestamps = indices.map { Double($0) / max(videoFPS, 1e-6) }
        return SamplePlan(indices: indices, timestamps: timestamps, sampledFPS: sampledFPS)
    }

    /// Global downscale factor so `frameCount · width · height` fits the pixel budget.
    public static func frameScale(frameCount: Int, width: Int, height: Int) -> Double {
        let current = Double(frameCount) * Double(width) * Double(height)
        guard current > Double(videoTotalPixels) else { return 1.0 }
        return (Double(videoTotalPixels) / current).squareRoot()
    }

    /// Sampled frames ready to hand to
    /// ``LocateAnythingSession/predictVideo(frames:timestamps:sampledFPS:displayWidth:displayHeight:question:maxNewTokens:mode:sampling:)``.
    ///
    /// `@unchecked Sendable`: `CGImage` is an immutable, thread-safe CoreFoundation
    /// type, and these frames are baked (memory-backed) at extraction time.
    public struct VideoFrames: @unchecked Sendable {
        public let frames: [CGImage]
        public let timestamps: [Double]
        public let sampledFPS: Float
        public let width: Int
        public let height: Int
    }

    /// Decode a video and extract the planned frames (2 fps, max 64) via AVFoundation.
    ///
    /// Frames are baked to memory-backed bitmaps and the track's preferred transform
    /// is applied, so rotated video decodes upright. The caller must have file access
    /// (a sandboxed app should open via a panel/drop first).
    public static func extractFrames(url: URL, frameCap: Int = maxFrames) async throws -> VideoFrames {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw LocateAnythingError.imageDecodeFailed
        }
        let rate = Double(try await track.load(.nominalFrameRate))
        let fps = rate > 0 ? rate : 30
        let totalFrames = max(1, Int((duration * fps).rounded()))
        let plan = plan(totalFrames: totalFrames, videoFPS: fps, frameCap: frameCap)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true   // honor rotation
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        var images: [CGImage] = []
        var stamps: [Double] = []
        for t in plan.timestamps {
            let time = CMTime(seconds: t, preferredTimescale: 600)
            if let (cg, _) = try? await generator.image(at: time) {
                images.append(LocateAnythingImageProcessor.baked(cg))
                stamps.append(t)
            }
        }
        guard let first = images.first else { throw LocateAnythingError.imageDecodeFailed }
        return VideoFrames(
            frames: images, timestamps: stamps, sampledFPS: plan.sampledFPS,
            width: first.width, height: first.height)
    }
}
