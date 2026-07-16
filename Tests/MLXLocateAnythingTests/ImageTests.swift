import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest

@testable import MLXLocateAnything

/// Regression: EXIF orientation must be applied on load so both the display and
/// the model see upright pixels. (Pure — no MLX; runs under `swift test`.)
final class ImageTests: XCTestCase {
    func testExifOrientationIsApplied() throws {
        // A 40×20 landscape bitmap tagged orientation .right (6): to display
        // upright it rotates 90°, so the loaded image should be 20×40.
        let w = 40, h = 20
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        let source = ctx.makeImage()!

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("la_orient_test_\(UUID().uuidString).jpg")
        defer { try? FileManager.default.removeItem(at: url) }
        let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.jpeg.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(
            dest, source,
            [kCGImagePropertyOrientation: 6] as CFDictionary)   // 6 = .right
        XCTAssertTrue(CGImageDestinationFinalize(dest))

        let loaded = try LocateAnythingImageProcessor.loadCGImage(path: url.path)
        XCTAssertEqual(loaded.width, 20, "orientation not applied — width should swap")
        XCTAssertEqual(loaded.height, 40, "orientation not applied — height should swap")
    }

    func testUprightImageUnchanged() throws {
        let w = 30, h = 18
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let src = ctx.makeImage()!
        let baked = LocateAnythingImageProcessor.baked(src)   // .up default
        XCTAssertEqual(baked.width, 30)
        XCTAssertEqual(baked.height, 18)
    }
}
