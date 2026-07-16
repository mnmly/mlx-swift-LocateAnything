import CoreGraphics
import MLXLocateAnything
import SwiftUI

/// The input image with detected boxes/points overlaid. Coordinates are
/// normalized 0…1 relative to the image; we map them onto the aspect-fit rect.
struct BoxOverlayView: View {
    let image: CGImage
    let boxes: [DetectedBox]
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GeometryReader { geo in
            let fit = Self.fitRect(CGFloat(image.width), CGFloat(image.height), in: geo.size)
            ZStack(alignment: .topLeading) {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .frame(width: geo.size.width, height: geo.size.height)
                    .clipShape(RoundedRectangle(cornerRadius: 12))

                ForEach(Array(boxes.enumerated()), id: \.offset) { idx, box in
                    BoxMark(box: box, index: idx, total: boxes.count, fit: fit, reduceMotion: reduceMotion)
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
        }
    }

    static func fitRect(_ imgW: CGFloat, _ imgH: CGFloat, in size: CGSize) -> CGRect {
        guard imgW > 0, imgH > 0 else { return .zero }
        let scale = min(size.width / imgW, size.height / imgH)
        let w = imgW * scale
        let h = imgH * scale
        return CGRect(x: (size.width - w) / 2, y: (size.height - h) / 2, width: w, height: h)
    }
}

private struct BoxMark: View {
    let box: DetectedBox
    let index: Int
    let total: Int
    let fit: CGRect
    let reduceMotion: Bool
    @State private var shown = false

    private var color: Color { Theme.color(forLabel: box.label, index: index) }

    var body: some View {
        Group {
            if box.isPoint { pointMark } else { boxMark }
        }
        .opacity(shown ? 1 : 0)
        .scaleEffect(shown ? 1 : (reduceMotion ? 1 : 0.88), anchor: .center)
        .onAppear {
            // Compress the per-mark stagger so the whole set lands within ~1.2s,
            // even with 100+ detections (otherwise late marks trail in for seconds).
            let perDelay = min(0.045, 1.2 / Double(max(total, 1)))
            let anim = reduceMotion
                ? Animation.easeOut(duration: 0.2)
                : Theme.arrive.delay(Double(index) * perDelay)
            withAnimation(anim) { shown = true }
        }
    }

    private var boxMark: some View {
        let r = viewRect
        return RoundedRectangle(cornerRadius: 5)
            .strokeBorder(color, lineWidth: 2.5)
            .background(RoundedRectangle(cornerRadius: 5).fill(color.opacity(0.12)))
            .frame(width: r.width, height: r.height)
            .overlay(alignment: .topLeading) {
                label
                    .fixedSize()
                    .alignmentGuide(.top) { $0[.bottom] + 3 }   // sit just above the box
                    .alignmentGuide(.leading) { _ in 0 }
            }
            .offset(x: r.minX, y: r.minY)
    }

    private var pointMark: some View {
        let p = viewPoint
        return ZStack {
            Circle().fill(color.opacity(0.25)).frame(width: 26, height: 26)
            Circle().strokeBorder(color, lineWidth: 2.5).frame(width: 14, height: 14)
            label.fixedSize().offset(y: -24)
        }
        .position(x: p.x, y: p.y)
    }

    private var label: some View {
        Text(box.label ?? "#\(index + 1)")
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color, in: Capsule())
            .shadow(color: .black.opacity(0.25), radius: 1.5, y: 0.5)
    }

    /// Box in view coordinates, tolerant of inverted (malformed) coordinates.
    private var viewRect: CGRect {
        let b = box.box
        guard b.count >= 4 else { return .zero }
        let x1 = min(b[0], b[2]), x2 = max(b[0], b[2])
        let y1 = min(b[1], b[3]), y2 = max(b[1], b[3])
        return CGRect(
            x: fit.minX + CGFloat(x1) * fit.width,
            y: fit.minY + CGFloat(y1) * fit.height,
            width: CGFloat(x2 - x1) * fit.width,
            height: CGFloat(y2 - y1) * fit.height)
    }

    private var viewPoint: CGPoint {
        let b = box.box
        guard b.count >= 2 else { return .zero }
        return CGPoint(
            x: fit.minX + CGFloat(b[0]) * fit.width,
            y: fit.minY + CGFloat(b[1]) * fit.height)
    }
}
