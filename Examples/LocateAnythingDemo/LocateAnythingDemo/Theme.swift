import SwiftUI

/// Visual constants. apple-design: critically-damped springs by default (no
/// overshoot); reserve a little bounce only for momentum-carrying motion.
enum Theme {
    /// Default UI spring — critically damped, snappy settle.
    static let spring = Animation.spring(response: 0.38, dampingFraction: 1.0)
    /// Slightly bouncy — only for elements that "arrive" (boxes landing).
    static let arrive = Animation.spring(response: 0.42, dampingFraction: 0.82)

    /// Distinct, legible box colors cycled per detection.
    static let boxColors: [Color] = [
        .green, .blue, .orange, .pink, .purple, .teal, .yellow, .mint, .cyan, .indigo,
    ]

    static func boxColor(_ i: Int) -> Color { boxColors[i % boxColors.count] }

    /// Deterministic color per label (same class → same color, like the HF Space);
    /// falls back to cycling by index for unlabeled detections.
    static func color(forLabel label: String?, index: Int) -> Color {
        guard let label, !label.isEmpty else { return boxColor(index) }
        let h = label.unicodeScalars.reduce(0) { ($0 &* 31 &+ Int($1.value)) & 0x7fff_ffff }
        return boxColors[h % boxColors.count]
    }
}
