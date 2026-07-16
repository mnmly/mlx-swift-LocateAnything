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
}
