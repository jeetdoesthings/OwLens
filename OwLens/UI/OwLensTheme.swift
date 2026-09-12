import SwiftUI
import UIKit

// MARK: - Typography

enum GeistWeight {
    case regular
    case medium
    case semiBold
    case bold

    var systemWeight: Font.Weight {
        switch self {
        case .regular:  return .regular
        case .medium:   return .medium
        case .semiBold: return .semibold
        case .bold:     return .bold
        }
    }
}

extension Font {
    static func geist(_ weight: GeistWeight = .regular, size: CGFloat) -> Font {
        .system(size: size, weight: weight.systemWeight)
    }

    static func geistMono(_ weight: GeistWeight = .regular, size: CGFloat) -> Font {
        .system(size: size, weight: weight.systemWeight, design: .monospaced).monospacedDigit()
    }
}

// MARK: - Theme & Color Tokens

enum OwLensTheme {
    // ── Monochromatic Accent: Pure White ──
    // Single neutral accent — no amber/gold. The UI speaks through
    // hierarchy (opacity) and spatial rhythm rather than colour.
    static let accent = Color.white
    
    // Legacy aliases kept so the rest of the codebase compiles without
    // a rename-everywhere pass. Every path converges to the same white.
    static let cinemaAmber = accent
    static let amberWarning = Color.white.opacity(0.85)
    static let cinemaGreen = accent
    static let cinemaCyan = accent
    static let cinemaAccent = accent

    // Active Recording Indicator (Red — strictly reserved for REC tally)
    static let recordingRed = Color(red: 235/255, green: 40/255, blue: 40/255)

    // Lock / Unlock Indicators (Green = Unlocked, Red = Locked)
    static let lockUnlocked = Color(red: 0.25, green: 0.88, blue: 0.40)
    static let lockLocked = recordingRed

    // ── Glass HUD Surfaces ──
    // Fewer layers, lower opacity — let the viewfinder breathe.
    static let glassBase = Color.black.opacity(0.45)
    static let glassBaseLight = Color.black.opacity(0.25)
    static let glassBaseHeavy = Color.black.opacity(0.65)
    static let glassActive = Color.white.opacity(0.92)
    static let glassActiveBg = Color.white.opacity(0.10)

    // ── Borders & Strokes ──
    // Thinner, subtler — borders should almost vanish.
    static let glassBorder = Color.white.opacity(0.10)
    static let glassBorderSubtle = Color.white.opacity(0.06)
    static let glassBorderActive = Color.white.opacity(0.22)
    static let glassBorderAmber = Color.white.opacity(0.20)
    static let glassBorderRed = recordingRed.opacity(0.5)

    // ── Text & Content Hierarchies ──
    static let textPrimary = Color.white.opacity(0.92)
    static let textSecondary = Color.white.opacity(0.55)
    static let textMuted = Color.white.opacity(0.32)
    static let textDisabled = Color.white.opacity(0.18)

    // ── Dimensions (Unified Curved-Rectangular Design) ──
    static let radiusCard: CGFloat = 8
    static let radiusSm: CGFloat = 6
    static let radiusMd: CGFloat = 8
    static let radiusLg: CGFloat = 10
    static let radiusXl: CGFloat = 16
    static let radiusPill: CGFloat = 8 // Transition legacy pill references to clean curved-rectangular
}

// MARK: - Haptic Feedback

enum Haptics {
    static func impact(_ style: UIImpactFeedbackGenerator.FeedbackStyle = .light) {
        let generator = UIImpactFeedbackGenerator(style: style)
        generator.prepare()
        generator.impactOccurred()
    }

    static func selection() {
        let generator = UISelectionFeedbackGenerator()
        generator.prepare()
        generator.selectionChanged()
    }

    static func notification(_ type: UINotificationFeedbackGenerator.FeedbackType) {
        let generator = UINotificationFeedbackGenerator()
        generator.prepare()
        generator.notificationOccurred(type)
    }
}

// MARK: - Glass View Modifiers

struct GlassPanelModifier: ViewModifier {
    var cornerRadius: CGFloat = OwLensTheme.radiusMd
    var border: Color = OwLensTheme.glassBorder
    var background: Color = OwLensTheme.glassBase
    var blurRadius: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(background)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(border, lineWidth: 0.5)
            )
    }
}

struct GlassPillModifier: ViewModifier {
    var isSelected: Bool = false
    var isDisabled: Bool = false
    var customBorder: Color? = nil

    func body(content: Content) -> some View {
        content
            .background(
                Capsule(style: .continuous)
                    .fill(isSelected ? OwLensTheme.glassActive : (isDisabled ? OwLensTheme.glassBase.opacity(0.4) : OwLensTheme.glassBase))
            )
            .overlay(
                Capsule(style: .continuous)
                    .strokeBorder(customBorder ?? (isSelected ? Color.clear : OwLensTheme.glassBorder), lineWidth: 0.5)
            )
    }
}

extension View {
    func glassPanel(
        cornerRadius: CGFloat = OwLensTheme.radiusMd,
        border: Color = OwLensTheme.glassBorder,
        background: Color = OwLensTheme.glassBase
    ) -> some View {
        modifier(GlassPanelModifier(cornerRadius: cornerRadius, border: border, background: background))
    }

    func glassPill(
        isSelected: Bool = false,
        isDisabled: Bool = false,
        customBorder: Color? = nil
    ) -> some View {
        modifier(GlassPillModifier(isSelected: isSelected, isDisabled: isDisabled, customBorder: customBorder))
    }
}
