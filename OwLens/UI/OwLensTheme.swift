import SwiftUI
import UIKit

// MARK: - Typography (Apple System San Francisco)

extension Font.Weight {
    static let semiBold: Font.Weight = .semibold
}

extension Font {
    static func appFont(_ weight: Font.Weight = .regular, size: CGFloat) -> Font {
        .system(size: size, weight: weight)
    }

    static func appMono(_ weight: Font.Weight = .regular, size: CGFloat) -> Font {
        .system(size: size, weight: weight, design: .monospaced).monospacedDigit()
    }

    // Compatibility aliases using default system font
    static func geist(_ weight: Font.Weight = .regular, size: CGFloat) -> Font {
        .system(size: size, weight: weight)
    }

    static func geistMono(_ weight: Font.Weight = .regular, size: CGFloat) -> Font {
        .system(size: size, weight: weight, design: .monospaced).monospacedDigit()
    }
}

// MARK: - Theme & Color Tokens

enum OwLensTheme {
    // Monochromatic Accent
    static let accent = Color.white
    
    // Cinema Amber / Gold (Reserved for Horizon Level Lock & IRE Guides)
    static let cinemaAmber = Color(red: 1.0, green: 0.82, blue: 0.25)
    static let levelAligned = cinemaAmber
    static let levelUnfinished = Color.white.opacity(0.50)
    static let amberWarning = Color(red: 1.0, green: 0.78, blue: 0.20)
    static let thermalWarning = Color(red: 1.0, green: 0.72, blue: 0.18)

    // Active Recording Indicator (Red — strictly reserved for REC tally)
    static let recordingRed = Color(red: 245/255, green: 45/255, blue: 45/255)
    static let thermalCritical = recordingRed

    // Audio VU Meter Colors
    static let audioNominal = Color(red: 0.25, green: 0.88, blue: 0.45)
    static let audioWarning = Color(red: 1.0, green: 0.78, blue: 0.20)
    static let audioPeak = Color(red: 1.0, green: 0.25, blue: 0.25)

    // Lock / Unlock Indicators
    static let lockUnlocked = Color(red: 0.25, green: 0.88, blue: 0.40)
    static let lockLocked = recordingRed

    // ── Glass HUD Surfaces ──
    static let glassBase = Color.black.opacity(0.40)
    static let glassBaseLight = Color.black.opacity(0.20)
    static let glassBaseHeavy = Color.black.opacity(0.60)
    static let glassActive = Color.white.opacity(0.95)
    static let glassActiveBg = Color.white.opacity(0.12)

    // ── Borders & Strokes ──
    static let glassBorder = Color.white.opacity(0.12)
    static let glassBorderSubtle = Color.white.opacity(0.06)
    static let glassBorderActive = Color.white.opacity(0.28)
    static let glassBorderAmber = cinemaAmber.opacity(0.50)
    static let glassBorderRed = recordingRed.opacity(0.6)

    // ── Text & Content Hierarchies ──
    static let textPrimary = Color.white.opacity(0.95)
    static let textSecondary = Color.white.opacity(0.65)
    static let textMuted = Color.white.opacity(0.38)
    static let textDisabled = Color.white.opacity(0.20)

    // ── Dimensions ──
    static let radiusCard: CGFloat = 10
    static let radiusSm: CGFloat = 6
    static let radiusMd: CGFloat = 10
    static let radiusLg: CGFloat = 12
    static let radiusXl: CGFloat = 16
    static let radiusPill: CGFloat = 10
    static let touchTargetMin: CGFloat = 44
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
    var useMaterial: Bool = true

    func body(content: Content) -> some View {
        content
            .background {
                if useMaterial {
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .fill(.ultraThinMaterial)
                        .overlay(
                            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                                .fill(background)
                        )
                } else {
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .fill(background)
                }
            }
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
    var useMaterial: Bool = true

    func body(content: Content) -> some View {
        content
            .background {
                if isSelected {
                    Capsule(style: .continuous)
                        .fill(OwLensTheme.glassActive)
                } else if useMaterial {
                    Capsule(style: .continuous)
                        .fill(.ultraThinMaterial)
                        .overlay(
                            Capsule(style: .continuous)
                                .fill(isDisabled ? OwLensTheme.glassBase.opacity(0.3) : OwLensTheme.glassBase)
                        )
                } else {
                    Capsule(style: .continuous)
                        .fill(isDisabled ? OwLensTheme.glassBase.opacity(0.4) : OwLensTheme.glassBase)
                }
            }
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
        background: Color = OwLensTheme.glassBase,
        useMaterial: Bool = true
    ) -> some View {
        modifier(GlassPanelModifier(cornerRadius: cornerRadius, border: border, background: background, useMaterial: useMaterial))
    }

    func glassPill(
        isSelected: Bool = false,
        isDisabled: Bool = false,
        customBorder: Color? = nil,
        useMaterial: Bool = true
    ) -> some View {
        modifier(GlassPillModifier(isSelected: isSelected, isDisabled: isDisabled, customBorder: customBorder, useMaterial: useMaterial))
    }
}
