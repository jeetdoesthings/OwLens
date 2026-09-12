import SwiftUI

struct ScopesOverlay: View {
    let data: ScopeData

    var body: some View {
        VStack(spacing: 4) {
            scopeBlock(title: "HISTOGRAM", height: 26) {
                histogramCanvas
            }
            scopeBlock(title: "WAVEFORM", height: 32) {
                waveformCanvas
            }
        }
        .frame(width: 80)
        .padding(5)
        .glassPanel(cornerRadius: OwLensTheme.radiusCard, border: OwLensTheme.glassBorderActive, background: OwLensTheme.glassBaseHeavy)
    }

    private func scopeBlock<Content: View>(
        title: String,
        height: CGFloat,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.geistMono(.medium, size: 7))
                .foregroundColor(OwLensTheme.textMuted)
                .tracking(0.3)

            content()
                .frame(height: height)
                .background(Color.black.opacity(0.55))
                .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 5, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.06), lineWidth: 0.5)
                )
        }
    }

    // Cinema RGB Histogram — original Red, Green, Blue channels with transparent layering
    private var histogramCanvas: some View {
        Canvas { context, size in
            drawHistogram(data.histogramRed, color: .red, context: &context, size: size)
            drawHistogram(data.histogramGreen, color: .green, context: &context, size: size)
            drawHistogram(data.histogramBlue, color: .blue, context: &context, size: size)
        }
    }

    private func drawHistogram(
        _ values: [Float],
        color: Color,
        context: inout GraphicsContext,
        size: CGSize
    ) {
        guard !values.isEmpty else { return }
        let step = size.width / CGFloat(values.count)
        for (index, value) in values.enumerated() {
            guard value > 0.01 else { continue }
            let height = max(1, CGFloat(value) * size.height)
            let rect = CGRect(
                x: CGFloat(index) * step,
                y: size.height - height,
                width: max(1, step - 0.5),
                height: height
            )
            context.fill(Path(rect), with: .color(color.opacity(0.45)))
        }
    }

    // Luma Waveform — original cinema oscilloscope green
    private var waveformCanvas: some View {
        Canvas { context, size in
            let columns = data.waveformColumns
            let rows = data.waveformRows
            guard columns > 0, rows > 0, data.waveform.count >= columns * rows else { return }

            let cellW = size.width / CGFloat(columns)
            let cellH = size.height / CGFloat(rows)
            for row in 0..<rows {
                for col in 0..<columns {
                    let value = data.waveform[row * columns + col]
                    guard value > 0.015 else { continue }
                    let alpha = min(0.85, 0.12 + Double(value) * 0.85)
                    let rect = CGRect(
                        x: CGFloat(col) * cellW,
                        y: CGFloat(row) * cellH,
                        width: max(1, cellW),
                        height: max(1, cellH)
                    )
                    context.fill(Path(rect), with: .color(Color.green.opacity(alpha)))
                }
            }

            // Reference graticules
            for guide in [0.25, 0.5, 0.75] {
                var path = Path()
                let y = size.height * guide
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: size.width, y: y))
                context.stroke(path, with: .color(Color.white.opacity(0.08)), lineWidth: 0.5)
            }
        }
    }
}
