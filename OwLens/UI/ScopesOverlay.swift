import SwiftUI

struct ScopesOverlay: View {
    let data: ScopeData

    var body: some View {
        VStack(spacing: 5) {
            scopeBlock(title: "RGB HISTOGRAM", height: 30) {
                histogramCanvas
            }
            scopeBlock(title: "LUMA WAVEFORM", height: 40) {
                waveformCanvas
            }
        }
        .frame(width: 120)
        .padding(5)
        .glassPanel(cornerRadius: OwLensTheme.radiusSm, border: OwLensTheme.glassBorderActive, background: OwLensTheme.glassBaseHeavy)
    }

    private func scopeBlock<Content: View>(
        title: String,
        height: CGFloat,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.geistMono(.medium, size: 7))
                .foregroundColor(OwLensTheme.textMuted)
                .tracking(0.5)

            content()
                .frame(height: height)
                .background(Color.black.opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.06), lineWidth: 0.5)
                )
        }
    }

    // Cinema RGB Histogram — distinct red, green, and blue channels with additive color blending
    private var histogramCanvas: some View {
        Canvas { context, size in
            context.blendMode = .plusLighter
            drawHistogram(data.histogramRed, color: Color(red: 1.0, green: 0.25, blue: 0.25, opacity: 0.65), context: &context, size: size)
            drawHistogram(data.histogramGreen, color: Color(red: 0.25, green: 0.88, blue: 0.35, opacity: 0.60), context: &context, size: size)
            drawHistogram(data.histogramBlue, color: Color(red: 0.20, green: 0.55, blue: 1.0, opacity: 0.65), context: &context, size: size)
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
            context.fill(Path(rect), with: .color(color))
        }
    }

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
                    let alpha = min(0.80, 0.10 + Double(value) * 0.85)
                    let rect = CGRect(
                        x: CGFloat(col) * cellW,
                        y: CGFloat(row) * cellH,
                        width: max(1, cellW),
                        height: max(1, cellH)
                    )
                    context.fill(Path(rect), with: .color(Color.white.opacity(alpha)))
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
