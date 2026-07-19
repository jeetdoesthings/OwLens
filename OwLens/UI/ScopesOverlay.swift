import SwiftUI

struct ScopesOverlay: View {
    let data: ScopeData

    var body: some View {
        VStack(spacing: 8) {
            scopeBlock(title: "RGB", height: 34) {
                histogramCanvas
            }
            scopeBlock(title: "WFM", height: 46) {
                waveformCanvas
            }
        }
        .frame(width: 136)
        .padding(6)
        .background(Color.black.opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
        )
    }

    private func scopeBlock<Content: View>(
        title: String,
        height: CGFloat,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 7, weight: .bold, design: .monospaced))
                .foregroundColor(.white.opacity(0.5))
            content()
                .frame(height: height)
                .background(Color.black.opacity(0.3))
                .clipShape(RoundedRectangle(cornerRadius: 3, style: .continuous))
        }
    }

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
            context.fill(Path(rect), with: .color(color.opacity(0.42)))
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
                    let alpha = min(0.82, 0.1 + Double(value) * 0.85)
                    let rect = CGRect(
                        x: CGFloat(col) * cellW,
                        y: CGFloat(row) * cellH,
                        width: max(1, cellW),
                        height: max(1, cellH)
                    )
                    context.fill(Path(rect), with: .color(.green.opacity(alpha)))
                }
            }

            for guide in [0.25, 0.5, 0.75] {
                var path = Path()
                let y = size.height * guide
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: size.width, y: y))
                context.stroke(path, with: .color(.white.opacity(0.08)), lineWidth: 1)
            }
        }
    }
}
