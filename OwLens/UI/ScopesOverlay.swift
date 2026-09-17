import SwiftUI

struct ScopesOverlay: View {
    let data: ScopeData

    var body: some View {
        VStack(spacing: 5) {
            // RGB Histogram Block
            scopeBlock(
                title: "HISTOGRAM",
                headerAccessory: AnyView(histogramBadge)
            ) {
                histogramCanvas
                    .frame(height: 30)
            }

            // Luma Waveform Block with IRE Scale
            scopeBlock(
                title: "WAVEFORM",
                headerAccessory: AnyView(waveformBadge)
            ) {
                waveformCanvas
                    .frame(height: 46)
            }
        }
        .frame(width: 108)
        .padding(.horizontal, 6)
        .padding(.vertical, 6)
        .glassPanel(
            cornerRadius: OwLensTheme.radiusCard,
            border: OwLensTheme.glassBorderActive,
            background: OwLensTheme.glassBaseHeavy
        )
    }

    // MARK: - Header Badges

    private var histogramBadge: some View {
        HStack(spacing: 3) {
            Circle().fill(Color(red: 1.0, green: 0.25, blue: 0.25)).frame(width: 3.5, height: 3.5)
            Circle().fill(Color(red: 0.20, green: 0.95, blue: 0.40)).frame(width: 3.5, height: 3.5)
            Circle().fill(Color(red: 0.30, green: 0.65, blue: 1.0)).frame(width: 3.5, height: 3.5)
        }
    }

    private var waveformBadge: some View {
        Text("IRE")
            .font(.geistMono(.bold, size: 7))
            .foregroundColor(Color(red: 1.0, green: 0.80, blue: 0.28).opacity(0.90))
            .padding(.horizontal, 3.5)
            .padding(.vertical, 1)
            .background(
                RoundedRectangle(cornerRadius: 3, style: .continuous)
                    .fill(Color(red: 1.0, green: 0.80, blue: 0.28).opacity(0.18))
            )
    }

    // MARK: - Scope Block Container

    private func scopeBlock<Content: View>(
        title: String,
        headerAccessory: AnyView? = nil,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 4) {
                Text(title)
                    .font(.geistMono(.semiBold, size: 8))
                    .foregroundColor(OwLensTheme.textSecondary)
                    .tracking(0.3)

                Spacer(minLength: 0)

                if let accessory = headerAccessory {
                    accessory
                }
            }

            content()
                .background(Color.black.opacity(0.60))
                .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 5, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.08), lineWidth: 0.5)
                )
        }
    }

    // MARK: - Cinema RGB Histogram

    private var histogramCanvas: some View {
        Canvas { context, size in
            // Subtle vertical graticule guides at 18% (shadow/mid boundary), 50% (midtones), and 90%
            let guides: [(CGFloat, Double)] = [(0.18, 0.06), (0.50, 0.08), (0.90, 0.06)]
            for (pos, opacity) in guides {
                var guidePath = Path()
                let x = size.width * pos
                guidePath.move(to: CGPoint(x: x, y: 0))
                guidePath.addLine(to: CGPoint(x: x, y: size.height))
                context.stroke(
                    guidePath,
                    with: .color(Color.white.opacity(opacity)),
                    style: StrokeStyle(lineWidth: 0.5, dash: [1.5, 1.5])
                )
            }

            // Draw RGB Channels with smooth filled area and top outline
            drawHistogramChannel(data.histogramRed, color: Color(red: 1.0, green: 0.22, blue: 0.22), context: &context, size: size)
            drawHistogramChannel(data.histogramGreen, color: Color(red: 0.18, green: 0.95, blue: 0.40), context: &context, size: size)
            drawHistogramChannel(data.histogramBlue, color: Color(red: 0.25, green: 0.60, blue: 1.0), context: &context, size: size)
        }
    }

    private func drawHistogramChannel(
        _ values: [Float],
        color: Color,
        context: inout GraphicsContext,
        size: CGSize
    ) {
        guard !values.isEmpty else { return }
        let count = values.count
        let step = size.width / CGFloat(count)

        // Filled area path
        var fillPath = Path()
        fillPath.move(to: CGPoint(x: 0, y: size.height))

        // Outline path
        var linePath = Path()

        for (i, v) in values.enumerated() {
            let h = CGFloat(v) * (size.height - 2)
            let x = CGFloat(i) * step + step * 0.5
            let y = max(1.0, size.height - h)
            if i == 0 {
                linePath.move(to: CGPoint(x: x, y: y))
            } else {
                linePath.addLine(to: CGPoint(x: x, y: y))
            }
            fillPath.addLine(to: CGPoint(x: x, y: y))
        }

        fillPath.addLine(to: CGPoint(x: size.width, y: size.height))
        fillPath.closeSubpath()

        context.fill(fillPath, with: .color(color.opacity(0.35)))
        context.stroke(linePath, with: .color(color.opacity(0.85)), lineWidth: 0.75)
    }

    // MARK: - Luma Waveform with Dedicated IRE Scale & Graticules

    private var waveformCanvas: some View {
        Canvas { context, size in
            let columns = data.waveformColumns
            let rows = data.waveformRows
            let scaleWidth: CGFloat = 18.0
            let traceWidth: CGFloat = max(10, size.width - scaleWidth)

            guard columns > 0, rows > 0, data.waveform.count >= columns * rows else { return }

            let cellW = traceWidth / CGFloat(columns)
            let cellH = size.height / CGFloat(rows)

            // ── 1. Oscilloscope Signal Trace ──
            for row in 0..<rows {
                for col in 0..<columns {
                    let value = data.waveform[row * columns + col]
                    guard value > 0.015 else { continue }
                    let v = Double(value)
                    let rect = CGRect(
                        x: CGFloat(col) * cellW,
                        y: CGFloat(row) * cellH,
                        width: max(1.0, cellW),
                        height: max(1.0, cellH)
                    )

                    if v > 0.45 {
                        // Dense phosphor core: mint-white / bright neon
                        let coreAlpha = min(0.95, 0.40 + v * 0.55)
                        let coreColor = Color(red: 0.45, green: 1.0, blue: 0.65).opacity(coreAlpha)
                        context.fill(Path(rect), with: .color(coreColor))
                    } else {
                        // Oscilloscope trace emerald green
                        let traceAlpha = min(0.70, 0.15 + v * 0.90)
                        let traceColor = Color(red: 0.0, green: 0.88, blue: 0.38).opacity(traceAlpha)
                        context.fill(Path(rect), with: .color(traceColor))
                    }
                }
            }

            // ── 2. Horizontal IRE Graticule Lines (across trace area) ──

            // 100 IRE (Top clipping line)
            var path100 = Path()
            path100.move(to: CGPoint(x: 0, y: 0.5))
            path100.addLine(to: CGPoint(x: traceWidth, y: 0.5))
            context.stroke(path100, with: .color(Color.white.opacity(0.18)), lineWidth: 0.75)

            // 75 IRE (Highlight ceiling warning)
            var path75 = Path()
            let y75 = size.height * 0.25
            path75.move(to: CGPoint(x: 0, y: y75))
            path75.addLine(to: CGPoint(x: traceWidth, y: y75))
            context.stroke(path75, with: .color(Color.white.opacity(0.14)), style: StrokeStyle(lineWidth: 0.5, dash: [2, 2]))

            // 50 IRE (Middle Gray reference — accented in Cinema Gold!)
            var path50 = Path()
            let y50 = size.height * 0.50
            path50.move(to: CGPoint(x: 0, y: y50))
            path50.addLine(to: CGPoint(x: traceWidth, y: y50))
            context.stroke(
                path50,
                with: .color(Color(red: 1.0, green: 0.78, blue: 0.22).opacity(0.50)),
                style: StrokeStyle(lineWidth: 0.75, dash: [3, 2])
            )

            // 25 IRE (Shadow region guide)
            var path25 = Path()
            let y25 = size.height * 0.75
            path25.move(to: CGPoint(x: 0, y: y25))
            path25.addLine(to: CGPoint(x: traceWidth, y: y25))
            context.stroke(path25, with: .color(Color.white.opacity(0.14)), style: StrokeStyle(lineWidth: 0.5, dash: [2, 2]))

            // 0 IRE (Black floor baseline)
            var path0 = Path()
            let y0 = size.height - 0.5
            path0.move(to: CGPoint(x: 0, y: y0))
            path0.addLine(to: CGPoint(x: traceWidth, y: y0))
            context.stroke(path0, with: .color(Color.white.opacity(0.18)), lineWidth: 0.75)

            // ── 3. Vertical Divider Line ──
            var divider = Path()
            divider.move(to: CGPoint(x: traceWidth, y: 0))
            divider.addLine(to: CGPoint(x: traceWidth, y: size.height))
            context.stroke(divider, with: .color(Color.white.opacity(0.10)), lineWidth: 0.5)

            // ── 4. Dedicated IRE Scale Marks & Labels ──
            let scaleX = size.width - 1.0
            let tickX = traceWidth + 2.0

            func drawLabel(_ text: String, y: CGFloat, color: Color, isBold: Bool = false) {
                var tick = Path()
                tick.move(to: CGPoint(x: traceWidth, y: y))
                tick.addLine(to: CGPoint(x: tickX, y: y))
                context.stroke(tick, with: .color(color.opacity(0.5)), lineWidth: 0.5)

                let font: Font = .system(size: 6.5, weight: isBold ? .bold : .medium, design: .monospaced)
                let resolved = Text(text).font(font).foregroundColor(color)
                context.draw(resolved, at: CGPoint(x: scaleX, y: y), anchor: .trailing)
            }

            drawLabel("100", y: 3.5, color: Color.white.opacity(0.50))
            drawLabel("75", y: y75, color: Color.white.opacity(0.40))
            drawLabel("50", y: y50, color: Color(red: 1.0, green: 0.80, blue: 0.28).opacity(0.95), isBold: true)
            drawLabel("25", y: y25, color: Color.white.opacity(0.40))
            drawLabel("0", y: size.height - 3.5, color: Color.white.opacity(0.50))
        }
    }
}
