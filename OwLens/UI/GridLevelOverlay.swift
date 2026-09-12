import SwiftUI

/// Rule-of-thirds grid + precision spirit-level horizon, framed to video aspect ratio.
struct GridLevelOverlay: View {
    var showGrid: Bool
    var showLevel: Bool
    /// Active recording format aspect (width/height), e.g. 4/3 open gate, 16/9.
    var videoAspect: CGFloat
    @ObservedObject var levelMonitor: LevelMonitor

    var body: some View {
        GeometryReader { geo in
            let frame = videoFrame(in: geo.size, aspect: videoAspect)
            ZStack {
                if showGrid {
                    grid(in: frame)
                        .frame(width: frame.width, height: frame.height)
                        .position(x: frame.midX, y: frame.midY)
                }
                if showLevel {
                    level(in: frame)
                        .frame(width: frame.width, height: frame.height)
                        .position(x: frame.midX, y: frame.midY)
                }
            }
        }
        .allowsHitTesting(false)
    }

    /// Aspect-fit math matching viewfinder framing.
    private func videoFrame(in size: CGSize, aspect: CGFloat) -> CGRect {
        guard size.width > 0, size.height > 0, aspect > 0 else {
            return CGRect(origin: .zero, size: size)
        }
        let viewAspect = size.width / size.height
        if aspect > viewAspect {
            let h = size.width / aspect
            return CGRect(x: 0, y: (size.height - h) / 2, width: size.width, height: h)
        } else {
            let w = size.height * aspect
            return CGRect(x: (size.width - w) / 2, y: 0, width: w, height: size.height)
        }
    }

    private func grid(in frame: CGRect) -> some View {
        let w = frame.width
        let h = frame.height
        let crossSize: CGFloat = 8

        return ZStack {
            // Rule of thirds lines
            Path { path in
                // Vertical lines
                path.move(to: CGPoint(x: w / 3, y: 0))
                path.addLine(to: CGPoint(x: w / 3, y: h))
                path.move(to: CGPoint(x: 2 * w / 3, y: 0))
                path.addLine(to: CGPoint(x: 2 * w / 3, y: h))

                // Horizontal lines
                path.move(to: CGPoint(x: 0, y: h / 3))
                path.addLine(to: CGPoint(x: w, y: h / 3))
                path.move(to: CGPoint(x: 0, y: 2 * h / 3))
                path.addLine(to: CGPoint(x: w, y: 2 * h / 3))
            }
            .stroke(Color.white.opacity(0.25), lineWidth: 0.5)

            // Center target crosshair
            Path { path in
                path.move(to: CGPoint(x: w / 2 - crossSize, y: h / 2))
                path.addLine(to: CGPoint(x: w / 2 + crossSize, y: h / 2))
                path.move(to: CGPoint(x: w / 2, y: h / 2 - crossSize))
                path.addLine(to: CGPoint(x: w / 2, y: h / 2 + crossSize))
            }
            .stroke(Color.white.opacity(0.45), lineWidth: 0.75)
        }
    }

    private func level(in frame: CGRect) -> some View {
        let tilt = -levelMonitor.tiltDegrees
        let isLevel = levelMonitor.isLevel
        let clamped = max(-45, min(45, tilt))
        let levelColor = isLevel ? Color.white : Color.white.opacity(0.50)

        return ZStack {
            // Artificial horizon line
            Rectangle()
                .fill(levelColor.opacity(isLevel ? 0.9 : 0.5))
                .frame(width: min(frame.width * 0.45, 240), height: 1)
                .rotationEffect(.degrees(clamped))

            // Center spirit reticle
            Circle()
                .strokeBorder(isLevel ? Color.white : Color.white.opacity(0.5), lineWidth: 1)
                .frame(width: 10, height: 10)

            // Tilt degree readout (clean text only, no box wrapping)
            Text(String(format: "%+.1f°", tilt))
                .font(.geistMono(.medium, size: 10))
                .foregroundColor(levelColor)
                .shadow(color: Color.black.opacity(0.6), radius: 2, x: 0, y: 1)
                .offset(y: 20)
        }
    }
}
