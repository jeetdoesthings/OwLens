import SwiftUI

/// Rule-of-thirds grid + spirit-level horizon, framed to video aspect (not full screen).
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

    /// Same aspect-fit math as CameraPreviewView Metal letterbox.
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
        return Path { path in
            path.move(to: CGPoint(x: w / 3, y: 0))
            path.addLine(to: CGPoint(x: w / 3, y: h))
            path.move(to: CGPoint(x: 2 * w / 3, y: 0))
            path.addLine(to: CGPoint(x: 2 * w / 3, y: h))
            path.move(to: CGPoint(x: 0, y: h / 3))
            path.addLine(to: CGPoint(x: w, y: h / 3))
            path.move(to: CGPoint(x: 0, y: 2 * h / 3))
            path.addLine(to: CGPoint(x: w, y: 2 * h / 3))
        }
        .stroke(Color.white.opacity(0.4), lineWidth: 0.8)
    }

    private func level(in frame: CGRect) -> some View {
        // Invert: physical tilt left → line tilts opposite (spirit-level feel)
        let tilt = -levelMonitor.tiltDegrees
        let isLevel = levelMonitor.isLevel
        let clamped = max(-45, min(45, tilt))
        return ZStack {
            Rectangle()
                .fill(isLevel ? Color.green.opacity(0.85) : Color.yellow.opacity(0.75))
                .frame(width: min(frame.width * 0.55, 280), height: 2)
                .rotationEffect(.degrees(clamped))

            Circle()
                .strokeBorder(isLevel ? Color.green : Color.white.opacity(0.7), lineWidth: 1.5)
                .frame(width: 10, height: 10)

            Text(String(format: "%+.1f°", tilt))
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundColor(isLevel ? .green : .white.opacity(0.85))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color.black.opacity(0.45))
                .clipShape(Capsule())
                .offset(y: 28)
        }
    }
}
