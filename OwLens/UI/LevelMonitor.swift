import Foundation
import CoreMotion
import Combine

/// Device tilt for on-screen spirit level (landscape cinema framing).
@MainActor
final class LevelMonitor: ObservableObject {
    /// Horizon tilt in degrees. 0 = level. Positive = clockwise.
    @Published private(set) var tiltDegrees: Double = 0
    /// True when within ~1.5° of level.
    @Published private(set) var isLevel: Bool = true

    private let motion = CMMotionManager()
    private var isRunning = false

    func start() {
        guard !isRunning, motion.isDeviceMotionAvailable else { return }
        isRunning = true
        motion.deviceMotionUpdateInterval = 1.0 / 30.0
        motion.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] data, _ in
            guard let self, let g = data?.gravity else { return }
            // Landscape-friendly horizon angle from gravity vector
            let angle = atan2(g.y, g.x) * 180.0 / .pi
            // Normalize to −90…90 for display
            var tilt = angle
            if tilt > 90 { tilt -= 180 }
            if tilt < -90 { tilt += 180 }
            self.tiltDegrees = tilt
            self.isLevel = abs(tilt) < 1.5
        }
    }

    func stop() {
        guard isRunning else { return }
        isRunning = false
        motion.stopDeviceMotionUpdates()
    }
}
