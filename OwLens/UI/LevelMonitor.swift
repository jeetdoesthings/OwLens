import Foundation
import CoreMotion
import Combine
import UIKit

/// Device tilt for on-screen spirit level (landscape cinema framing).
@MainActor
final class LevelMonitor: ObservableObject {
    /// Horizon tilt in degrees. 0 = level. Positive = clockwise.
    @Published private(set) var tiltDegrees: Double = 0
    /// True when within ~1.0° of level.
    @Published private(set) var isLevel: Bool = true

    private let motion = CMMotionManager()
    private var isRunning = false

    func start() {
        guard !isRunning, motion.isDeviceMotionAvailable else { return }
        isRunning = true
        motion.deviceMotionUpdateInterval = 1.0 / 12.0
        motion.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] data, _ in
            guard let self, let g = data?.gravity else { return }
            let interfaceOrientation = UIApplication.shared.connectedScenes
                .compactMap { $0 as? UIWindowScene }
                .first?.interfaceOrientation ?? .landscapeRight

            let angle: Double
            if interfaceOrientation == .landscapeLeft {
                angle = atan2(-g.y, -g.x) * 180.0 / .pi
            } else {
                angle = atan2(g.y, g.x) * 180.0 / .pi
            }

            // Normalize to −90…90 for display
            var tilt = angle
            if tilt > 90 { tilt -= 180 }
            if tilt < -90 { tilt += 180 }
            let newIsLevel = abs(tilt) < 1.0

            if newIsLevel && !self.isLevel {
                Haptics.selection()
            }

            // Deadband threshold prevents sub-pixel noise from spamming SwiftUI renders
            if abs(tilt - self.tiltDegrees) >= 0.15 || newIsLevel != self.isLevel {
                self.tiltDegrees = tilt
                self.isLevel = newIsLevel
            }
        }
    }

    func stop() {
        guard isRunning else { return }
        isRunning = false
        motion.stopDeviceMotionUpdates()
    }
}
