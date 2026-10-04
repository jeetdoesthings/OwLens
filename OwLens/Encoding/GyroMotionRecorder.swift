import Foundation
import CoreMotion
import AVFoundation
import UIKit

/// Represents an individual IMU motion sample for embedded video telemetry and .gcsv export.
public struct MotionSample: Sendable {
    /// Timestamp relative to recording start, in milliseconds.
    public let timeMs: Double
    /// Gyroscope rotation rate (rad/s) aligned to camera optical frame.
    public let gx: Double
    public let gy: Double
    public let gz: Double
    /// Total acceleration (g) aligned to camera optical frame.
    public let ax: Double
    public let ay: Double
    public let az: Double

    /// Creates a 16-byte standard Google CAMM Type 2 (gyroscope) packet:
    /// - uint16 reserved = 0
    /// - uint16 typ = 2 (gyroscope)
    /// - float32 gx (rad/s, little-endian)
    /// - float32 gy (rad/s, little-endian)
    /// - float32 gz (rad/s, little-endian)
    public func toCAMMData() -> Data {
        var data = Data(count: 16)
        data.withUnsafeMutableBytes { rawPtr in
            rawPtr.storeBytes(of: UInt16(0).littleEndian, toByteOffset: 0, as: UInt16.self)
            rawPtr.storeBytes(of: UInt16(2).littleEndian, toByteOffset: 2, as: UInt16.self)
            rawPtr.storeBytes(of: Float(gx).bitPattern.littleEndian, toByteOffset: 4, as: UInt32.self)
            rawPtr.storeBytes(of: Float(gy).bitPattern.littleEndian, toByteOffset: 8, as: UInt32.self)
            rawPtr.storeBytes(of: Float(gz).bitPattern.littleEndian, toByteOffset: 12, as: UInt32.self)
        }
        return data
    }
}

/// High-precision IMU motion recorder for embedded Gyroflow video stabilization.
/// Samples CMMotionManager at 100Hz on an isolated background queue and feeds
/// synchronized samples directly into the AVAssetWriter timed metadata track.
public final class GyroMotionRecorder: @unchecked Sendable {
    private let motionManager = CMMotionManager()
    private let motionQueue: OperationQueue
    private let lock = NSLock()

    private var samples: [MotionSample] = []
    private var isRecording = false
    private var startHostTime: TimeInterval = 0
    private var orientation: UIInterfaceOrientation = .landscapeRight
    private var targetFPS: Double = 24.0

    public init() {
        let queue = OperationQueue()
        queue.name = "com.owlens.motionRecorder"
        queue.qualityOfService = .userInitiated
        queue.maxConcurrentOperationCount = 1
        self.motionQueue = queue
    }

    /// Begin sampling device motion at 100 Hz for the embedded metadata track.
    public func start(orientation: UIInterfaceOrientation, fps: Double) {
        lock.lock()
        defer { lock.unlock() }

        self.orientation = orientation
        self.targetFPS = fps
        self.samples.removeAll()
        self.samples.reserveCapacity(15000) // ~2.5 minutes at 100Hz
        self.startHostTime = CACurrentMediaTime()
        self.isRecording = true

        guard motionManager.isDeviceMotionAvailable else {
            print("[GyroMotionRecorder] Device motion unavailable on this hardware")
            return
        }

        motionManager.deviceMotionUpdateInterval = 1.0 / 100.0
        motionManager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: motionQueue) { [weak self] motion, _ in
            guard let self, let motion else { return }
            self.recordSample(motion: motion)
        }
        print("[GyroMotionRecorder] Started 100Hz IMU sampling for embedded metadata")
    }

    /// Anchor session start time to optical capture timeline for frame sync.
    public func anchorSession(startHostTime: TimeInterval) {
        lock.lock()
        defer { lock.unlock() }
        self.startHostTime = startHostTime
    }

    private func recordSample(motion: CMDeviceMotion) {
        lock.lock()
        defer { lock.unlock() }
        guard isRecording else { return }

        let now = CACurrentMediaTime()
        let elapsedMs = max(0, (now - startHostTime) * 1000.0)

        // Raw gyro rotation rate in rad/s from CMMotionManager
        let rx = motion.rotationRate.x
        let ry = motion.rotationRate.y
        let rz = motion.rotationRate.z

        // Raw total acceleration in g
        let rawAx = motion.userAcceleration.x + motion.gravity.x
        let rawAy = motion.userAcceleration.y + motion.gravity.y
        let rawAz = motion.userAcceleration.z + motion.gravity.z

        // Map device IMU axes to camera video frame axes based on recording orientation
        let gx: Double
        let gy: Double
        let gz: Double
        let ax: Double
        let ay: Double
        let az: Double

        switch orientation {
        case .landscapeRight:
            // Standard cinema grip: charging port on right, Dynamic Island on left
            gx = -ry
            gy = rx
            gz = -rz
            ax = -rawAy
            ay = rawAx
            az = -rawAz
        case .landscapeLeft:
            // Inverted landscape: charging port on left, Dynamic Island on right
            gx = ry
            gy = -rx
            gz = -rz
            ax = rawAy
            ay = -rawAx
            az = -rawAz
        case .portrait:
            gx = rx
            gy = ry
            gz = -rz
            ax = rawAx
            ay = rawAy
            az = -rawAz
        case .portraitUpsideDown:
            gx = -rx
            gy = -ry
            gz = -rz
            ax = -rawAx
            ay = -rawAy
            az = -rawAz
        case .unknown:
            gx = -ry
            gy = rx
            gz = -rz
            ax = -rawAy
            ay = rawAx
            az = -rawAz
        @unknown default:
            gx = -ry
            gy = rx
            gz = -rz
            ax = -rawAy
            ay = rawAx
            az = -rawAz
        }

        let sample = MotionSample(
            timeMs: elapsedMs,
            gx: gx,
            gy: gy,
            gz: gz,
            ax: ax,
            ay: ay,
            az: az
        )
        samples.append(sample)
    }

    /// Retrieve the closest motion sample for the current video frame.
    public func latestSample(at elapsedSeconds: Double) -> MotionSample? {
        lock.lock()
        defer { lock.unlock() }
        guard !samples.isEmpty else { return nil }

        let targetMs = elapsedSeconds * 1000.0
        if let last = samples.last, last.timeMs <= targetMs {
            return last
        }

        for sample in samples.reversed() {
            if sample.timeMs <= targetMs {
                return sample
            }
        }
        return samples.first
    }

    /// Export all recorded motion samples in official Gyroflow .gcsv format.
    public func exportGCSVData(videoFileName: String? = nil) -> Data {
        lock.lock()
        let recordedSamples = samples
        lock.unlock()

        var csv = "GYROFLOW IMU LOG\n"
        csv += "version,1.3\n"
        csv += "id,Apple_iPhone\n"
        csv += "vendor,Apple\n"
        if let videoFileName {
            csv += "videofilename,\(videoFileName)\n"
        }
        csv += "tscale,0.001\n"
        csv += "gscale,1.0\n"
        csv += "ascale,1.0\n"
        csv += "orientation,XYZ\n"
        csv += "t,gx,gy,gz,ax,ay,az\n"

        for s in recordedSamples {
            let tStr = String(format: "%.3f", s.timeMs)
            let gxStr = String(format: "%.6f", s.gx)
            let gyStr = String(format: "%.6f", s.gy)
            let gzStr = String(format: "%.6f", s.gz)
            let axStr = String(format: "%.6f", s.ax)
            let ayStr = String(format: "%.6f", s.ay)
            let azStr = String(format: "%.6f", s.az)
            csv += "\(tStr),\(gxStr),\(gyStr),\(gzStr),\(axStr),\(ayStr),\(azStr)\n"
        }
        return Data(csv.utf8)
    }

    /// Saves the .gcsv sidecar file to the Files app.
    /// 1. Always saves to Documents/OwLens Gyro/<baseName>.gcsv (visible in Files app -> "On My iPhone -> OwLens -> OwLens Gyro").
    /// 2. If additionalFolderURL is provided (e.g. user selected custom Files destination folder), also saves alongside the video file.
    @discardableResult
    public static func saveGCSVFile(data: Data, videoFileName: String, additionalFolderURL: URL? = nil) -> URL? {
        guard let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return nil
        }

        let gyroFolder = documentsURL.appendingPathComponent("OwLens Gyro", isDirectory: true)
        try? FileManager.default.createDirectory(at: gyroFolder, withIntermediateDirectories: true, attributes: nil)

        let gcsvName = (videoFileName as NSString).deletingPathExtension + ".gcsv"
        let primaryTargetURL = gyroFolder.appendingPathComponent(gcsvName)

        do {
            try data.write(to: primaryTargetURL, options: .atomic)
            print("[GyroMotionRecorder] Saved .gcsv to Files app: \(primaryTargetURL.path)")
        } catch {
            print("[GyroMotionRecorder] Failed to save .gcsv to primary folder: \(error)")
        }

        if let additionalFolderURL {
            let shouldStop = additionalFolderURL.startAccessingSecurityScopedResource()
            defer {
                if shouldStop {
                    additionalFolderURL.stopAccessingSecurityScopedResource()
                }
            }
            let sidecarURL = additionalFolderURL.appendingPathComponent(gcsvName)
            do {
                try data.write(to: sidecarURL, options: .atomic)
                print("[GyroMotionRecorder] Also saved .gcsv alongside video: \(sidecarURL.path)")
            } catch {
                print("[GyroMotionRecorder] Failed to save secondary .gcsv: \(error)")
            }
        }

        return primaryTargetURL
    }

    /// Stop motion updates and release memory buffer.
    public func stop() {
        lock.lock()
        defer { lock.unlock() }
        isRecording = false
        motionManager.stopDeviceMotionUpdates()
        samples.removeAll()
        print("[GyroMotionRecorder] Stopped motion updates.")
    }
}
