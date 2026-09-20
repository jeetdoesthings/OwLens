import AVFoundation
import CoreGraphics
import Foundation

/// Capture / encode frame rate — **24 and 30 only**.
/// Output is constant-frame-rate; missing RAW stills are held so file is true 24/30 fps.
enum CaptureFrameRate: Double, CaseIterable, Identifiable {
    case fps24 = 24
    case fps30 = 30

    var id: Double { rawValue }

    var label: String {
        switch self {
        case .fps24: return "24"
        case .fps30: return "30"
        }
    }

    var displayName: String { "\(label) fps" }
}

/// Output framing + encode resolution.
enum RecordingFormat: String, CaseIterable, Identifiable {
    case openGate = "openGate"
    case hd169 = "hd169"
    case uhd4k = "uhd4k"

    var id: String { rawValue }

    var width: Int {
        switch self {
        case .openGate: return 1920
        case .hd169: return 1920
        case .uhd4k: return 3840
        }
    }

    var height: Int {
        switch self {
        case .openGate: return 1440
        case .hd169: return 1080
        case .uhd4k: return 2160
        }
    }

    var aspectRatio: CGFloat {
        CGFloat(width) / CGFloat(height)
    }

    var shortLabel: String {
        switch self {
        case .openGate: return "OG"
        case .hd169: return "1080"
        case .uhd4k: return "4K"
        }
    }

    var displayName: String {
        switch self {
        case .openGate: return "Open Gate 4:3"
        case .hd169: return "1080p 16:9"
        case .uhd4k: return "4K 16:9"
        }
    }

    var detailLabel: String { "\(width)×\(height)" }

    var suggestedBitratePreset: BitratePreset {
        switch self {
        case .openGate: return .mbps100
        case .hd169: return .mbps80
        case .uhd4k: return .mbps100
        }
    }

    var maxBitratePreset: BitratePreset {
        // 200Mbps exceeds HEVC realtime encoder limits on iPhone — cap all at 150Mbps.
        .mbps150
    }
}

enum BitratePreset: Int, CaseIterable, Identifiable {
    case mbps50 = 50
    case mbps80 = 80
    case mbps100 = 100
    case mbps150 = 150

    var id: Int { rawValue }
    var label: String { "\(rawValue)" }
    var displayName: String { "\(rawValue) Mbps" }
    var bitsPerSecond: Int { rawValue * 1_000_000 }
}

enum VideoSaveDestination: String, CaseIterable, Identifiable {
    case photos = "photos"
    case files = "files"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .photos: return "Photos"
        case .files: return "Files"
        }
    }
}

/// Mic option from AVAudioSession ports (Off / iPhone / external by name).
struct AudioSourceOption: Identifiable, Hashable {
    let id: String
    let name: String
    /// nil = Off (no audio track). Otherwise AVAudioSessionPortDescription.uid
    let portUID: String?

    static let none = AudioSourceOption(id: "none", name: "Off", portUID: nil)
}

/// Back-camera lens discovered on this device (dynamic per iPhone model).
struct LensOption: Identifiable, Hashable {
    let id: String
    let name: String
    let shortLabel: String
    let deviceType: AVCaptureDevice.DeviceType
    let uniqueID: String
}

enum PreviewDisplayMode: Int, CaseIterable, Identifiable {
    case log = 0
    case normalVideo = 1

    var id: Int { rawValue }

    var label: String {
        switch self {
        case .log: return "LOG"
        case .normalVideo: return "VID"
        }
    }
}

// MARK: - Storage Estimator

/// Utilities for querying free device storage and estimating remaining recording time.
enum StorageEstimator {
    /// Safety reserve (500 MB) left untouched so AVAssetWriter can finalize moov atoms and PhotoKit can save.
    static let safetyReserveBytes: Int64 = 500 * 1024 * 1024

    /// Queries the currently available disk capacity suitable for important user recordings.
    static func availableDiskSpaceBytes() -> Int64 {
        do {
            let values = try URL(fileURLWithPath: NSHomeDirectory()).resourceValues(forKeys: [
                .volumeAvailableCapacityForImportantUsageKey,
                .volumeAvailableCapacityKey
            ])
            if let important = values.volumeAvailableCapacityForImportantUsage, important > 0 {
                return important
            }
            if let normal = values.volumeAvailableCapacity, normal > 0 {
                return Int64(normal)
            }
        } catch {}
        if let attrs = try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory()),
           let freeSize = attrs[.systemFreeSize] as? NSNumber {
            return freeSize.int64Value
        }
        return 0
    }

    /// Estimates bytes written per second for the given recording configuration.
    static func estimatedBytesPerSecond(
        format: RecordingFormat,
        fps: CaptureFrameRate,
        codec: VideoCodecOption,
        bitratePreset: BitratePreset,
        includeAudio: Bool
    ) -> Double {
        let audioBps: Double = includeAudio ? 128_000.0 : 0.0
        let videoBps: Double
        switch codec {
        case .hevc:
            let effective = min(bitratePreset.bitsPerSecond, format.maxBitratePreset.bitsPerSecond)
            videoBps = Double(effective)
        case .proRes422:
            let baseRate: Double
            switch format {
            case .uhd4k: baseRate = 500_000_000.0
            case .openGate: baseRate = 156_000_000.0
            case .hd169: baseRate = 117_000_000.0
            }
            videoBps = baseRate * (fps.rawValue / 24.0)
        case .proRes422HQ:
            let baseRate: Double
            switch format {
            case .uhd4k: baseRate = 750_000_000.0
            case .openGate: baseRate = 235_000_000.0
            case .hd169: baseRate = 176_000_000.0
            }
            videoBps = baseRate * (fps.rawValue / 24.0)
        }
        return (videoBps + audioBps) / 8.0
    }

    /// Computes usable recording time in seconds after reserving safety margin.
    static func estimatedRemainingSeconds(
        availableBytes: Int64,
        bytesPerSecond: Double
    ) -> Int {
        let usable = max(0, availableBytes - safetyReserveBytes)
        guard bytesPerSecond > 0 else { return 0 }
        return Int(Double(usable) / bytesPerSecond)
    }

    /// Formats remaining seconds into a concise display string (e.g. "45:12" or "1h 20m").
    static func formatRemainingTime(seconds: Int) -> String {
        if seconds <= 0 { return "00:00" }
        let hours = seconds / 3600
        let mins = (seconds % 3600) / 60
        let secs = seconds % 60
        if hours > 0 {
            return String(format: "%dh %02dm", hours, mins)
        } else {
            return String(format: "%02d:%02d", mins, secs)
        }
    }
}
