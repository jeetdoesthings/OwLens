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
        case .uhd4k: return .mbps150
        }
    }
}

enum BitratePreset: Int, CaseIterable, Identifiable {
    case mbps50 = 50
    case mbps80 = 80
    case mbps100 = 100
    case mbps150 = 150
    case mbps200 = 200

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
