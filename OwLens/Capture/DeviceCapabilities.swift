import AVFoundation
import Foundation
import UIKit

#if canImport(Darwin)
import Darwin
#endif

// MARK: - Chip tier & Bayer

/// Coarse SoC generation for default fps / resolution guidance.
enum ChipTier: String, CaseIterable {
    case a12a13 = "A12/A13"
    case a14 = "A14"
    case a15 = "A15"
    case a16Plus = "A16+"
    case unknown = "Unknown"
}

/// CFA pattern ids matching Debayer.metal / CaptureController.
enum BayerPatternID: Int32, CaseIterable {
    case rggb = 0
    case grbg = 1
    case gbrg = 2
    case bggr = 3

    var label: String {
        switch self {
        case .rggb: return "RGGB"
        case .grbg: return "GRBG"
        case .gbrg: return "GBRG"
        case .bggr: return "BGGR"
        }
    }
}

/// Output codec choice
enum VideoCodecOption: String, CaseIterable, Identifiable {
    case hevc = "hevc"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .hevc: return "HEVC"
        }
    }
}

// MARK: - DeviceCapabilities

/// One-shot capability probe at launch (before camera session when possible).
/// Additive — does not alter the proven iPhone 12 Pro pipeline when that device is detected.
struct DeviceCapabilities: Sendable {
    /// e.g. `iPhone13,3`
    let machineIdentifier: String
    let marketingName: String
    let chipTier: ChipTier

    /// Bayer RAW stills available (photo output empty list ⇒ unsupported for log).
    let supportsBayerRAW: Bool
    let bayerRawFormats: [OSType]

    /// Explicit allow-list of models we have verified end-to-end.
    let isVerifiedDevice: Bool

    /// Recommended defaults for this tier (A14/12 Pro match existing app defaults).
    let recommendedFPS: CaptureFrameRate
    let recommendedFormat: RecordingFormat
    let recommendedBitrate: BitratePreset

    /// Per-model CFA override if known; nil ⇒ use live OSType / DNG metadata.
    let bayerPatternOverride: BayerPatternID?

    /// Multi-line log blob for tester reports.
    let diagnosticSummary: String

    // MARK: - Probe

    /// Run once at launch / before session. Lightweight; may touch photo output briefly for RAW check.
    static func probe() -> DeviceCapabilities {
        let machine = hardwareMachineIdentifier()
        let tier = chipTier(for: machine)
        let marketing = marketingName(for: machine)
        let verified = verifiedDeviceIDs.contains(machine)

        let (supportsRAW, bayerFormats) = probeBayerRAWSupport()
        let cfaOverride = bayerOverrideTable[machine]

        let recFPS: CaptureFrameRate
        let recFormat: RecordingFormat
        let recBitrate: BitratePreset
        switch tier {
        case .a12a13:
            recFPS = .fps24
            recFormat = .openGate
            recBitrate = .mbps80
        case .a14:
            // Match existing 12 Pro defaults — do not change proven path
            recFPS = .fps24
            recFormat = .openGate
            recBitrate = .mbps100
        case .a15:
            recFPS = .fps24
            recFormat = .openGate
            recBitrate = .mbps100
        case .a16Plus:
            recFPS = .fps30
            recFormat = .hd169
            recBitrate = .mbps150
        case .unknown:
            recFPS = .fps24
            recFormat = .openGate
            recBitrate = .mbps80
        }

        let formatsDesc = bayerFormats.map { fourCCString($0) }.joined(separator: ", ")
        let summary = """
        [DeviceCapabilities]
        machine=\(machine)
        marketing=\(marketing)
        chip=\(tier.rawValue)
        verified=\(verified)
        bayerRAW=\(supportsRAW)
        bayerFormats=[\(formatsDesc)]
        cfaOverride=\(cfaOverride?.label ?? "nil(use live)")
        recommended=\(recFormat.shortLabel) @ \(recFPS.label)fps \(recBitrate.label)Mbps
        """

        print(summary)

        return DeviceCapabilities(
            machineIdentifier: machine,
            marketingName: marketing,
            chipTier: tier,
            supportsBayerRAW: supportsRAW,
            bayerRawFormats: bayerFormats,
            isVerifiedDevice: verified,
            recommendedFPS: recFPS,
            recommendedFormat: recFormat,
            recommendedBitrate: recBitrate,
            bayerPatternOverride: cfaOverride,
            diagnosticSummary: summary
        )
    }

    // MARK: - Allow list (verified)

    /// Models we have actually tested end-to-end. Expand as testers report.
    static let verifiedDeviceIDs: Set<String> = [
        "iPhone13,3", // iPhone 12 Pro
        "iPhone13,4", // iPhone 12 Pro Max (same generation; treat as verified tier)
    ]

    // MARK: - CFA override table (update as devices are tested)

    /// Prefer live OSType/DNG when nil. Values match Debayer.metal pattern ids.
    static let bayerOverrideTable: [String: BayerPatternID] = [
        // iPhone 12 Pro / Pro Max — confirmed RGGB via 14-bit Bayer FourCC
        "iPhone13,3": .rggb,
        "iPhone13,4": .rggb,
        // Add more after tester confirmation, e.g.:
        // "iPhone14,2": .rggb, // 13 Pro
    ]

    /// Default when machine unknown and OSType unmapped.
    static let defaultBayerPattern: BayerPatternID = .rggb

    // MARK: - Hardware id

    static func hardwareMachineIdentifier() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let mirror = Mirror(reflecting: systemInfo.machine)
        let id = mirror.children.reduce(into: "") { result, element in
            guard let value = element.value as? Int8, value != 0 else { return }
            result.append(Character(UnicodeScalar(UInt8(value))))
        }
        return id.isEmpty ? "unknown" : id
    }

    // MARK: - Chip mapping (identifier → tier)

    /// Map product type to chip tier. Incomplete list is OK — unknown falls back safely.
    static func chipTier(for machine: String) -> ChipTier {
        // Simulator
        if machine == "x86_64" || machine == "arm64" || machine.hasPrefix("i386") {
            // Could be sim; treat unknown
            if machine == "arm64" || machine == "x86_64" {
                // On device arm64 is real; utsname.machine is iPhone*, not arm64, on device
            }
        }
        if machine.hasPrefix("i386") || machine == "x86_64" {
            return .unknown
        }
        // On Apple Silicon Mac sim sometimes reports arm64
        if machine == "arm64" {
            return .unknown
        }

        switch machine {
        // A12 / A13 — XR, XS, 11, SE (2nd)
        case "iPhone11,2", "iPhone11,4", "iPhone11,6", "iPhone11,8",
             "iPhone12,1", "iPhone12,3", "iPhone12,5", "iPhone12,8":
            return .a12a13
        // A14 — 12 series
        case "iPhone13,1", "iPhone13,2", "iPhone13,3", "iPhone13,4":
            return .a14
        // A15 — 13 series, SE 3, 14 / 14 Plus
        case "iPhone14,2", "iPhone14,3", "iPhone14,4", "iPhone14,5",
             "iPhone14,6", "iPhone14,7", "iPhone14,8":
            return .a15
        // A16+ — 14 Pro, 15, 16…
        case "iPhone15,2", "iPhone15,3",
             "iPhone15,4", "iPhone15,5",
             "iPhone16,1", "iPhone16,2",
             "iPhone17,1", "iPhone17,2", "iPhone17,3", "iPhone17,4":
            return .a16Plus
        default:
            if let n = majorPhoneFamily(machine) {
                if n >= 15 { return .a16Plus }
                if n >= 14 { return .a15 }
                if n >= 13 { return .a14 }
                if n >= 11 { return .a12a13 }
            }
            return .unknown
        }
    }

    private static func majorPhoneFamily(_ machine: String) -> Int? {
        // iPhone13,3 → 13
        guard machine.hasPrefix("iPhone") else { return nil }
        let rest = machine.dropFirst("iPhone".count)
        let num = rest.prefix(while: { $0.isNumber })
        return Int(num)
    }

    static func marketingName(for machine: String) -> String {
        let map: [String: String] = [
            "iPhone13,1": "iPhone 12 mini",
            "iPhone13,2": "iPhone 12",
            "iPhone13,3": "iPhone 12 Pro",
            "iPhone13,4": "iPhone 12 Pro Max",
            "iPhone14,2": "iPhone 13 Pro",
            "iPhone14,3": "iPhone 13 Pro Max",
            "iPhone14,4": "iPhone 13 mini",
            "iPhone14,5": "iPhone 13",
            "iPhone14,7": "iPhone 14",
            "iPhone14,8": "iPhone 14 Plus",
            "iPhone15,2": "iPhone 14 Pro",
            "iPhone15,3": "iPhone 14 Pro Max",
            "iPhone15,4": "iPhone 15",
            "iPhone15,5": "iPhone 15 Plus",
            "iPhone16,1": "iPhone 15 Pro",
            "iPhone16,2": "iPhone 15 Pro Max",
        ]
        return map[machine] ?? machine
    }

    // MARK: - RAW probe

    /// Lightweight session to query Bayer RAW without starting capture.
    private static func probeBayerRAWSupport() -> (Bool, [OSType]) {
        let session = AVCaptureSession()
        session.beginConfiguration()
        session.sessionPreset = .photo
        // Single-lens wide only — never dual/triple virtual multi-cam for RAW probe
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              !camera.isVirtualDevice,
              camera.deviceType == .builtInWideAngleCamera,
              let input = try? AVCaptureDeviceInput(device: camera),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return (false, [])
        }
        print("[DeviceCapabilities] RAW probe camera type=\(camera.deviceType.rawValue) virtual=\(camera.isVirtualDevice)")
        session.addInput(input)

        let photoOutput = AVCapturePhotoOutput()
        guard session.canAddOutput(photoOutput) else {
            session.commitConfiguration()
            return (false, [])
        }
        session.addOutput(photoOutput)
        session.commitConfiguration()

        // After commit, list is populated (empty mid-configuration)
        let all = photoOutput.availableRawPhotoPixelFormatTypes
        let bayer = all.filter { AVCapturePhotoOutput.isBayerRAWPixelFormat($0) }
        print("[DeviceCapabilities] RAW probe all=\(all.map { fourCCString($0) }) bayer=\(bayer.map { fourCCString($0) })")
        return (!bayer.isEmpty, bayer)
    }

    // MARK: - CFA resolution helper

    /// Resolve CFA: per-frame OSType → override table → default RGGB.
    static func resolveBayerPattern(
        fromPixelFormat format: OSType?,
        fromMetadataPattern metadataPattern: Int32?,
        capabilities: DeviceCapabilities
    ) -> Int32 {
        if let override = capabilities.bayerPatternOverride {
            return override.rawValue
        }
        if let meta = metadataPattern {
            return meta
        }
        if let format, let fromFourCC = CaptureController.cfaPatternOptional(forBayerFormat: format) {
            return fromFourCC
        }
        return defaultBayerPattern.rawValue
    }
}

// MARK: - FourCC helper (local)

private func fourCCString(_ type: OSType) -> String {
    let chars: [UInt8] = [
        UInt8((type >> 24) & 0xFF),
        UInt8((type >> 16) & 0xFF),
        UInt8((type >> 8) & 0xFF),
        UInt8(type & 0xFF)
    ]
    let s = String(bytes: chars, encoding: .ascii) ?? String(type)
    return "'\(s)'"
}
