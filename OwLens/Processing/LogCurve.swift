import simd

/// Log curve types available for encoding.
/// Start with sLog3 — they're well-documented and good
/// enough for most grading workflows. Exact proprietary curves (Apple Log,
/// ARRI LogC3, etc.) require licensed specs or reverse engineering.
enum LogCurveType: Int, CaseIterable, Identifiable {
    case linear = 0
    case sLog3Approx = 2

    var id: Int { rawValue }

    var displayName: String {
        switch self {
        case .linear: return "Linear"
        case .sLog3Approx: return "S-Log3"
        }
    }

    /// The curves exposed in the UI (hide linear unless debugging).
    static var uiCases: [LogCurveType] { [.sLog3Approx] }
}

/// CPU-side log curve math — used for LUT generation and validation.
/// The actual per-frame encoding runs on GPU via the Metal shader in Debayer.metal.
enum LogCurve {
    /// Sony S-Log3 published transfer function (industry-documented formula).
    static func sLog3Approx(_ linear: Float) -> Float {
        if linear >= 0.01125000 {
            return (420.0 + log10((linear + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0
        } else {
            return (linear * (171.2102946929 - 95.0) / 0.01125000 + 95.0) / 1023.0
        }
    }

    /// Inverse S-Log3 — used for LUT generation (Section 8).
    static func inverseSLog3Approx(_ encoded: Float) -> Float {
        if encoded >= (171.2102946929) / 1023.0 {
            let a = pow(10.0, (encoded * 1023.0 - 420.0) / 261.5)
            return a * (0.18 + 0.01) - 0.01
        } else {
            return (encoded * 1023.0 - 95.0) * 0.01125000 / (171.2102946929 - 95.0)
        }
    }

    static func apply(_ rgb: SIMD3<Float>, type: LogCurveType) -> SIMD3<Float> {
        switch type {
        case .linear:
            return simd_clamp(rgb, SIMD3(0,0,0), SIMD3(1,1,1))
        case .sLog3Approx:
            return SIMD3(sLog3Approx(rgb.x), sLog3Approx(rgb.y), sLog3Approx(rgb.z))
        }
    }

    static func inverse(_ rgb: SIMD3<Float>, type: LogCurveType) -> SIMD3<Float> {
        switch type {
        case .linear:
            return simd_clamp(rgb, SIMD3(0,0,0), SIMD3(1,1,1))
        case .sLog3Approx:
            return SIMD3(inverseSLog3Approx(rgb.x), inverseSLog3Approx(rgb.y), inverseSLog3Approx(rgb.z))
        }
    }
}
