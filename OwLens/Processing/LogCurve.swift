import simd

/// Log curve types available for encoding.
/// Start with sLog3 — they're well-documented and good
/// enough for most grading workflows. Exact proprietary curves (Apple Log,
/// ARRI LogC3, etc.) require licensed specs or reverse engineering.
enum LogCurveType: Int, CaseIterable, Identifiable {
    case linear = 0
    case sLog3Approx = 2
    case appleLog2 = 3

    var id: Int { rawValue }

    var displayName: String {
        switch self {
        case .linear: return "Linear"
        case .sLog3Approx: return "S-Log3"
        case .appleLog2: return "Apple Log 2"
        }
    }

    var fileLabel: String {
        switch self {
        case .linear: return "Linear"
        case .sLog3Approx: return "SLog3"
        case .appleLog2: return "AppleLog2"
        }
    }

    /// The curves exposed in the UI (hide linear unless debugging).
    static var uiCases: [LogCurveType] { [.appleLog2, .sLog3Approx] }
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

    /// Apple Log 2 published transfer function (Apple Log Profile White Paper).
    /// Takes scene-linear reflectance R directly (0.18 = 18% gray reference).
    static func appleLog2Encode(_ linear: Float) -> Float {
        let r0: Float = -0.05641088
        let rt: Float = 0.01
        let c: Float = 47.28711236
        let beta: Float = 0.00964052
        let gamma: Float = 0.08550479
        let delta: Float = 0.69336945

        if linear < r0 {
            return 0.0
        } else if linear < rt {
            let diff = linear - r0
            return c * diff * diff
        } else {
            return gamma * log2(linear + beta) + delta
        }
    }

    /// Inverse Apple Log 2 — decodes encoded pixel value P back to scene reflectance R.
    static func appleLog2Decode(_ encoded: Float) -> Float {
        let r0: Float = -0.05641088
        let rt: Float = 0.01
        let c: Float = 47.28711236
        let beta: Float = 0.00964052
        let gamma: Float = 0.08550479
        let delta: Float = 0.69336945
        let pt: Float = c * (rt - r0) * (rt - r0)

        if encoded < 0.0 {
            return r0
        } else if encoded < pt {
            return sqrt(encoded / c) + r0
        } else {
            return pow(2.0, (encoded - delta) / gamma) - beta
        }
    }

    /// Target maximum scene reflectance Rmax mapped to sensor clipping (1.0).
    /// - Apple Log 2: R = 12.0 maps to exactly 1.0 (code 1023 / 100% IRE).
    /// - Sony S-Log3: R = 10.0 represents 1000% reflectance (standard Sony ceiling ~85% IRE).
    /// - Linear: R = 1.0.
    static func defaultRMax(for type: LogCurveType) -> Float {
        switch type {
        case .linear: return 1.0
        case .appleLog2: return 12.0
        case .sLog3Approx: return 10.0
        }
    }

    /// Smooth filmic highlight shoulder mapping sensor linear [0, 1] to scene reflectance [0, rMax].
    /// Perfectly preserves 100% linear calibration for midtones & shadows (r <= rKnee),
    /// while smoothly rolling off highlights up to the container ceiling with C1 continuity.
    static func applyHighlightShoulder(_ r: Float, rKnee: Float = 0.36, rMax: Float = 12.0) -> Float {
        guard rMax > rKnee + 1e-4 else { return r }
        if r <= rKnee {
            return r
        }
        let delta = rMax - rKnee
        let dr = 1.0 - rKnee
        let s0 = dr / delta
        let s1: Float = 0.0
        let a = s1 + s0 - 2.0
        let b = 3.0 - 2.0 * s0 - s1
        let c = s0

        let t = simd_clamp((r - rKnee) / max(dr, 1e-4), 0.0, 1.0)
        let g = ((a * t + b) * t + c) * t
        return rKnee + delta * g
    }

    static func applyHighlightShoulder3(_ rgb: SIMD3<Float>, rKnee: Float = 0.36, rMax: Float = 12.0) -> SIMD3<Float> {
        let peak = max(rgb.x, max(rgb.y, rgb.z))
        guard rMax > rKnee + 1e-4, peak > rKnee else { return rgb }
        let peakShoulder = applyHighlightShoulder(peak, rKnee: rKnee, rMax: rMax)
        let scale = peakShoulder / max(peak, 1e-6)
        let scaled = rgb * scale

        // Smooth C¹ cubic highlight desaturation to pure neutral white as scene intensity
        // approaches sensor clipping / peak dynamic range (peak >= 0.85 -> 1.65).
        // Eliminates the Bayer clipping magenta/pink cast on the sky and clouds while
        // maintaining 100% color fidelity in midtones, skin tones, and rich sunsets.
        let u = simd_clamp((peak - 0.85) * 1.25, 0.0, 1.0)
        let desat = u * u * (3.0 - 2.0 * u)
        return simd_mix(scaled, SIMD3<Float>(repeating: peakShoulder), SIMD3<Float>(repeating: desat))
    }

    static func apply(_ rgb: SIMD3<Float>, type: LogCurveType, headroomScale: Float = 1.0) -> SIMD3<Float> {
        var input = rgb
        if headroomScale > 1.0 {
            input = applyHighlightShoulder3(input, rKnee: 0.36, rMax: headroomScale)
        }
        switch type {
        case .linear:
            return simd_clamp(input, SIMD3(0,0,0), SIMD3(1,1,1))
        case .sLog3Approx:
            return SIMD3(sLog3Approx(input.x), sLog3Approx(input.y), sLog3Approx(input.z))
        case .appleLog2:
            return SIMD3(appleLog2Encode(input.x), appleLog2Encode(input.y), appleLog2Encode(input.z))
        }
    }

    static func inverse(_ rgb: SIMD3<Float>, type: LogCurveType) -> SIMD3<Float> {
        switch type {
        case .linear:
            return simd_clamp(rgb, SIMD3(0,0,0), SIMD3(1,1,1))
        case .sLog3Approx:
            return SIMD3(inverseSLog3Approx(rgb.x), inverseSLog3Approx(rgb.y), inverseSLog3Approx(rgb.z))
        case .appleLog2:
            return SIMD3(appleLog2Decode(rgb.x), appleLog2Decode(rgb.y), appleLog2Decode(rgb.z))
        }
    }
}
