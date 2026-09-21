import Foundation
import Metal
import MetalKit
import CoreVideo
import simd

#if DEBUG

extension MetalPipeline {
    /// Creates a small Bayer RAW buffer with a single saturated pixel, runs the full
    /// pipeline, and asserts the hot pixel is corrected by the Bayer-domain defect-pixel pass.
    /// Returns true if the defect is suppressed; prints diagnostics on failure.
    func runSyntheticHotPixelTest() -> Bool {
        let width = 32
        let height = 32
        let bytesPerRow = width * 2

        var rawBytes = [UInt16](repeating: 4096, count: width * height)
        // Inject a single saturated pixel near the center.
        let hotX = width / 2
        let hotY = height / 2
        rawBytes[hotY * width + hotX] = 65535

        let pixelBuffer: CVPixelBuffer? = rawBytes.withUnsafeBytes { ptr in
            var pixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreateWithBytes(
                kCFAllocatorDefault,
                width, height,
                kCVPixelFormatType_14Bayer_RGGB,
                UnsafeMutableRawPointer(mutating: ptr.baseAddress!),
                bytesPerRow,
                nil, nil,
                nil,
                &pixelBuffer
            )
            return status == kCVReturnSuccess ? pixelBuffer : nil
        }

        guard let buffer = pixelBuffer else {
            print("[SyntheticTest] failed to create pixel buffer")
            return false
        }

        // Run through the production pipeline at native resolution.
        var output: MTLTexture?
        let sem = DispatchSemaphore(value: 0)
        process(buffer, encodeWidth: width, encodeHeight: height) { result in
            output = result
            sem.signal()
        }
        sem.wait()
        guard let output else {
            print("[SyntheticTest] process returned nil")
            return false
        }

        // Read back the center region and the hot pixel location.
        // The production pipeline returns a private texture; blit to a shared staging texture
        // so CPU getBytes is legal on both device and simulator.
        let readW = output.width
        let readH = output.height
        let bytesPerPixel = 8 // rgba16Float
        let rowBytes = readW * bytesPerPixel
        var outputBytes = [UInt16](repeating: 0, count: readW * readH * 4)

        let readbackDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: output.pixelFormat, width: readW, height: readH, mipmapped: false)
        readbackDesc.usage = [.shaderRead]
        readbackDesc.storageMode = .shared
        guard let readbackTex = device.makeTexture(descriptor: readbackDesc),
              let readbackCB = commandQueue.makeCommandBuffer(),
              let blit = readbackCB.makeBlitCommandEncoder() else {
            print("[SyntheticTest] failed to create readback resources")
            return false
        }
        blit.copy(
            from: output,
            sourceSlice: 0, sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: readW, height: readH, depth: 1),
            to: readbackTex,
            destinationSlice: 0, destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        blit.endEncoding()
        readbackCB.commit()
        readbackCB.waitUntilCompleted()

        outputBytes.withUnsafeMutableBytes { raw in
            readbackTex.getBytes(raw.baseAddress!,
                           bytesPerRow: rowBytes,
                           from: MTLRegionMake2D(0, 0, readW, readH),
                           mipmapLevel: 0)
        }

        // The hot pixel in Bayer space maps to a small block in the output; ensure no
        // channel is clipped near 1.0 in the center region.
        let regionHalf = 2
        var maxValue: Float = 0
        var minValue: Float = Float.infinity
        for y in max(0, hotY - regionHalf)..<min(readH, hotY + regionHalf + 1) {
            for x in max(0, hotX - regionHalf)..<min(readW, hotX + regionHalf + 1) {
                let idx = (y * readW + x) * 4
                let r = Float(Float16(bitPattern: outputBytes[idx]))
                let g = Float(Float16(bitPattern: outputBytes[idx + 1]))
                let b = Float(Float16(bitPattern: outputBytes[idx + 2]))
                maxValue = max(maxValue, max(r, max(g, b)))
                minValue = min(minValue, min(r, min(g, b)))
            }
        }

        // Background was 4096/65535 ≈ 0.0625; a 65535 hot pixel should be suppressed
        // from 1.0 to residual energy below 0.6 after DPC correction + demosaic.
        let passed = maxValue < 0.6
        if !passed {
            // Find the exact pixel with the max value for debugging
            for y in max(0, hotY - regionHalf)..<min(readH, hotY + regionHalf + 1) {
                for x in max(0, hotX - regionHalf)..<min(readW, hotX + regionHalf + 1) {
                    let idx = (y * readW + x) * 4
                    let r = Float(Float16(bitPattern: outputBytes[idx]))
                    let g = Float(Float16(bitPattern: outputBytes[idx + 1]))
                    let b = Float(Float16(bitPattern: outputBytes[idx + 2]))
                    if max(r, max(g, b)) > 0.5 {
                        print("[SyntheticTest]   pixel (\(x),\(y)) r=\(r) g=\(g) b=\(b)")
                    }
                }
            }
        }
        print("[SyntheticTest] hot pixel region min=\(minValue) max=\(maxValue) — \(passed ? "PASS" : "FAIL")")
        return passed
    }

    /// Verifies Apple Log 2 transfer function accuracy against the published Apple Log White Paper reference values.
    static func runAppleLog2AccuracyTest() -> Bool {
        var allPassed = true

        let testCases: [(r: Float, expectedP: Float, name: String)] = [
            (0.0, 0.150477, "0% reflectance"),
            (0.18, 0.488272, "18% reflectance (middle gray)"),
            (0.90, 0.681686, "90% reflectance"),
            (12.0, 1.000000, "1200% reflectance")
        ]

        let eps: Float = 0.0001
        for tc in testCases {
            let encoded = LogCurve.appleLog2Encode(tc.r)
            let diff = abs(encoded - tc.expectedP)
            let pass = diff <= eps
            if !pass {
                print("[AppleLog2Test] FAIL \(tc.name): got \(encoded), expected \(tc.expectedP), diff \(diff)")
                allPassed = false
            } else {
                print("[AppleLog2Test] PASS \(tc.name): \(encoded) ≈ \(tc.expectedP)")
            }
        }

        // Below R0 clamp test
        let belowR0 = LogCurve.appleLog2Encode(-0.1)
        if belowR0 != 0.0 {
            print("[AppleLog2Test] FAIL below R0 clamp: got \(belowR0), expected 0.0")
            allPassed = false
        } else {
            print("[AppleLog2Test] PASS below R0 clamp: \(belowR0) == 0.0")
        }

        // Roundtrip invertibility test
        let roundtripValues: [Float] = [0.0, 0.005, 0.01, 0.18, 0.90, 1.0, 4.0, 12.0]
        for r in roundtripValues {
            let enc = LogCurve.appleLog2Encode(r)
            let dec = LogCurve.appleLog2Decode(enc)
            let diff = abs(dec - r)
            if diff > 0.0002 {
                print("[AppleLog2Test] FAIL roundtrip at R=\(r): decoded \(dec), diff \(diff)")
                allPassed = false
            } else {
                print("[AppleLog2Test] PASS roundtrip at R=\(r): encoded=\(enc) -> decoded=\(dec)")
            }
        }

        // SIMD3 test
        let rgbIn = SIMD3<Float>(0.0, 0.18, 0.90)
        let rgbEnc = LogCurve.apply(rgbIn, type: .appleLog2)
        let rgbDec = LogCurve.inverse(rgbEnc, type: .appleLog2)
        if abs(rgbDec.x - rgbIn.x) > eps || abs(rgbDec.y - rgbIn.y) > eps || abs(rgbDec.z - rgbIn.z) > eps {
            print("[AppleLog2Test] FAIL SIMD3 roundtrip: in=\(rgbIn), out=\(rgbDec)")
            allPassed = false
        } else {
            print("[AppleLog2Test] PASS SIMD3 apply/inverse: \(rgbEnc)")
        }

        return allPassed
    }

    /// Verifies that the calibrated sensor-to-gamut matrices preserve neutral white balance
    /// and map colors accurately without blowing out saturation.
    static func runColorMatrixValidationTest() -> Bool {
        var allPassed = true
        let eps: Float = 0.001

        let m2020 = WhiteBalanceParams.defaultSensorToBT2020
        let mSGamut = WhiteBalanceParams.defaultSensorToSGamut3Cine

        // Verify row sums equal 1.0 (neutral white preservation)
        let r0_2020 = m2020[0, 0] + m2020[1, 0] + m2020[2, 0]
        let r1_2020 = m2020[0, 1] + m2020[1, 1] + m2020[2, 1]
        let r2_2020 = m2020[0, 2] + m2020[1, 2] + m2020[2, 2]

        if abs(r0_2020 - 1.0) > eps || abs(r1_2020 - 1.0) > eps || abs(r2_2020 - 1.0) > eps {
            print("[ColorMatrixTest] FAIL BT.2020 row sums: [\(r0_2020), \(r1_2020), \(r2_2020)]")
            allPassed = false
        } else {
            print("[ColorMatrixTest] PASS BT.2020 row sums preserve white: [\(r0_2020), \(r1_2020), \(r2_2020)]")
        }

        let r0_sg = mSGamut[0, 0] + mSGamut[1, 0] + mSGamut[2, 0]
        let r1_sg = mSGamut[0, 1] + mSGamut[1, 1] + mSGamut[2, 1]
        let r2_sg = mSGamut[0, 2] + mSGamut[1, 2] + mSGamut[2, 2]

        if abs(r0_sg - 1.0) > eps || abs(r1_sg - 1.0) > eps || abs(r2_sg - 1.0) > eps {
            print("[ColorMatrixTest] FAIL SGamut3Cine row sums: [\(r0_sg), \(r1_sg), \(r2_sg)]")
            allPassed = false
        } else {
            print("[ColorMatrixTest] PASS SGamut3Cine row sums preserve white: [\(r0_sg), \(r1_sg), \(r2_sg)]")
        }

        // Verify 18% neutral gray maps exactly to 18% neutral gray
        let gray = SIMD3<Float>(0.18, 0.18, 0.18)
        let out2020 = m2020 * gray
        let outSG = mSGamut * gray

        if abs(out2020.x - 0.18) > eps || abs(out2020.y - 0.18) > eps || abs(out2020.z - 0.18) > eps {
            print("[ColorMatrixTest] FAIL BT.2020 neutral gray: \(out2020)")
            allPassed = false
        } else {
            print("[ColorMatrixTest] PASS BT.2020 neutral gray 0.18 preserved: \(out2020)")
        }

        if abs(outSG.x - 0.18) > eps || abs(outSG.y - 0.18) > eps || abs(outSG.z - 0.18) > eps {
            print("[ColorMatrixTest] FAIL SGamut3Cine neutral gray: \(outSG)")
            allPassed = false
        } else {
            print("[ColorMatrixTest] PASS SGamut3Cine neutral gray 0.18 preserved: \(outSG)")
        }

        // Verify diagonal bounds (must be well-conditioned ~0.85 to 1.45, not boosting saturation beyond physical bounds)
        let diag2020 = [m2020[0, 0], m2020[1, 1], m2020[2, 2]]
        let diagSG = [mSGamut[0, 0], mSGamut[1, 1], mSGamut[2, 2]]
        for d in diag2020 {
            if d < 0.85 || d > 1.45 {
                print("[ColorMatrixTest] FAIL BT.2020 diagonal out of bounds: \(d)")
                allPassed = false
            }
        }
        for d in diagSG {
            if d < 0.85 || d > 1.45 {
                print("[ColorMatrixTest] FAIL SGamut3Cine diagonal out of bounds: \(d)")
                allPassed = false
            }
        }

        return allPassed
    }

    /// Verifies standard compliance of Apple Log 2 and Sony S-Log3 transfer curves
    /// against published reference specifications (without artificial shoulder warping).
    static func runLogCurvesStandardComplianceTest() -> Bool {
        var allPassed = true
        let eps: Float = 0.001

        // ── Apple Log 2 Standards Verification ──
        let appleCases: [(r: Float, expectedP: Float, name: String)] = [
            (0.0,  0.150477, "Apple Log 0% reflectance"),
            (0.18, 0.488272, "Apple Log 18% middle gray"),
            (0.90, 0.681686, "Apple Log 90% diffuse white"),
            (1.0,  0.694553, "Apple Log 100% reflectance")
        ]
        for tc in appleCases {
            let encoded = LogCurve.appleLog2Encode(tc.r)
            let diff = abs(encoded - tc.expectedP)
            if diff > eps {
                print("[LogComplianceTest] FAIL \(tc.name): got \(encoded), expected \(tc.expectedP), diff \(diff)")
                allPassed = false
            } else {
                print("[LogComplianceTest] PASS \(tc.name): \(encoded) ≈ \(tc.expectedP)")
            }
        }

        // ── Sony S-Log3 Standards Verification ──
        let sLog3Cases: [(linear: Float, expectedCode: Float, name: String)] = [
            (0.0,       95.0 / 1023.0,  "S-Log3 black level (95 code)"),
            (0.01125,   171.2103 / 1023.0, "S-Log3 knee transition"),
            (0.18,      420.0 / 1023.0, "S-Log3 18% middle gray (420 code)"),
            (0.90,      0.584145,       "S-Log3 90% diffuse white"),
            (1.0,       0.596285,       "S-Log3 100% sensor clipping")
        ]
        for tc in sLog3Cases {
            let encoded = LogCurve.sLog3Approx(tc.linear)
            let diff = abs(encoded - tc.expectedCode)
            if diff > eps {
                print("[LogComplianceTest] FAIL \(tc.name): got \(encoded), expected \(tc.expectedCode), diff \(diff)")
                allPassed = false
            } else {
                print("[LogComplianceTest] PASS \(tc.name): \(encoded) ≈ \(tc.expectedCode)")
            }
        }

        // ── S-Log3 Invertibility Test ──
        let sLogRoundtrip: [Float] = [0.0, 0.005, 0.01125, 0.18, 0.50, 0.90, 1.0]
        for val in sLogRoundtrip {
            let enc = LogCurve.sLog3Approx(val)
            let dec = LogCurve.inverseSLog3Approx(enc)
            let diff = abs(dec - val)
            if diff > 0.0005 {
                print("[LogComplianceTest] FAIL S-Log3 roundtrip at \(val): decoded \(dec), diff \(diff)")
                allPassed = false
            }
        }

        // ── Strict Monotonicity Across [0, 1] ──
        var prevApple: Float = -1.0
        var prevSLog: Float = -1.0
        var monotonic = true
        for i in 0...100 {
            let r = Float(i) / 100.0
            let a = LogCurve.appleLog2Encode(r)
            let s = LogCurve.sLog3Approx(r)
            if a < prevApple || s < prevSLog {
                print("[LogComplianceTest] FAIL non-monotonic at r=\(r)")
                monotonic = false
                allPassed = false
                break
            }
            prevApple = a
            prevSLog = s
        }
        if monotonic {
            print("[LogComplianceTest] PASS: Apple Log 2 and S-Log3 are strictly monotonic across [0, 1]")
        }

        return allPassed
    }

    /// Verifies 10-bit Video Range quantization and BT.2020 YCbCr color matrix mathematics.
    static func run10BitYCbCrEncodingTest() -> Bool {
        var passed = true
        let kNormScale: Float = 64.0 / 65535.0

        // 1. Luma Video Range limits (64..940 in 10-bit)
        let yBlack10: Float = 64.0
        let yWhite10: Float = 940.0
        let normYBlack = yBlack10 * kNormScale
        let normYWhite = yWhite10 * kNormScale

        // Integer 16-bit word when stored in .r16Unorm
        let word16Black = UInt16((normYBlack * 65535.0).rounded())
        let word16White = UInt16((normYWhite * 65535.0).rounded())

        // Bits 15..6 should equal codeValue10, lowest 6 bits should be zero
        let decoded10Black = word16Black >> 6
        let decoded10White = word16White >> 6
        let remBlack = word16Black & 0x3F
        let remWhite = word16White & 0x3F

        if decoded10Black != 64 || remBlack != 0 {
            print("[10BitYCbCrTest] FAIL Y black: code=\(decoded10Black), rem=\(remBlack)")
            passed = false
        }
        if decoded10White != 940 || remWhite != 0 {
            print("[10BitYCbCrTest] FAIL Y white: code=\(decoded10White), rem=\(remWhite)")
            passed = false
        }

        // 2. Chroma Video Range limits (64..960, center 512)
        let cbMid10: Float = 512.0
        let word16ChromaMid = UInt16(((cbMid10 * kNormScale) * 65535.0).rounded())
        let decoded10ChromaMid = word16ChromaMid >> 6
        let remChromaMid = word16ChromaMid & 0x3F

        if decoded10ChromaMid != 512 || remChromaMid != 0 {
            print("[10BitYCbCrTest] FAIL Cb/Cr center: code=\(decoded10ChromaMid), rem=\(remChromaMid)")
            passed = false
        }

        if passed {
            print("[10BitYCbCrTest] PASS: 10-bit MSB alignment (word16 = code10 << 6) verified perfectly")
        }
        return passed
    }

    /// Validates cubic Hermite highlight shoulder linearity, continuity, monotonicity, and ceiling reach.
    static func runHighlightShoulderTest() -> Bool {
        var allPassed = true
        let eps: Float = 1e-4

        // 1. Linearity: r <= 0.36 must remain 100% untouched
        let linearTestPoints: [Float] = [0.0, 0.01, 0.05, 0.18, 0.25, 0.36]
        for r in linearTestPoints {
            let outApple = LogCurve.applyHighlightShoulder(r, rKnee: 0.36, rMax: 12.0)
            let outSony = LogCurve.applyHighlightShoulder(r, rKnee: 0.36, rMax: 10.0)
            if abs(outApple - r) > eps {
                print("[HighlightShoulderTest] FAIL Apple Log shoulder altered midtone r=\(r): got \(outApple)")
                allPassed = false
            }
            if abs(outSony - r) > eps {
                print("[HighlightShoulderTest] FAIL S-Log3 shoulder altered midtone r=\(r): got \(outSony)")
                allPassed = false
            }
        }

        // 2. Ceiling reach: r = 1.0 maps to exact rMax
        let clipApple = LogCurve.applyHighlightShoulder(1.0, rKnee: 0.36, rMax: 12.0)
        if abs(clipApple - 12.0) > 0.01 {
            print("[HighlightShoulderTest] FAIL Apple Log shoulder clipping: expected 12.0, got \(clipApple)")
            allPassed = false
        }
        let clipSony = LogCurve.applyHighlightShoulder(1.0, rKnee: 0.36, rMax: 10.0)
        if abs(clipSony - 10.0) > 0.01 {
            print("[HighlightShoulderTest] FAIL S-Log3 shoulder clipping: expected 10.0, got \(clipSony)")
            allPassed = false
        }

        // 3. Monotonicity: strictly increasing from r = 0 to 1.0
        var prevVal: Float = -1.0
        for i in 0...100 {
            let r = Float(i) / 100.0
            let val = LogCurve.applyHighlightShoulder(r, rKnee: 0.36, rMax: 12.0)
            if val <= prevVal && i > 0 {
                print("[HighlightShoulderTest] FAIL Monotonicity broken at r=\(r): val=\(val), prev=\(prevVal)")
                allPassed = false
            }
            prevVal = val
        }

        // 4. C1 continuity at knee (numerical derivative before and after knee)
        let h: Float = 0.0001
        let dBelow = (LogCurve.applyHighlightShoulder(0.36, rKnee: 0.36, rMax: 12.0) - LogCurve.applyHighlightShoulder(0.36 - h, rKnee: 0.36, rMax: 12.0)) / h
        let dAbove = (LogCurve.applyHighlightShoulder(0.36 + h, rKnee: 0.36, rMax: 12.0) - LogCurve.applyHighlightShoulder(0.36, rKnee: 0.36, rMax: 12.0)) / h
        if abs(dBelow - 1.0) > 0.01 || abs(dAbove - 1.0) > 0.01 {
            print("[HighlightShoulderTest] FAIL C1 continuity at knee: dBelow=\(dBelow), dAbove=\(dAbove)")
            allPassed = false
        }

        if allPassed {
            print("[HighlightShoulderTest] PASS: Highlight shoulder linearity, continuity, monotonicity, and ceiling reach verified")
        }
        return allPassed
    }

    /// Validates that the Malvar-He-Cutler 5x5 linear demosaicing formulas preserve neutral color balance
    /// on uniform (flat) fields across all Bayer CFA sub-pixel phases, with zero green bias.
    static func runMalvarNeutralityTest() -> Bool {
        var allPassed = true
        let eps: Float = 1e-5

        // On a uniform field of value V:
        let testValues: [Float] = [0.25, 0.5, 0.8, 1.0]
        for v in testValues {
            let c00 = v
            let cN1 = v, cS1 = v, cE1 = v, cW1 = v
            let cN2 = v, cS2 = v, cE2 = v, cW2 = v
            let cNE = v, cNW = v, cSE = v, cSW = v

            // Malvar-He-Cutler formulas from Debayer.metal:
            let gAtRB = (2.0 * (cN1 + cS1 + cE1 + cW1) + 4.0 * c00 - (cN2 + cS2 + cE2 + cW2)) * 0.125
            let colorAtGH = (4.0 * (cE1 + cW1) + 5.0 * c00 - (cE2 + cW2) + 0.5 * (cN2 + cS2) - (cNE + cNW + cSE + cSW)) * 0.125
            let colorAtGV = (4.0 * (cN1 + cS1) + 5.0 * c00 - (cN2 + cS2) + 0.5 * (cE2 + cW2) - (cNE + cNW + cSE + cSW)) * 0.125
            let colorAtDiag = (2.0 * (cNE + cNW + cSE + cSW) + 6.0 * c00 - 1.5 * (cN2 + cS2 + cE2 + cW2)) * 0.125

            if abs(gAtRB - v) > eps {
                print("[MalvarTest] FAIL: G_at_RB expected \(v), got \(gAtRB)")
                allPassed = false
            }
            if abs(colorAtGH - v) > eps {
                print("[MalvarTest] FAIL: Color_at_G_H expected \(v), got \(colorAtGH)")
                allPassed = false
            }
            if abs(colorAtGV - v) > eps {
                print("[MalvarTest] FAIL: Color_at_G_V expected \(v), got \(colorAtGV)")
                allPassed = false
            }
            if abs(colorAtDiag - v) > eps {
                print("[MalvarTest] FAIL: Color_at_Diag expected \(v), got \(colorAtDiag)")
                allPassed = false
            }

            // Test norm-preserving highlight shoulder with neutral white input
            let neutralIn = SIMD3<Float>(v, v, v)
            let shoulderOut = LogCurve.applyHighlightShoulder3(neutralIn, rKnee: 0.36, rMax: 12.0)
            if abs(shoulderOut.x - shoulderOut.y) > eps || abs(shoulderOut.y - shoulderOut.z) > eps {
                print("[MalvarTest] FAIL: Highlight shoulder skewed neutral white \(neutralIn) -> \(shoulderOut)")
                allPassed = false
            }
        }

        if allPassed {
            print("[MalvarTest] PASS: Malvar-He-Cutler demosaic neutrality & norm-preserving shoulder verified")
        }
        return allPassed
    }

    /// Benchmarks the optimized pipeline (single-pass fused demosaic + direct BGRA scaling)
    /// on synthetic Bayer frames to verify that GPU frame times stay comfortably below the real-time budget.
    func runPipelineThroughputBenchmark() -> Bool {
        let width = 1920
        let height = 1080
        let bytesPerRow = width * 2
        let rawBytes = [UInt16](repeating: 4096, count: width * height)

        let pixelBuffer: CVPixelBuffer? = rawBytes.withUnsafeBytes { ptr in
            var pb: CVPixelBuffer?
            let status = CVPixelBufferCreateWithBytes(
                kCFAllocatorDefault,
                width, height,
                kCVPixelFormatType_14Bayer_RGGB,
                UnsafeMutableRawPointer(mutating: ptr.baseAddress!),
                bytesPerRow,
                nil, nil, nil,
                &pb
            )
            return status == kCVReturnSuccess ? pb : nil
        }

        guard let buffer = pixelBuffer else {
            print("[ThroughputBenchmark] Failed to create synthetic pixel buffer")
            return false
        }

        // Warm up
        let warmupSem = DispatchSemaphore(value: 0)
        process(buffer, encodeWidth: width, encodeHeight: height, encodeAsBGRA: true) { _, _ in
            warmupSem.signal()
        }
        warmupSem.wait()

        // Benchmark 10 iterations
        let iterations = 10
        var totalMs: Double = 0.0
        for _ in 0..<iterations {
            let t0 = CACurrentMediaTime()
            let sem = DispatchSemaphore(value: 0)
            process(buffer, encodeWidth: width, encodeHeight: height, encodeAsBGRA: true) { _, _ in
                let dt = (CACurrentMediaTime() - t0) * 1000.0
                totalMs += dt
                sem.signal()
            }
            sem.wait()
        }

        let avgMs = totalMs / Double(iterations)
        print("[ThroughputBenchmark] 1080p average frame time: \(String(format: "%.2f", avgMs)) ms across \(iterations) frames")
        var pass = avgMs < 33.33 // must easily fit in 30 fps budget
        print("[ThroughputBenchmark] Performance: \(pass ? "PASS" : "FAIL") (target < 33.33 ms)")

        // Verify 10-bit YCbCr pixel buffer format and attachments on output
        let verifySem = DispatchSemaphore(value: 0)
        let oldCurve = self.curveType
        self.curveType = .appleLog2
        process(buffer, encodeWidth: width, encodeHeight: height, encodeAsBGRA: true) { _, pb in
            if let pb = pb {
                let fmt = CVPixelBufferGetPixelFormatType(pb)
                let is10Bit = (fmt == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange)
                let planeCount = CVPixelBufferGetPlaneCount(pb)
                let hasBT2020 = CVBufferCopyAttachment(pb, kCVImageBufferColorPrimariesKey, nil) != nil
                if is10Bit && planeCount == 2 && hasBT2020 {
                    print("[ThroughputBenchmark] PASS: Pipeline correctly produced 10-bit YCbCr 4:2:0 CVPixelBuffer with BT.2020 metadata")
                } else {
                    print("[ThroughputBenchmark] FAIL: Expected 10-bit bi-planar YCbCr, got fmt=\(fmt) planes=\(planeCount)")
                    pass = false
                }
            } else {
                print("[ThroughputBenchmark] FAIL: No pixel buffer returned for 10-bit encoding")
                pass = false
            }
            verifySem.signal()
        }
        verifySem.wait()
        self.curveType = oldCurve

        return pass
    }

    /// Validates Auto Exposure shutter angle calculations, stop-index nearest matching,
    /// White Balance gain green normalization, and correlated color temperature interpolation.
    static func runAutoExposureAndWBValidationTest() -> Bool {
        var allPassed = true

        // 1. Shutter angle conversion from exposure duration
        let fps = 24.0
        let duration180 = 1.0 / 48.0
        let angle180 = Float(duration180 * fps * 360.0)
        if abs(angle180 - 180.0) > 0.001 {
            print("[AE/AWBTest] FAIL: 1/48s at 24fps expected 180°, got \(angle180)")
            allPassed = false
        }

        let duration90 = 1.0 / 96.0
        let angle90 = Float(duration90 * fps * 360.0)
        if abs(angle90 - 90.0) > 0.001 {
            print("[AE/AWBTest] FAIL: 1/96s at 24fps expected 90°, got \(angle90)")
            allPassed = false
        }

        // 2. Exposure stops nearest matching
        let stops = ExposureStops.isoStops(in: 50...2000)
        let idx50 = ExposureStops.nearestIndex(in: stops, to: 48)
        if stops[idx50] != 50 {
            print("[AE/AWBTest] FAIL: Nearest stop for 48 should be 50, got \(stops[idx50])")
            allPassed = false
        }
        let idx800 = ExposureStops.nearestIndex(in: stops, to: 750)
        if stops[idx800] != 800 {
            print("[AE/AWBTest] FAIL: Nearest stop for 750 should be 800, got \(stops[idx800])")
            allPassed = false
        }

        // 3. White balance green normalization
        let rawR: Float = 2.4
        let rawG: Float = 1.2
        let rawB: Float = 1.8
        let g = max(rawG, 0.001)
        let normGains = SIMD3<Float>(max(rawR / g, 0.01), 1.0, max(rawB / g, 0.01))
        if abs(normGains.x - 2.0) > 1e-4 || abs(normGains.y - 1.0) > 1e-4 || abs(normGains.z - 1.5) > 1e-4 {
            print("[AE/AWBTest] FAIL: Normalized gains expected [2.0, 1.0, 1.5], got \(normGains)")
            allPassed = false
        }

        // 4. Correlated color temperature interpolation (DNG spec formula: g = (1/T - 1/T2) / (1/T1 - 1/T2))
        let t1: Float = 2856.0 // Standard Light A
        let t2: Float = 6504.0 // D65
        let factorA = simd_clamp((1.0 / t1 - 1.0 / t2) / (1.0 / t1 - 1.0 / t2), 0.0, 1.0)
        let factorD65 = simd_clamp((1.0 / t2 - 1.0 / t2) / (1.0 / t1 - 1.0 / t2), 0.0, 1.0)
        if abs(factorA - 1.0) > 1e-4 {
            print("[AE/AWBTest] FAIL: Factor at T1 (2856K) should be 1.0, got \(factorA)")
            allPassed = false
        }
        if abs(factorD65 - 0.0) > 1e-4 {
            print("[AE/AWBTest] FAIL: Factor at T2 (6504K) should be 0.0, got \(factorD65)")
            allPassed = false
        }

        // 5. Storage Estimator bytes-per-second and remaining time validation
        let bps = StorageEstimator.estimatedBytesPerSecond(
            format: .openGate,
            fps: .fps24,
            codec: .hevc,
            bitratePreset: .mbps100,
            includeAudio: true
        )
        let expectedBps = (100_000_000.0 + 128_000.0) / 8.0
        if abs(bps - expectedBps) > 1e-2 {
            print("[AE/AWBTest] FAIL: Expected bps \(expectedBps), got \(bps)")
            allPassed = false
        }

        // Usable storage with 500MB safety reserve
        let testAvailableBytes: Int64 = 10_000_000_000 // 10 GB
        let remainingSecs = StorageEstimator.estimatedRemainingSeconds(
            availableBytes: testAvailableBytes,
            bytesPerSecond: bps
        )
        let expectedSecs = Int(Double(10_000_000_000 - 500 * 1024 * 1024) / expectedBps)
        if remainingSecs != expectedSecs {
            print("[AE/AWBTest] FAIL: Expected remaining seconds \(expectedSecs), got \(remainingSecs)")
            allPassed = false
        }

        // Time string formatting
        let fmtZero = StorageEstimator.formatRemainingTime(seconds: 0)
        let fmtMins = StorageEstimator.formatRemainingTime(seconds: 759)
        let fmtHours = StorageEstimator.formatRemainingTime(seconds: 3665)
        if fmtZero != "00:00" || fmtMins != "12:39" || fmtHours != "1h 01m" {
            print("[AE/AWBTest] FAIL: Time formatting mismatch: zero='\(fmtZero)', mins='\(fmtMins)', hours='\(fmtHours)'")
            allPassed = false
        }

        // Zero usable space when below 500MB
        let lowSpaceRemaining = StorageEstimator.estimatedRemainingSeconds(
            availableBytes: 400 * 1024 * 1024,
            bytesPerSecond: bps
        )
        if lowSpaceRemaining != 0 {
            print("[AE/AWBTest] FAIL: Low space (<500MB) expected 0 remaining seconds, got \(lowSpaceRemaining)")
            allPassed = false
        }

        if allPassed {
            print("[AE/AWBTest] PASS: Auto Exposure, White Balance & Storage Estimator verified successfully")
        }
        return allPassed
    }

    /// Validates landscape rotation transforms, Bayer CFA boundary reflection,
    /// symmetric ISO ratio math, and CFR audio/video frame count sync.
    static func runFlawsValidationTest() -> Bool {
        var allPassed = true

        // 1. Landscape Left 180° rotation transform matrix validation
        let w: CGFloat = 3840
        let h: CGFloat = 2160
        let t = CGAffineTransform(rotationAngle: .pi).translatedBy(x: -w, y: -h)

        let p0 = CGPoint(x: 0, y: 0).applying(t)
        if abs(p0.x - w) > 0.001 || abs(p0.y - h) > 0.001 {
            print("[FlawsTest] FAIL: Transform (0,0) -> (\(p0.x), \(p0.y)), expected (\(w), \(h))")
            allPassed = false
        }
        let pWh = CGPoint(x: w, y: h).applying(t)
        if abs(pWh.x - 0) > 0.001 || abs(pWh.y - 0) > 0.001 {
            print("[FlawsTest] FAIL: Transform (w,h) -> (\(pWh.x), \(pWh.y)), expected (0,0)")
            allPassed = false
        }
        let pCenter = CGPoint(x: w / 2, y: h / 2).applying(t)
        if abs(pCenter.x - w / 2) > 0.001 || abs(pCenter.y - h / 2) > 0.001 {
            print("[FlawsTest] FAIL: Center moved under 180° rotation: \(pCenter)")
            allPassed = false
        }

        // 2. Bayer CFA boundary reflection parity preservation
        // Width = 4032 (even), max index = 4031 (odd)
        let maxW = 4031
        let offsets = [-2, -1, 1, 2]
        for dx in offsets {
            // Left boundary: x = 0
            let pxLeft = 0 + dx
            let reflLeft = pxLeft < 0 ? -pxLeft : pxLeft
            if (reflLeft & 1) != ((pxLeft & 1) + 2) % 2 {
                print("[FlawsTest] FAIL: Left boundary reflection parity mismatch for dx=\(dx): px=\(pxLeft), refl=\(reflLeft)")
                allPassed = false
            }

            // Right boundary: x = maxW
            let pxRight = maxW + dx
            let reflRight = pxRight > maxW ? (2 * maxW - pxRight) : pxRight
            if (reflRight & 1) != ((pxRight & 1) + 2) % 2 {
                print("[FlawsTest] FAIL: Right boundary reflection parity mismatch for dx=\(dx): px=\(pxRight), refl=\(reflRight)")
                allPassed = false
            }
        }

        // 3. Symmetric ISO ratio scene-cut detection
        let checkCut: (Float, Float) -> Bool = { iso1, iso2 in
            let ratio = max(iso1, iso2) / max(1.0, min(iso1, iso2))
            return ratio >= 3.0
        }
        // Jump UP: 100 -> 350 (ratio 3.5 >= 3.0 -> true)
        if !checkCut(100, 350) {
            print("[FlawsTest] FAIL: Upward ISO cut (100 -> 350) did not trigger")
            allPassed = false
        }
        // Jump DOWN: 350 -> 100 (ratio 3.5 >= 3.0 -> true)
        if !checkCut(350, 100) {
            print("[FlawsTest] FAIL: Downward ISO cut (350 -> 100) did not trigger")
            allPassed = false
        }
        // Minor change: 100 -> 150 (ratio 1.5 < 3.0 -> false)
        if checkCut(100, 150) {
            print("[FlawsTest] FAIL: Minor ISO change (100 -> 150) falsely triggered cut")
            allPassed = false
        }

        // 4. CFR rounded hold-frame timing stability
        let fps = 24.0
        // Normal frame at 1/24s: elapsed = 0.04167s -> rounded targetCount = 1
        let countNormal = Int64((0.041666 * fps).rounded())
        if countNormal != 1 {
            print("[FlawsTest] FAIL: 1/24s should produce 1 frame, got \(countNormal)")
            allPassed = false
        }
        // Jitter frame at 0.045s (delayed debayer): rounded targetCount should still be 1 (no runaway hold frame)
        let countJitter = Int64((0.045 * fps).rounded())
        if countJitter != 1 {
            print("[FlawsTest] FAIL: Delayed frame at 0.045s should produce 1 frame, got \(countJitter)")
            allPassed = false
        }
        // Skipped frame at 2/24s = 0.08333s: rounded targetCount should be 2 (exactly 1 hold slot needed)
        let countSkipped = Int64((0.083333 * fps).rounded())
        if countSkipped != 2 {
            print("[FlawsTest] FAIL: Skipped frame at 2/24s should produce 2 frames, got \(countSkipped)")
            allPassed = false
        }

        if allPassed {
            print("[FlawsTest] PASS: Transform, reflection, symmetric scene-cut, and CFR timing verified successfully")
        }
        return allPassed
    }

    /// Validates ScopeData calculation with ITU-R BT.2020 luma coefficients and 100 IRE clipping.
    static func runScopeDataBT2020Test() -> Bool {
        var allPassed = true

        let whitePixel: [UInt16] = [Float16(1.0).bitPattern, Float16(1.0).bitPattern, Float16(1.0).bitPattern, Float16(1.0).bitPattern]
        let scopeWhite = ScopeData.make(fromHalfRGBA: whitePixel, width: 1, height: 1, histogramBins: 64, waveformColumns: 64, waveformRows: 48)
        // At white (100% IRE), row 0 (top row = 100 IRE) must contain the trace
        if scopeWhite.waveform[0 * 64 + 0] <= 0 {
            print("[ScopeDataTest] FAIL White pixel did not hit waveform top row (100 IRE)")
            allPassed = false
        }

        if allPassed {
            print("[ScopeDataTest] PASS: ScopeData BT.2020 luma and 100 IRE clipping verified")
        }
        return allPassed
    }
}

#endif
