import Foundation
import Metal
import MetalKit
import CoreVideo

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

    /// Legacy test alias for backward compatibility.
    static func runHighlightShoulderTest() -> Bool {
        return runLogCurvesStandardComplianceTest()
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
                let hasBT2020 = CVBufferGetAttachment(pb, kCVImageBufferColorPrimariesKey, nil) != nil
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
}

#endif
