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

    /// Verifies that the highlight shoulder curve smoothly maps sensor clipping to full container headroom.
    static func runHighlightShoulderTest() -> Bool {
        var passed = true
        // 1. Below knee (0.18 mid gray and 0.36 knee) must be strictly linear
        let midGray = LogCurve.applyHighlightShoulder(0.18, rKnee: 0.36, rMax: 10.0)
        if abs(midGray - 0.18) > 0.0001 {
            print("[HighlightShoulderTest] FAIL mid gray not identity: \(midGray)")
            passed = false
        }
        let knee = LogCurve.applyHighlightShoulder(0.36, rKnee: 0.36, rMax: 10.0)
        if abs(knee - 0.36) > 0.0001 {
            print("[HighlightShoulderTest] FAIL knee not identity: \(knee)")
            passed = false
        }

        // 2. Monotonicity & Smooth Roll-off across intermediate highlight values
        let sampleInputs: [Float] = [0.36, 0.50, 0.70, 0.90, 0.95, 0.99, 1.00]
        var lastR: Float = 0.0
        for r in sampleInputs {
            let R = LogCurve.applyHighlightShoulder(r, rKnee: 0.36, rMax: 10.0)
            if R <= lastR && r > 0.36 {
                print("[HighlightShoulderTest] FAIL non-monotonic at r=\(r): R=\(R) <= lastR=\(lastR)")
                passed = false
            }
            lastR = R
        }

        // 3. Verify no cliff at 0.99: R(0.99) should smoothly reach > 9.0 (not compressed to ~2.6)
        let r99 = LogCurve.applyHighlightShoulder(0.99, rKnee: 0.36, rMax: 10.0)
        if r99 < 9.0 {
            print("[HighlightShoulderTest] FAIL highlight cliff at 0.99: got \(r99), expected > 9.0")
            passed = false
        }

        // 4. Sensor clipping (1.0) must reach rMax (10.0)
        let maxVal = LogCurve.applyHighlightShoulder(1.0, rKnee: 0.36, rMax: 10.0)
        if abs(maxVal - 10.0) > 0.01 {
            print("[HighlightShoulderTest] FAIL max value: got \(maxVal), expected 10.0")
            passed = false
        }
        // 5. Apple Log 2 encoded code value at sensor clipping must reach > 0.95
        let codeAtClip = LogCurve.appleLog2Encode(maxVal)
        if codeAtClip < 0.95 {
            print("[HighlightShoulderTest] FAIL log code at clip too low: \(codeAtClip)")
            passed = false
        } else {
            print("[HighlightShoulderTest] PASS: sensor clipping smoothly reaches Apple Log code \(codeAtClip) without cliffs")
        }
        return passed
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
        let pass = avgMs < 33.33 // must easily fit in 30 fps budget
        print("[ThroughputBenchmark] Result: \(pass ? "PASS" : "FAIL") (target < 33.33 ms)")
        return pass
    }
}

#endif
