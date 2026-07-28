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
        let totalBytes = bytesPerRow * height

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
}

#endif
