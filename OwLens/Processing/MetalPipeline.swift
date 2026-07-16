import Metal
import MetalKit
import MetalPerformanceShaders
import CoreVideo
import simd

struct DebayerParams {
    var bayerPattern: Int32
    var blackLevel: Float
    var whiteLevel: Float
    var _pad: Float = 0
}

struct WhiteBalanceParams {
    var gains: SIMD3<Float>
    var colorMatrix: simd_float3x3

    static let identity = WhiteBalanceParams(
        gains: SIMD3<Float>(1, 1, 1),
        colorMatrix: matrix_identity_float3x3
    )

    static let daylight = WhiteBalanceParams(
        gains: SIMD3<Float>(1.8, 1.0, 1.4),
        colorMatrix: matrix_identity_float3x3
    )

    static let tungsten = WhiteBalanceParams(
        gains: SIMD3<Float>(1.2, 1.0, 2.2),
        colorMatrix: matrix_identity_float3x3
    )

    func metalBytes() -> [Float] {
        var bytes = [Float](repeating: 0, count: 16)
        bytes[0] = gains.x
        bytes[1] = gains.y
        bytes[2] = gains.z
        bytes[3] = 0
        let c0 = colorMatrix.columns.0
        let c1 = colorMatrix.columns.1
        let c2 = colorMatrix.columns.2
        bytes[4] = c0.x; bytes[5] = c0.y; bytes[6] = c0.z; bytes[7] = 0
        bytes[8] = c1.x; bytes[9] = c1.y; bytes[10] = c1.z; bytes[11] = 0
        bytes[12] = c2.x; bytes[13] = c2.y; bytes[14] = c2.z; bytes[15] = 0
        return bytes
    }
}

/// Matches `FusedParams` in Debayer.metal — must be identical layout.
struct FusedParams {
    var bayerPattern: Int32
    var blackLevel: Float
    var whiteLevel: Float
    var curveType: Int32
    var wbGains: SIMD3<Float>
    var _pad: Float = 0
}

/// Metal pipeline: Bayer → (CFA-safe 2× bin) → **fused** debayer+WB+log → optional crop/scale.
///
/// Optimizations vs. original 3-pass pipeline:
///   • Single kernel dispatch (debayerWBLog) instead of 3 separate encoders
///   • Pre-allocated texture pool — zero per-frame allocations
///   • Synchronous `waitUntilCompleted` only on final BGRA readback
final class MetalPipeline: @unchecked Sendable {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private let binPipeline: MTLComputePipelineState
    private let fusedPipeline: MTLComputePipelineState
    // Keep legacy pipelines for fallback / future use
    private let debayerPipeline: MTLComputePipelineState
    private let wbPipeline: MTLComputePipelineState
    private let logCurvePipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?
    private let scaler: MPSImageBilinearScale

    // ── Texture pool (avoids per-frame allocation) ──
    private var pooledBinTex: MTLTexture?
    private var pooledBinW: Int = 0
    private var pooledBinH: Int = 0
    private var pooledFusedTex: MTLTexture?
    private var pooledFusedW: Int = 0
    private var pooledFusedH: Int = 0
    private var pooledBGRATex: MTLTexture?
    private var pooledBGRAW: Int = 0
    private var pooledBGRAH: Int = 0

    var curveType: LogCurveType = .sLog3Approx
    var wbParams: WhiteBalanceParams = .identity
    var bayerPattern: Int32 = 0
    var blackLevel: Float = 0
    var whiteLevel: Float = 16383.0 / 65535.0

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let binFunc = library.makeFunction(name: "binBayerCFA"),
              let fusedFunc = library.makeFunction(name: "debayerWBLog"),
              let debayerFunc = library.makeFunction(name: "debayerBilinear"),
              let wbFunc = library.makeFunction(name: "applyWhiteBalanceAndColorMatrix"),
              let logCurveFunc = library.makeFunction(name: "applyLogCurve") else {
            return nil
        }
        self.device = device
        self.commandQueue = queue
        self.scaler = MPSImageBilinearScale(device: device)
        do {
            self.binPipeline = try device.makeComputePipelineState(function: binFunc)
            self.fusedPipeline = try device.makeComputePipelineState(function: fusedFunc)
            self.debayerPipeline = try device.makeComputePipelineState(function: debayerFunc)
            self.wbPipeline = try device.makeComputePipelineState(function: wbFunc)
            self.logCurvePipeline = try device.makeComputePipelineState(function: logCurveFunc)
        } catch {
            print("[MetalPipeline] Failed to create compute pipelines: \(error)")
            return nil
        }
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
    }

    // MARK: - Texture pool helpers

    private func getOrCreateBinTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledBinTex, pooledBinW == width, pooledBinH == height {
            return tex
        }
        let tex = makeR16Texture(width: width, height: height)
        pooledBinTex = tex
        pooledBinW = width
        pooledBinH = height
        return tex
    }

    private func getOrCreateFusedTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledFusedTex, pooledFusedW == width, pooledFusedH == height {
            return tex
        }
        let tex = makePrivateTexture(width: width, height: height)
        pooledFusedTex = tex
        pooledFusedW = width
        pooledFusedH = height
        return tex
    }

    private func getOrCreateBGRATexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledBGRATex, pooledBGRAW == width, pooledBGRAH == height {
            return tex
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderWrite, .shaderRead]
        desc.storageMode = .shared
        let tex = device.makeTexture(descriptor: desc)
        pooledBGRATex = tex
        pooledBGRAW = width
        pooledBGRAH = height
        return tex
    }

    // MARK: - Main process (fused single-kernel path)

    /// Process RAW to log RGB using the fused single-kernel path.
    /// Uses CFA-preserving 2× bin when sensor wider than 2500px.
    func process(_ pixelBuffer: CVPixelBuffer, encodeWidth: Int = 1920, encodeHeight: Int = 1440) -> MTLTexture? {
        let fullW = CVPixelBufferGetWidth(pixelBuffer)
        let fullH = CVPixelBufferGetHeight(pixelBuffer)
        guard fullW > 0, fullH > 0 else { return nil }

        guard let fullBayer = makeRawTexture(from: pixelBuffer) else {
            print("[MetalPipeline] Failed to create input texture")
            return nil
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return nil }

        // Optional CFA-safe half bin
        let bayerIn: MTLTexture
        let bayerW: Int
        let bayerH: Int
        if fullW > 2500 {
            let halfW = (fullW / 2) & ~1
            let halfH = (fullH / 2) & ~1
            guard let halfTex = getOrCreateBinTexture(width: halfW, height: halfH),
                  let enc = commandBuffer.makeComputeCommandEncoder() else { return nil }
            enc.setComputePipelineState(binPipeline)
            enc.setTexture(fullBayer, index: 0)
            enc.setTexture(halfTex, index: 1)
            dispatch(enc, width: halfW, height: halfH, state: binPipeline)
            enc.endEncoding()
            bayerIn = halfTex
            bayerW = halfW
            bayerH = halfH
        } else {
            bayerIn = fullBayer
            bayerW = fullW
            bayerH = fullH
        }

        // ── Single fused kernel: demosaic + WB + log ──
        guard let fusedOut = getOrCreateFusedTexture(width: bayerW, height: bayerH) else { return nil }

        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(fusedPipeline)
            enc.setTexture(bayerIn, index: 0)
            enc.setTexture(fusedOut, index: 1)
            var params = FusedParams(
                bayerPattern: bayerPattern,
                blackLevel: blackLevel,
                whiteLevel: max(whiteLevel, blackLevel + 1e-6),
                curveType: Int32(curveType.rawValue),
                wbGains: wbParams.gains
            )
            enc.setBytes(&params, length: MemoryLayout<FusedParams>.stride, index: 0)
            dispatch(enc, width: bayerW, height: bayerH, state: fusedPipeline)
            enc.endEncoding()
        }

        let finalTex = cropToAspectAndScale(fusedOut, targetWidth: encodeWidth, targetHeight: encodeHeight, cb: commandBuffer) ?? fusedOut
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return finalTex
    }

    func scale(_ texture: MTLTexture, width: Int, height: Int, cb: MTLCommandBuffer) -> MTLTexture? {
        guard width > 0, height > 0 else { return nil }
        if texture.width == width && texture.height == height { return texture }
        let outDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        outDesc.usage = [.shaderWrite, .shaderRead]
        outDesc.storageMode = .private
        guard let output = device.makeTexture(descriptor: outDesc) else { return nil }
        scaler.encode(commandBuffer: cb, sourceTexture: texture, destinationTexture: output)
        return output
    }

    func cropToAspectAndScale(_ texture: MTLTexture, targetWidth: Int, targetHeight: Int, cb: MTLCommandBuffer) -> MTLTexture? {
        guard targetWidth > 0, targetHeight > 0 else { return nil }
        let srcW = texture.width
        let srcH = texture.height
        guard srcW > 0, srcH > 0 else { return nil }
 
        let srcAspect = Float(srcW) / Float(srcH)
        let dstAspect = Float(targetWidth) / Float(targetHeight)
 
        var cropW = srcW
        var cropH = srcH
        var originX = 0
        var originY = 0
 
        if srcAspect > dstAspect + 0.001 {
            cropW = max(2, Int((Float(srcH) * dstAspect).rounded(.toNearestOrEven)))
            originX = max(0, (srcW - cropW) / 2)
        } else if srcAspect < dstAspect - 0.001 {
            cropH = max(2, Int((Float(srcW) / dstAspect).rounded(.toNearestOrEven)))
            originY = max(0, (srcH - cropH) / 2)
        }
 
        cropW = min(srcW - originX, cropW) & ~1
        cropH = min(srcH - originY, cropH) & ~1
        originX &= ~1
        originY &= ~1
 
        guard cropW >= 2, cropH >= 2 else {
            return scale(texture, width: targetWidth, height: targetHeight, cb: cb)
        }
 
        if cropW == srcW && cropH == srcH && srcW == targetWidth && srcH == targetHeight {
            return texture
        }
 
        let sourceForScale: MTLTexture
        if cropW == srcW && cropH == srcH {
            sourceForScale = texture
        } else {
            guard let cropped = makePrivateTexture(width: cropW, height: cropH),
                  let blit = cb.makeBlitCommandEncoder() else {
                return scale(texture, width: targetWidth, height: targetHeight, cb: cb)
            }
            blit.copy(
                from: texture,
                sourceSlice: 0, sourceLevel: 0,
                sourceOrigin: MTLOrigin(x: originX, y: originY, z: 0),
                sourceSize: MTLSize(width: cropW, height: cropH, depth: 1),
                to: cropped,
                destinationSlice: 0, destinationLevel: 0,
                destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
            )
            blit.endEncoding()
            sourceForScale = cropped
        }
 
        return scale(sourceForScale, width: targetWidth, height: targetHeight, cb: cb)
    }

    func textureToPixelBufferBGRA(
        _ texture: MTLTexture,
        completion: @escaping (CVPixelBuffer?) -> Void
    ) {
        let width = texture.width
        let height = texture.height
 
        let attrs: [String: Any] = [
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            nil, width, height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer)
        
        guard status == kCVReturnSuccess,
              let pb = pixelBuffer,
              let textureCache = textureCache else {
            completion(nil)
            return
        }
 
        var cvTextureOut: CVMetalTexture?
        let cacheStatus = CVMetalTextureCacheCreateTextureFromImage(
            nil, textureCache, pb, nil,
            .bgra8Unorm, width, height, 0, &cvTextureOut)
 
        guard cacheStatus == kCVReturnSuccess,
              let cvTex = cvTextureOut,
              let bgraTex = CVMetalTextureGetTexture(cvTex),
              let cb = commandQueue.makeCommandBuffer() else {
            completion(nil)
            return
        }
 
        // rgba16Float → bgra8Unorm via MPS scale (handles format convert) directly into the pixel buffer's memory!
        scaler.encode(commandBuffer: cb, sourceTexture: texture, destinationTexture: bgraTex)
        
        cb.addCompletedHandler { _ in
            completion(pb)
        }
        cb.commit()
    }

    // MARK: - Texture helpers

    private func makeRawTexture(from pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        if let textureCache {
            var cvTextureIn: CVMetalTexture?
            let status = CVMetalTextureCacheCreateTextureFromImage(
                nil, textureCache, pixelBuffer, nil,
                .r16Unorm, width, height, 0, &cvTextureIn)
            if status == kCVReturnSuccess,
               let cvTextureIn,
               let tex = CVMetalTextureGetTexture(cvTextureIn) {
                return tex
            }
        }
        return copyBayerToTexture(pixelBuffer)
    }

    private func copyBayerToTexture(_ pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Unorm, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead]
        desc.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: desc) else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        texture.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0,
            withBytes: base,
            bytesPerRow: bytesPerRow)
        return texture
    }

    private func makeR16Texture(width: Int, height: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Unorm, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }

    private func makePrivateTexture(width: Int, height: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }

    private func dispatch(_ enc: MTLComputeCommandEncoder, width: Int, height: Int, state: MTLComputePipelineState) {
        let tw = state.threadExecutionWidth
        let th = max(1, state.maxTotalThreadsPerThreadgroup / tw)
        let threadsPerGroup = MTLSize(width: tw, height: th, depth: 1)
        let groups = MTLSize(
            width: (width + tw - 1) / tw,
            height: (height + th - 1) / th,
            depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
    }
}
