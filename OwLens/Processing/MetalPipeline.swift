import Metal
import MetalKit
import MetalPerformanceShaders
import CoreVideo
import simd

struct DebayerParams {
    var bayerPattern: Int32
    var blackLevel: Float
    var whiteLevel: Float
    var lscCoefficients: SIMD4<Float>
}

struct WhiteBalanceParams {
    var gains: SIMD3<Float>
    var colorMatrix: simd_float3x3

    static let identity = WhiteBalanceParams(
        gains: SIMD3<Float>(1, 1, 1),
        colorMatrix: matrix_identity_float3x3
    )
}

/// Matches `FusedParams` in Debayer.metal — must be identical layout.
struct FusedParams {
    var bayerPattern: Int32
    var blackLevel: Float
    var whiteLevel: Float
    var curveType: Int32
    var wbGains: SIMD3<Float>
    var lscCoefficients: SIMD4<Float>
}
struct BilateralParams {
    var iso: Float
}
struct DenoiseParams {
    var iso: Float
    var radius: Int32
}
struct TemporalParams {
    var iso: Float
    var maxBlend: Float
}

/// Metal pipeline: Bayer → (optional CFA-safe 2× bin) → linear debayer/WB → spatial denoise → log OETF.
///
/// Legacy fused kernels are still compiled for fallback and experimentation.
///   • Pre-allocated texture pool — zero per-frame allocations
///   • Synchronous `waitUntilCompleted` only on final BGRA readback
final class MetalPipeline: @unchecked Sendable {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private let binPipeline: MTLComputePipelineState
    private let fusedPipeline: MTLComputePipelineState
    private let chromaBilateralPipeline: MTLComputePipelineState
    private let linearPipeline: MTLComputePipelineState
    private let denoisePipeline: MTLComputePipelineState
    private let temporalPipeline: MTLComputePipelineState
    // Keep legacy pipelines for fallback / future use
    private let debayerPipeline: MTLComputePipelineState
    private let wbPipeline: MTLComputePipelineState
    private let logOnlyPipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?
    private let scaler: MPSImageBilinearScale

    // ── Texture pool (avoids per-frame allocation) ──
    private var pooledBinTex: MTLTexture?
    private var pooledBinW: Int = 0
    private var pooledBinH: Int = 0
    private var pooledFusedTex: MTLTexture?
    private var pooledFusedW: Int = 0
    private var pooledFusedH: Int = 0
    
    private var pooledLinearTex: MTLTexture?
    private var pooledLinearW: Int = 0
    private var pooledLinearH: Int = 0
    
    private var pooledDenoisedTex: MTLTexture?
    private var pooledDenoisedW: Int = 0
    private var pooledDenoisedH: Int = 0

    private var pooledTemporalHistoryTex: MTLTexture?
    private var pooledTemporalHistoryW: Int = 0
    private var pooledTemporalHistoryH: Int = 0
    private var temporalHistoryValid: Bool = false
    
    private var pooledBGRATex: MTLTexture?
    private var pooledBGRAW: Int = 0
    private var pooledBGRAH: Int = 0
    
    private var pooledCropTex: MTLTexture?
    private var pooledCropW: Int = 0
    private var pooledCropH: Int = 0
    private var pooledScaleTex: MTLTexture?
    private var pooledScaleW: Int = 0
    private var pooledScaleH: Int = 0
    private var pooledScopeTex: MTLTexture?
    private var pooledScopeW: Int = 0
    private var pooledScopeH: Int = 0
    
    private var pixelBufferPool: CVPixelBufferPool?
    private var pixelBufferPoolW: Int = 0
    private var pixelBufferPoolH: Int = 0

    var curveType: LogCurveType = .sLog3Approx
    var wbParams: WhiteBalanceParams = .identity
    var bayerPattern: Int32 = 0
    var blackLevel: Float = 0
    var whiteLevel: Float = 16383.0 / 65535.0
    var lscCoefficients: SIMD4<Float> = SIMD4<Float>(repeating: 0)
    var iso: Float = 0
    var isAutoWBEnabled: Bool = true

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let binFunc = library.makeFunction(name: "binBayerCFA"),
              let fusedFunc = library.makeFunction(name: "debayerWBLog"),
              let chromaFunc = library.makeFunction(name: "chromaBilateral"),
              let linearFunc = library.makeFunction(name: "debayerWBLinear"),
              let denoiseFunc = library.makeFunction(name: "spatialDenoise"),
              let temporalFunc = library.makeFunction(name: "temporalDenoise"),
              let debayerFunc = library.makeFunction(name: "debayerBilinear"),
              let wbFunc = library.makeFunction(name: "applyWhiteBalanceAndColorMatrix"),
              let logOnlyFunc = library.makeFunction(name: "applyLogOnly") else {
            return nil
        }
        self.device = device
        self.commandQueue = queue
        self.scaler = MPSImageBilinearScale(device: device)
        do {
            self.binPipeline = try device.makeComputePipelineState(function: binFunc)
            self.fusedPipeline = try device.makeComputePipelineState(function: fusedFunc)
            self.chromaBilateralPipeline = try device.makeComputePipelineState(function: chromaFunc)
            self.linearPipeline = try device.makeComputePipelineState(function: linearFunc)
            self.denoisePipeline = try device.makeComputePipelineState(function: denoiseFunc)
            self.temporalPipeline = try device.makeComputePipelineState(function: temporalFunc)
            self.debayerPipeline = try device.makeComputePipelineState(function: debayerFunc)
            self.wbPipeline = try device.makeComputePipelineState(function: wbFunc)
            self.logOnlyPipeline = try device.makeComputePipelineState(function: logOnlyFunc)
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

    private func getOrCreateLinearTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledLinearTex, pooledLinearW == width, pooledLinearH == height {
            return tex
        }
        let tex = makePrivateTexture(width: width, height: height)
        pooledLinearTex = tex
        pooledLinearW = width
        pooledLinearH = height
        return tex
    }

    private func getOrCreateDenoisedTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledDenoisedTex, pooledDenoisedW == width, pooledDenoisedH == height {
            return tex
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderWrite, .shaderRead]
        desc.storageMode = .private
        
        guard let tex = device.makeTexture(descriptor: desc) else { return nil }
        pooledDenoisedTex = tex
        pooledDenoisedW = width
        pooledDenoisedH = height
        return tex
    }

    private func getOrCreateTemporalHistoryTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledTemporalHistoryTex, pooledTemporalHistoryW == width, pooledTemporalHistoryH == height {
            return tex
        }
        let tex = makePrivateTexture(width: width, height: height)
        pooledTemporalHistoryTex = tex
        pooledTemporalHistoryW = width
        pooledTemporalHistoryH = height
        temporalHistoryValid = false
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

    private func getOrCreateCropTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledCropTex, pooledCropW == width, pooledCropH == height {
            return tex
        }
        let tex = makePrivateTexture(width: width, height: height)
        pooledCropTex = tex
        pooledCropW = width
        pooledCropH = height
        return tex
    }

    private func getOrCreateScaleTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledScaleTex, pooledScaleW == width, pooledScaleH == height {
            return tex
        }
        let tex = makePrivateTexture(width: width, height: height)
        pooledScaleTex = tex
        pooledScaleW = width
        pooledScaleH = height
        return tex
    }

    private func getOrCreateScopeTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledScopeTex, pooledScopeW == width, pooledScopeH == height {
            return tex
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .shared
        let tex = device.makeTexture(descriptor: desc)
        pooledScopeTex = tex
        pooledScopeW = width
        pooledScopeH = height
        return tex
    }

    private func getOrCreatePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        if pixelBufferPool == nil || pixelBufferPoolW != width || pixelBufferPoolH != height {
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferMetalCompatibilityKey as String: true,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
            ]
            let poolAttrs: [String: Any] = [
                kCVPixelBufferPoolMinimumBufferCountKey as String: 6
            ]
            var pool: CVPixelBufferPool?
            let status = CVPixelBufferPoolCreate(
                nil,
                poolAttrs as CFDictionary,
                attrs as CFDictionary,
                &pool
            )
            guard status == kCVReturnSuccess else { return nil }
            pixelBufferPool = pool
            pixelBufferPoolW = width
            pixelBufferPoolH = height
        }

        guard let pixelBufferPool else { return nil }
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(nil, pixelBufferPool, &pixelBuffer)
        guard status == kCVReturnSuccess else { return nil }
        return pixelBuffer
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

        // ── 4-Pass Linear+Temporal Denoising Pipeline ──
        // Pass 1: debayerWBLinear (Demosaic + LSC + WB -> Linear RGB)
        // Pass 2: spatialDenoise (Luma + Chroma Bilateral in Linear space, shadow-adaptive)
        // Pass 3: temporalDenoise (Motion-adaptive blend with frame history)
        // Pass 4: applyLogOnly (Linear RGB -> S-Log3 / Log2)
        
        guard let linearOut = getOrCreateLinearTexture(width: bayerW, height: bayerH),
              let denoisedOut = getOrCreateDenoisedTexture(width: bayerW, height: bayerH),
              let temporalHistoryOut = getOrCreateTemporalHistoryTexture(width: bayerW, height: bayerH),
              let fusedOut = getOrCreateFusedTexture(width: bayerW, height: bayerH) else { return nil }

        // Pass 1: Linear Demosaic + WB
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(linearPipeline)
            enc.setTexture(bayerIn, index: 0)
            enc.setTexture(linearOut, index: 1)
            var params = FusedParams(
                bayerPattern: bayerPattern,
                blackLevel: blackLevel,
                whiteLevel: max(whiteLevel, blackLevel + 1e-6),
                curveType: Int32(curveType.rawValue),
                wbGains: wbParams.gains,
                lscCoefficients: lscCoefficients
            )
            enc.setBytes(&params, length: MemoryLayout<FusedParams>.stride, index: 0)
            dispatch(enc, width: bayerW, height: bayerH, state: linearPipeline)
            enc.endEncoding()
        }

        // Pass 2: Spatial Denoise in Linear Space (Luma + Chroma, ISO-adaptive radius)
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(denoisePipeline)
            enc.setTexture(linearOut, index: 0)
            enc.setTexture(denoisedOut, index: 1)
            
            let radius: Int32 = iso > 200 ? 3 : 2
            var dParams = DenoiseParams(iso: iso, radius: radius)
            enc.setBytes(&dParams, length: MemoryLayout<DenoiseParams>.stride, index: 0)
            
            dispatch(enc, width: bayerW, height: bayerH, state: denoisePipeline)
            enc.endEncoding()
        }

        // Pass 3: Temporal Denoise in Linear Space (Y+UV motion gating)
        let temporalOut = linearOut
        if temporalHistoryValid {
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(temporalPipeline)
                enc.setTexture(denoisedOut, index: 0)
                enc.setTexture(temporalHistoryOut, index: 1)
                enc.setTexture(temporalOut, index: 2)

                let isoClamped = max(iso, 33.0)
                let isoNorm = min(max((isoClamped - 33.0) / (1600.0 - 33.0), 0.0), 1.0)
                let maxBlend: Float = 0.20 + 0.35 * isoNorm
                var tParams = TemporalParams(iso: isoClamped, maxBlend: maxBlend)
                enc.setBytes(&tParams, length: MemoryLayout<TemporalParams>.stride, index: 0)

                dispatch(enc, width: bayerW, height: bayerH, state: temporalPipeline)
                enc.endEncoding()
            }
        } else if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.copy(
                from: denoisedOut,
                sourceSlice: 0, sourceLevel: 0,
                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: bayerW, height: bayerH, depth: 1),
                to: temporalOut,
                destinationSlice: 0, destinationLevel: 0,
                destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
            )
            blit.endEncoding()
        } else {
            return nil
        }

        // Update temporal history from current filtered linear frame.
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.copy(
                from: temporalOut,
                sourceSlice: 0, sourceLevel: 0,
                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: bayerW, height: bayerH, depth: 1),
                to: temporalHistoryOut,
                destinationSlice: 0, destinationLevel: 0,
                destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
            )
            blit.endEncoding()
            temporalHistoryValid = true
        } else {
            return nil
        }

        // Pass 4: Log OETF
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(logOnlyPipeline)
            enc.setTexture(temporalOut, index: 0)
            enc.setTexture(fusedOut, index: 1)
            
            var cType = Int32(curveType.rawValue)
            enc.setBytes(&cType, length: MemoryLayout<Int32>.stride, index: 0)
            
            dispatch(enc, width: bayerW, height: bayerH, state: logOnlyPipeline)
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
        guard let output = getOrCreateScaleTexture(width: width, height: height) else { return nil }
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
            guard let cropped = getOrCreateCropTexture(width: cropW, height: cropH),
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

    func makeScopeData(from texture: MTLTexture, sampleWidth: Int = 96, sampleHeight: Int = 54) -> ScopeData? {
        let width = max(16, sampleWidth)
        let height = max(16, sampleHeight)
        guard let output = getOrCreateScopeTexture(width: width, height: height),
              let cb = commandQueue.makeCommandBuffer() else { return nil }

        scaler.encode(commandBuffer: cb, sourceTexture: texture, destinationTexture: output)
        cb.commit()
        cb.waitUntilCompleted()

        let componentsPerPixel = 4
        let bytesPerComponent = MemoryLayout<UInt16>.stride
        let bytesPerRow = width * componentsPerPixel * bytesPerComponent
        var pixels = [UInt16](repeating: 0, count: width * height * componentsPerPixel)
        output.getBytes(
            &pixels,
            bytesPerRow: bytesPerRow,
            from: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0
        )
        return ScopeData.make(fromHalfRGBA: pixels, width: width, height: height)
    }

    func textureToPixelBufferBGRA(
        _ texture: MTLTexture,
        completion: @escaping (CVPixelBuffer?) -> Void
    ) {
        let width = texture.width
        let height = texture.height
 
        guard let pb = getOrCreatePixelBuffer(width: width, height: height),
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
