import Metal
import MetalKit
import MetalPerformanceShaders
import CoreVideo
import simd

/// Wraps a non-Sendable value so it can be captured by a `@Sendable` closure
/// (e.g. `MTLTexture`, `CVPixelBuffer?`, or completion callbacks inside
/// `addCompletedHandler`). The wrapped value is only read from the callback's
/// serialized execution context, so `@unchecked` isolation is sound. This
/// mirrors the codebase's existing `SendablePixelBuffer` idiom.
struct SendableBox<Value>: @unchecked Sendable {
    let value: Value
}

struct DebayerParams {
    var bayerPattern: Int32
    var blackLevel: Float
    var whiteLevel: Float
    var lscCoefficients: SIMD4<Float>
}

struct DefectPixelParams {
    var shotCoeff: Float
    var readCoeff: Float
}

struct WhiteBalanceParams {
    var gains: SIMD3<Float>
    var colorMatrix: simd_float3x3

    static let identity = WhiteBalanceParams(
        gains: SIMD3<Float>(1, 1, 1),
        colorMatrix: matrix_identity_float3x3
    )

    /// Calibrated CCM mapping white-balanced iPhone Sony Bayer sensor RGB to ITU-R BT.2020 container (D65).
    /// Calibrated so that DaVinci Resolve CST (BT.2020 -> Rec.709) produces natural, un-distorted skin tones
    /// and preserves neutral white balance (row sums equal 1.0).
    static let defaultSensorToBT2020 = simd_float3x3(
        SIMD3<Float>( 0.8595, -0.0380, -0.0073), // column 0
        SIMD3<Float>( 0.1842,  1.1232, -0.0569), // column 1
        SIMD3<Float>(-0.0437, -0.0852,  1.0643)  // column 2
    )

    /// Calibrated CCM mapping white-balanced iPhone Sony Bayer sensor RGB to Sony S-Gamut3.Cine container (D65).
    /// Preserves neutral white balance (row sums equal 1.0).
    static let defaultSensorToSGamut3Cine = simd_float3x3(
        SIMD3<Float>( 0.8955,  0.0099,  0.0175), // column 0
        SIMD3<Float>( 0.0807,  0.8915, -0.0014), // column 1
        SIMD3<Float>( 0.0237,  0.0986,  0.9838)  // column 2
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
    var greenBalance: Float
    var headroomScale: Float
}

struct LogOnlyParams {
    var curveType: Int32
    var headroomScale: Float
}

struct LSCParams {
    var radialR: Float
    var radialG: Float
    var radialB: Float
    var radial4R: Float
    var radial4G: Float
    var radial4B: Float
    var azimuthR: Float
    var azimuthG: Float
    var azimuthB: Float
}
struct DenoiseParams {
    var iso: Float
    var radius: Int32
    var shotCoeff: Float
    var readCoeff: Float
    var strength: Float  // 0.0-1.0 adaptive denoise intensity
}
struct RingTemporalParams {
    var iso: Float
    var maxBlend: Float
    var slotCount: Int32
    var validSlots: Int32
    var chromaW: Int32
    var chromaH: Int32
    var cursor: Int32
    var lambda: Float
    var shotCoeff: Float
    var readCoeff: Float
}
struct StoreChromaParams {
    var slice: Int32
}

enum ProcessingQuality {
    /// Lower latency, skips expensive passes. Used for preview when not recording.
    case previewFast
    /// Full quality. Used when recording or when user explicitly wants highest quality.
    case recordQuality
}

/// Metal pipeline: Bayer → (optional CFA-safe phase-preserving 2× reduce) → linear debayer/WB
/// → full-res luma denoise + half-res chroma denoise → temporal denoise → log OETF.
///
/// Legacy fused kernels are still compiled for fallback and experimentation.
///   • Pre-allocated texture pool — zero per-frame allocations
///   • Synchronous `waitUntilCompleted` only on final BGRA readback
final class MetalPipeline: @unchecked Sendable {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private let scopeCommandQueue: MTLCommandQueue
    private let binPipeline: MTLComputePipelineState
    private let linearPipeline: MTLComputePipelineState
    private let denoisePipeline: MTLComputePipelineState
    private let extractChromaPipeline: MTLComputePipelineState
    private let denoiseChromaPipeline: MTLComputePipelineState
    private let recombineChromaPipeline: MTLComputePipelineState
    private let temporalRingPipeline: MTLComputePipelineState
    private let storeChromaHistoryPipeline: MTLComputePipelineState
    private let storeLumaHistoryPipeline: MTLComputePipelineState
    private let globalMotionPipeline: MTLComputePipelineState
    private let logOnlyPipeline: MTLComputePipelineState
    private let debayerFusedPipeline: MTLComputePipelineState
    private let convertFormatPipeline: MTLComputePipelineState
    private let unsharpPipeline: MTLComputePipelineState
    private let lumaStatsPipeline: MTLComputePipelineState
    private let defectPixelPipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?
    private let scaler: MPSImageScale

    // Global motion metric buffer (1 float) used to skip temporal blending during motion.
    private let motionMetricBuffer: MTLBuffer

    // ── Texture pool (avoids per-frame allocation) ──
    private var pooledRawTex: MTLTexture?
    private var pooledRawW: Int = 0
    private var pooledRawH: Int = 0
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

    private var pooledSharpenTex: MTLTexture?
    private var pooledSharpenW: Int = 0
    private var pooledSharpenH: Int = 0

    private var pooledChromaRawTex: MTLTexture?
    private var pooledChromaRawW: Int = 0
    private var pooledChromaRawH: Int = 0
    private var pooledChromaDenoisedTex: MTLTexture?
    private var pooledChromaDenoisedW: Int = 0
    private var pooledChromaDenoisedH: Int = 0
    private var pooledChromaMergedTex: MTLTexture?
    private var pooledChromaMergedW: Int = 0
    private var pooledChromaMergedH: Int = 0

    private var pooledCorrectedBayerTex: MTLTexture?
    private var pooledCorrectedBayerW: Int = 0
    private var pooledCorrectedBayerH: Int = 0

    // Temporal ring buffer (N-slot history, per-channel resolution split)
    private let temporalRingCapacity = 3
    private var lumaHistoryArray: MTLTexture?
    private var chromaHistoryArray: MTLTexture?
    private var lumaHistoryW: Int = 0
    private var lumaHistoryH: Int = 0
    private var chromaHistoryW: Int = 0
    private var chromaHistoryH: Int = 0
    private var temporalRingCursor: Int = 0
    private var temporalRingValidCount: Int = 0

    // Local-sigma luma stats (full-res; skipped at 4K for performance)
    private var pooledLumaStatsTex: MTLTexture?
    private var pooledLumaStatsW: Int = 0
    private var pooledLumaStatsH: Int = 0
    private lazy var dummyStatsTex: MTLTexture? = {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Float, width: 1, height: 1, mipmapped: false)
        desc.usage = [.shaderRead]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }()

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
    /// Scene reflectance headroom multiplier (e.g. 10.0 for Apple Log 2 / S-Log3).
    /// Maps sensor clipping to the top of the log container with a filmic highlight shoulder.
    var headroomScale: Float = 10.0
    var wbParams: WhiteBalanceParams = .identity
    var bayerPattern: Int32 = 0
    var blackLevel: Float = 0
    var whiteLevel: Float = 16383.0 / 65535.0
    var lscCoefficients: SIMD4<Float> = SIMD4<Float>(repeating: 0)
    var lscParams: LSCParams = LSCParams(
        radialR: 0, radialG: 0, radialB: 0,
        radial4R: 0, radial4G: 0, radial4B: 0,
        azimuthR: 0, azimuthG: 0, azimuthB: 0)
    var greenBalance: Float = 1.0
    var iso: Float = 0
    var noiseShotCoeff: Float = 0.012
    var noiseReadCoeff: Float = 0.0004
    var isAutoWBEnabled: Bool = true
    /// Quality mode for the next process() call. Set before calling process().
    var processingQuality: ProcessingQuality = .previewFast

    /// Device thermal state for dynamic workload shedding (.serious / .critical).
    var thermalState: ProcessInfo.ThermalState = .nominal

    /// Static denoise strength 0.0-1.0, set once per recording session by CameraViewModel
    /// based on resolution, chip tier, and recording format. 0 = no spatial/chroma denoise,
    /// 1 = maximum (2× sigma radius). Multiplies sigmaRef in bilateral kernels.
    var denoiseStrength: Float = 0.5
    /// Adaptive edge sharpness strength (0.0 = off, 0.5 = natural cinema sharpness, 1.0 = sharp).
    var sharpnessStrength: Float = 0.5

    /// Real-time frame-budget gate for the full denoise stack. The complete
    /// spatial+chroma+temporal denoise (~11 full-res passes) only fits the 30 fps
    /// CFR budget at small working resolutions; at ~3 MP it costs ~45 ms — which
    /// is exactly the 0.5× ultrawide case and overruns the 33 ms/30 fps budget,
    /// forcing the VideoWriter hold-fill (judder). This gate makes `process`
    /// skip the spatial/chroma passes when the (binned) Bayer pixel count
    /// exceeds the threshold — the same fallback 4K recording already gets via
    /// `previewFast`. Raise on faster silicon, lower on slower chips.
    private static let maxFullDenoisePixelsForRecord: Int = 2_000_000

    init?(customLibraryURL: URL? = nil) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = customLibraryURL.flatMap({ try? device.makeLibrary(URL: $0) }) ?? device.makeDefaultLibrary(),
              let binFunc = library.makeFunction(name: "binBayerCFA"),
              let linearFunc = library.makeFunction(name: "debayerWBLinear"),
              let denoiseFunc = library.makeFunction(name: "spatialDenoise"),
              let extractChromaFunc = library.makeFunction(name: "extractHalfResChroma"),
              let denoiseChromaFunc = library.makeFunction(name: "denoiseHalfResChroma"),
              let recombineChromaFunc = library.makeFunction(name: "recombineLumaWithHalfResChroma"),
              let temporalRingFunc = library.makeFunction(name: "temporalDenoiseRing"),
              let storeChromaFunc = library.makeFunction(name: "storeChromaHistory"),
              let globalMotionFunc = library.makeFunction(name: "estimateGlobalMotion"),
              let logOnlyFunc = library.makeFunction(name: "applyLogOnly"),
              let debayerFusedFunc = library.makeFunction(name: "debayerFusedLog"),
              let convertFormatFunc = library.makeFunction(name: "convertRgba16FloatToBgra8"),
              let lumaStatsFunc = library.makeFunction(name: "estimateLumaVariance"),
              let defectPixelFunc = library.makeFunction(name: "correctDefectPixelsBayer"),
              let storeLumaHistoryFunc = library.makeFunction(name: "storeLumaHistory"),
              let unsharpFunc = library.makeFunction(name: "unsharpMaskAdaptive"),
              let motionMetricBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared) else {
            return nil
        }
        guard let scopeQueue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = queue
        self.scopeCommandQueue = scopeQueue
        self.scaler = MPSImageLanczosScale(device: device)
        do {
            self.binPipeline = try device.makeComputePipelineState(function: binFunc)
            self.linearPipeline = try device.makeComputePipelineState(function: linearFunc)
            self.unsharpPipeline = try device.makeComputePipelineState(function: unsharpFunc)
            self.denoisePipeline = try device.makeComputePipelineState(function: denoiseFunc)
            self.extractChromaPipeline = try device.makeComputePipelineState(function: extractChromaFunc)
            self.denoiseChromaPipeline = try device.makeComputePipelineState(function: denoiseChromaFunc)
            self.recombineChromaPipeline = try device.makeComputePipelineState(function: recombineChromaFunc)
            self.temporalRingPipeline = try device.makeComputePipelineState(function: temporalRingFunc)
            self.storeChromaHistoryPipeline = try device.makeComputePipelineState(function: storeChromaFunc)
            self.globalMotionPipeline = try device.makeComputePipelineState(function: globalMotionFunc)
            self.logOnlyPipeline = try device.makeComputePipelineState(function: logOnlyFunc)
            self.debayerFusedPipeline = try device.makeComputePipelineState(function: debayerFusedFunc)
            self.convertFormatPipeline = try device.makeComputePipelineState(function: convertFormatFunc)
            self.lumaStatsPipeline = try device.makeComputePipelineState(function: lumaStatsFunc)
            self.defectPixelPipeline = try device.makeComputePipelineState(function: defectPixelFunc)
            self.storeLumaHistoryPipeline = try device.makeComputePipelineState(function: storeLumaHistoryFunc)
            self.motionMetricBuffer = motionMetricBuffer
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

    private func getOrCreateSharpenTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledSharpenTex, pooledSharpenW == width, pooledSharpenH == height {
            return tex
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderWrite, .shaderRead]
        desc.storageMode = .private
        guard let tex = device.makeTexture(descriptor: desc) else { return nil }
        pooledSharpenTex = tex
        pooledSharpenW = width
        pooledSharpenH = height
        return tex
    }

    private func getOrCreateChromaRawTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledChromaRawTex, pooledChromaRawW == width, pooledChromaRawH == height {
            return tex
        }
        let tex = makeRGTexture(width: width, height: height)
        pooledChromaRawTex = tex
        pooledChromaRawW = width
        pooledChromaRawH = height
        return tex
    }

    private func getOrCreateChromaDenoisedTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledChromaDenoisedTex, pooledChromaDenoisedW == width, pooledChromaDenoisedH == height {
            return tex
        }
        let tex = makeRGTexture(width: width, height: height)
        pooledChromaDenoisedTex = tex
        pooledChromaDenoisedW = width
        pooledChromaDenoisedH = height
        return tex
    }

    private func getOrCreateChromaMergedTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledChromaMergedTex, pooledChromaMergedW == width, pooledChromaMergedH == height {
            return tex
        }
        let tex = makePrivateTexture(width: width, height: height)
        pooledChromaMergedTex = tex
        pooledChromaMergedW = width
        pooledChromaMergedH = height
        return tex
    }

    private func getOrCreateCorrectedBayerTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledCorrectedBayerTex, pooledCorrectedBayerW == width, pooledCorrectedBayerH == height { return tex }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        pooledCorrectedBayerTex = device.makeTexture(descriptor: desc)
        pooledCorrectedBayerW = width
        pooledCorrectedBayerH = height
        return pooledCorrectedBayerTex
    }

    private func getOrCreateLumaStatsTexture(width: Int, height: Int) -> MTLTexture? {
        if let tex = pooledLumaStatsTex, pooledLumaStatsW == width, pooledLumaStatsH == height { return tex }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg16Float,
            width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        pooledLumaStatsTex = device.makeTexture(descriptor: desc)
        pooledLumaStatsW = width
        pooledLumaStatsH = height
        return pooledLumaStatsTex
    }

    // MARK: - Temporal ring buffer allocation

    /// Allocate or reuse ring buffer textures (2D array type). Resets ring on size change.
    private func ensureTemporalRing(lumaW: Int, lumaH: Int, chromaW: Int, chromaH: Int) {
        guard lumaW > 0, lumaH > 0, chromaW > 0, chromaH > 0 else { return }
        if lumaHistoryW == lumaW, lumaHistoryH == lumaH,
           chromaHistoryW == chromaW, chromaHistoryH == chromaH,
           lumaHistoryArray != nil { return }

        // Luma history stored as single-channel R16Unorm (not rgba16Float) to save 4x bandwidth.
        // The storeLumaHistory kernel converts RGB to Y before writing.
        let lumaDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Unorm, width: lumaW, height: lumaH, mipmapped: false)
        lumaDesc.textureType = .type2DArray
        lumaDesc.arrayLength = temporalRingCapacity
        lumaDesc.usage = [.shaderRead, .shaderWrite]
        lumaDesc.storageMode = .private
        lumaHistoryArray = device.makeTexture(descriptor: lumaDesc)
        chromaHistoryArray = makeTexture2DArray(width: chromaW, height: chromaH, arrayLength: temporalRingCapacity)
        lumaHistoryW = lumaW
        lumaHistoryH = lumaH
        chromaHistoryW = chromaW
        chromaHistoryH = chromaH
        temporalRingCursor = 0
        temporalRingValidCount = 0
    }

    func clearTemporalHistory() {
        temporalRingCursor = 0
        temporalRingValidCount = 0
    }

    /// Releases recording and temporal textures to reduce VRAM and cache pressure when idle or stopping recording.
    func trimMemory() {
        clearTemporalHistory()
        lumaHistoryArray = nil
        chromaHistoryArray = nil
        lumaHistoryW = 0
        lumaHistoryH = 0
        chromaHistoryW = 0
        chromaHistoryH = 0

        pooledDenoisedTex = nil
        pooledDenoisedW = 0
        pooledDenoisedH = 0

        pooledChromaRawTex = nil
        pooledChromaRawW = 0
        pooledChromaRawH = 0

        pooledChromaDenoisedTex = nil
        pooledChromaDenoisedW = 0
        pooledChromaDenoisedH = 0

        pooledChromaMergedTex = nil
        pooledChromaMergedW = 0
        pooledChromaMergedH = 0

        pooledLumaStatsTex = nil
        pooledLumaStatsW = 0
        pooledLumaStatsH = 0

        pooledRawTex = nil
        pooledRawW = 0
        pooledRawH = 0

        if let cache = textureCache {
            CVMetalTextureCacheFlush(cache, 0)
        }
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
                kCVPixelBufferPoolMinimumBufferCountKey as String: 12
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

    // MARK: - Main process (linear denoise path)

    /// Process RAW to log RGB using the linear denoise path.
    /// Uses CFA-preserving 2× reduction only when the reduced frame still covers the encode size.
    /// The completion handler is called on an internal Metal queue once the GPU work finishes.
    func process(_ pixelBuffer: CVPixelBuffer,
                 encodeWidth: Int = 1920,
                 encodeHeight: Int = 1440,
                 completion: @escaping (MTLTexture?) -> Void) {
        process(pixelBuffer, encodeWidth: encodeWidth, encodeHeight: encodeHeight, encodeAsBGRA: false) { texture, _ in
            completion(texture)
        }
    }

    func process(_ pixelBuffer: CVPixelBuffer,
                 encodeWidth: Int = 1920,
                 encodeHeight: Int = 1440,
                 encodeAsBGRA: Bool = false,
                 completion: @escaping (MTLTexture?, CVPixelBuffer?) -> Void) {
#if DEBUG
        let t0 = CACurrentMediaTime()
#endif
        let fullW = CVPixelBufferGetWidth(pixelBuffer)
        let fullH = CVPixelBufferGetHeight(pixelBuffer)
        guard fullW > 0, fullH > 0 else { completion(nil, nil); return }

        guard let fullBayer = makeRawTexture(from: pixelBuffer) else {
            print("[MetalPipeline] Failed to create input texture")
            completion(nil, nil); return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { completion(nil, nil); return }

        // Optional CFA-safe half reduction. This is phase-preserving, not true averaged sensor binning.
        // For recording, 4K, or high-res, NEVER decimate the RAW Bayer frame.
        // We preserve 100% of native sensor pixels (e.g. 4032x3024) to achieve sharp,
        // pristine cinema detail. Only reduce in previewFast when half-res covers the frame.
        var bayerIn: MTLTexture
        let bayerW: Int
        let bayerH: Int
        let halfW = (fullW / 2) & ~1
        let halfH = (fullH / 2) & ~1
        let is4K = encodeWidth >= 3840
        let isRecordQuality = (processingQuality == .recordQuality)
        let canReduceRaw = !isRecordQuality && !is4K && (halfW >= encodeWidth && halfH >= encodeHeight)
        if canReduceRaw {
            guard let halfTex = getOrCreateBinTexture(width: halfW, height: halfH),
                  let enc = commandBuffer.makeComputeCommandEncoder() else { completion(nil, nil); return }
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

        // Skip defect pixel correction at 4K where a single hot pixel is 1/12M and invisible.
        if !is4K && bayerW <= 3000 {
            // Defect pixel correction operates on (possibly binned) Bayer data.
            guard let correctedBayerPass = getOrCreateCorrectedBayerTexture(width: bayerW, height: bayerH),
                  let encDPC = commandBuffer.makeComputeCommandEncoder() else { completion(nil, nil); return }
            encDPC.setComputePipelineState(defectPixelPipeline)
            encDPC.setTexture(bayerIn, index: 0)
            encDPC.setTexture(correctedBayerPass, index: 1)
            var debayerParams = DebayerParams(
                bayerPattern: bayerPattern,
                blackLevel: blackLevel,
                whiteLevel: max(whiteLevel, blackLevel + 1e-6),
                lscCoefficients: lscCoefficients
            )
            var defectNoiseParams = DefectPixelParams(shotCoeff: noiseShotCoeff, readCoeff: noiseReadCoeff)
            encDPC.setBytes(&debayerParams, length: MemoryLayout<DebayerParams>.stride, index: 0)
            encDPC.setBytes(&defectNoiseParams, length: MemoryLayout<DefectPixelParams>.stride, index: 1)
            dispatch(encDPC, width: bayerW, height: bayerH, state: defectPixelPipeline)
            encDPC.endEncoding()
            bayerIn = correctedBayerPass
        }

        // ── Linear+Temporal Denoising Pipeline ──
        // Pass 1: debayerWBLinear (Demosaic + LSC + WB -> Linear RGB)
        // Pass 2: spatialDenoise (full-res luma detail preservation)
        // Pass 3: half-res chroma extraction + chroma-only bilateral denoise
        // Pass 4: recombine full-res luma with upsampled half-res chroma
        // Pass 5: temporalDenoise — ring buffer N-slot weighted average
        // Pass 6: applyLogOnly (Linear RGB -> S-Log3 / Log2)
        
        // ── Adaptive spatial/chroma denoise (gated by real-time frame budget & thermal) ──
        let overRealtimeBudget = bayerW * bayerH > Self.maxFullDenoisePixelsForRecord
        let isThermalCritical = thermalState == .critical
        let isThermalThrottled = thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue
        let runDenoise = denoiseStrength >= 0.15 && !overRealtimeBudget && !isThermalCritical

        guard let fusedOut = getOrCreateFusedTexture(width: bayerW, height: bayerH) else { completion(nil, nil); return }
        var outputTex: MTLTexture = fusedOut

        if !runDenoise {
            // ── Ultra-Fast Fused Path: Demosaic + LSC + WB + CCM + Log OETF in ONE single kernel dispatch ──
            // Eliminates 1 full pass and ~196 MB/frame of intermediate memory traffic.
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(debayerFusedPipeline)
                enc.setTexture(bayerIn, index: 0)
                enc.setTexture(fusedOut, index: 1)
                var params = FusedParams(
                    bayerPattern: bayerPattern,
                    blackLevel: blackLevel,
                    whiteLevel: max(whiteLevel, blackLevel + 1e-6),
                    curveType: Int32(curveType.rawValue),
                    wbGains: wbParams.gains,
                    lscCoefficients: lscCoefficients,
                    greenBalance: greenBalance,
                    headroomScale: curveType == .linear ? 1.0 : headroomScale
                )
                enc.setBytes(&params, length: MemoryLayout<FusedParams>.stride, index: 0)
                enc.setBytes(&lscParams, length: MemoryLayout<LSCParams>.stride, index: 1)
                var cMatrix = wbParams.colorMatrix
                enc.setBytes(&cMatrix, length: MemoryLayout<simd_float3x3>.stride, index: 2)
                dispatch(enc, width: bayerW, height: bayerH, state: debayerFusedPipeline)
                enc.endEncoding()
            }

            if sharpnessStrength > 0.001, let sharpenTex = getOrCreateSharpenTexture(width: bayerW, height: bayerH),
               let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(unsharpPipeline)
                enc.setTexture(fusedOut, index: 0)
                enc.setTexture(sharpenTex, index: 1)
                var s = sharpnessStrength
                enc.setBytes(&s, length: MemoryLayout<Float>.stride, index: 0)
                dispatch(enc, width: bayerW, height: bayerH, state: unsharpPipeline)
                enc.endEncoding()
                outputTex = sharpenTex
            }
        } else {
            // ── Full Multi-Pass Denoise Path ──
            let chromaW = max(1, (bayerW + 1) / 2)
            let chromaH = max(1, (bayerH + 1) / 2)
            guard let linearOut = getOrCreateLinearTexture(width: bayerW, height: bayerH),
                  let chromaMergedOut = getOrCreateChromaMergedTexture(width: bayerW, height: bayerH),
                  let denoisedOut = getOrCreateDenoisedTexture(width: bayerW, height: bayerH),
                  let chromaRawOut = getOrCreateChromaRawTexture(width: chromaW, height: chromaH),
                  let chromaDenoisedOut = getOrCreateChromaDenoisedTexture(width: chromaW, height: chromaH) else { completion(nil, nil); return }

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
                    lscCoefficients: lscCoefficients,
                    greenBalance: greenBalance,
                    headroomScale: 1.0
                )
                enc.setBytes(&params, length: MemoryLayout<FusedParams>.stride, index: 0)
                enc.setBytes(&lscParams, length: MemoryLayout<LSCParams>.stride, index: 1)
                var cMatrix = wbParams.colorMatrix
                enc.setBytes(&cMatrix, length: MemoryLayout<simd_float3x3>.stride, index: 2)
                dispatch(enc, width: bayerW, height: bayerH, state: linearPipeline)
                enc.endEncoding()
            }

            // Ensure temporal ring buffers exist at the correct per-channel resolutions.
            ensureTemporalRing(lumaW: bayerW, lumaH: bayerH, chromaW: chromaW, chromaH: chromaH)

            // Pass 1.5: Per-pixel local-sigma guide. Skipped at 4K, in previewFast, or during thermal throttle.
            let useLocalSigma = processingQuality == .recordQuality && !is4K && bayerW < 3000 && !isThermalThrottled
            let lumaStatsTex: MTLTexture? = useLocalSigma
                ? getOrCreateLumaStatsTexture(width: bayerW, height: bayerH)
                : dummyStatsTex
            if useLocalSigma, let statsTex = lumaStatsTex,
               let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(lumaStatsPipeline)
                enc.setTexture(linearOut, index: 0)
                enc.setTexture(statsTex, index: 1)
                dispatch(enc, width: bayerW, height: bayerH, state: lumaStatsPipeline)
                enc.endEncoding()
            }

            let denoiseRadius: Int32 = (processingQuality == .previewFast || isThermalThrottled) ? 2 : (iso > 200 ? 3 : 2)

            // Pass 2: Spatial denoise full-res luma.
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(denoisePipeline)
                enc.setTexture(linearOut, index: 0)
                enc.setTexture(denoisedOut, index: 1)
                enc.setTexture(lumaStatsTex, index: 2)
                var dParams = DenoiseParams(iso: iso, radius: denoiseRadius, shotCoeff: noiseShotCoeff, readCoeff: noiseReadCoeff, strength: denoiseStrength)
                enc.setBytes(&dParams, length: MemoryLayout<DenoiseParams>.stride, index: 0)
                dispatch(enc, width: bayerW, height: bayerH, state: denoisePipeline)
                enc.endEncoding()
            }

            // Pass 3a: Average chroma into half-res UV plane.
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(extractChromaPipeline)
                enc.setTexture(linearOut, index: 0)
                enc.setTexture(chromaRawOut, index: 1)
                dispatch(enc, width: chromaW, height: chromaH, state: extractChromaPipeline)
                enc.endEncoding()
            }

            // Pass 3b: Cross-bilateral chroma denoise.
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(denoiseChromaPipeline)
                enc.setTexture(chromaRawOut, index: 0)
                enc.setTexture(linearOut, index: 1)
                enc.setTexture(chromaDenoisedOut, index: 2)
                enc.setTexture(lumaStatsTex, index: 3)
                var dParams = DenoiseParams(iso: iso, radius: denoiseRadius, shotCoeff: noiseShotCoeff, readCoeff: noiseReadCoeff, strength: denoiseStrength)
                enc.setBytes(&dParams, length: MemoryLayout<DenoiseParams>.stride, index: 0)
                dispatch(enc, width: chromaW, height: chromaH, state: denoiseChromaPipeline)
                enc.endEncoding()
            }

            // Pass 4: Recombine full-res luma with upsampled half-res chroma.
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(recombineChromaPipeline)
                enc.setTexture(denoisedOut, index: 0)
                enc.setTexture(chromaDenoisedOut, index: 1)
                enc.setTexture(chromaMergedOut, index: 2)
                dispatch(enc, width: bayerW, height: bayerH, state: recombineChromaPipeline)
                enc.endEncoding()
            }

            // Store half-res chroma history for temporal chroma ring (record quality only).
            if processingQuality == .recordQuality, let chromaArr = chromaHistoryArray {
                if let enc = commandBuffer.makeComputeCommandEncoder() {
                    enc.setComputePipelineState(storeChromaHistoryPipeline)
                    enc.setTexture(chromaMergedOut, index: 0)
                    enc.setTexture(chromaArr, index: 1)
                    var scParams = StoreChromaParams(slice: Int32(temporalRingCursor))
                    enc.setBytes(&scParams, length: MemoryLayout<StoreChromaParams>.stride, index: 0)
                    dispatch(enc, width: chromaW, height: chromaH, state: storeChromaHistoryPipeline)
                    enc.endEncoding()
                }
            }

            // Pass 5: Temporal Denoise — N-slot ring buffer weighted average.
            let temporalOut = linearOut
            let doTemporal = temporalRingValidCount > 1
            if doTemporal {
                // Estimate global motion metric.
                if let enc = commandBuffer.makeComputeCommandEncoder() {
                    enc.setComputePipelineState(globalMotionPipeline)
                    enc.setTexture(chromaMergedOut, index: 0)
                    enc.setTexture(lumaHistoryArray, index: 1)
                    enc.setBuffer(motionMetricBuffer, offset: 0, index: 0)
                    var cursor = Int32(temporalRingCursor)
                    var slots = Int32(temporalRingCapacity)
                    enc.setBytes(&cursor, length: MemoryLayout<Int32>.stride, index: 1)
                    enc.setBytes(&slots, length: MemoryLayout<Int32>.stride, index: 2)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                    enc.endEncoding()
                }

                if let enc = commandBuffer.makeComputeCommandEncoder() {
                    enc.setComputePipelineState(temporalRingPipeline)
                    enc.setTexture(chromaMergedOut, index: 0)
                    enc.setTexture(temporalOut, index: 1)

                    let isoClamped = max(iso, 33.0)
                    let isoNorm = min(max((isoClamped - 33.0) / (1600.0 - 33.0), 0.0), 1.0)
                    let maxBlend: Float = 0.10 + 0.15 * isoNorm
                    var ringParams = RingTemporalParams(
                        iso: isoClamped,
                        maxBlend: maxBlend,
                        slotCount: Int32(temporalRingCapacity),
                        validSlots: Int32(temporalRingValidCount),
                        chromaW: Int32(chromaW),
                        chromaH: Int32(chromaH),
                        cursor: Int32(temporalRingCursor),
                        lambda: 0.7,
                        shotCoeff: noiseShotCoeff,
                        readCoeff: noiseReadCoeff
                    )
                    enc.setBytes(&ringParams, length: MemoryLayout<RingTemporalParams>.stride, index: 0)
                    enc.setBuffer(motionMetricBuffer, offset: 0, index: 1)

                    enc.setTexture(lumaHistoryArray, index: 2)
                    enc.setTexture(chromaHistoryArray, index: 3)

                    dispatch(enc, width: bayerW, height: bayerH, state: temporalRingPipeline)
                    enc.endEncoding()
                }
            } else if let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.copy(
                    from: chromaMergedOut,
                    sourceSlice: 0, sourceLevel: 0,
                    sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                    sourceSize: MTLSize(width: bayerW, height: bayerH, depth: 1),
                    to: temporalOut,
                    destinationSlice: 0, destinationLevel: 0,
                    destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
                )
                blit.endEncoding()
            }

            // Push luma history into ring buffer via compute kernel (RGB→Y conversion).
            if let lumaArr = lumaHistoryArray,
               let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(storeLumaHistoryPipeline)
                enc.setTexture(chromaMergedOut, index: 0)
                enc.setTexture(lumaArr, index: 1)
                var scParams = StoreChromaParams(slice: Int32(temporalRingCursor))
                enc.setBytes(&scParams, length: MemoryLayout<StoreChromaParams>.stride, index: 0)
                dispatch(enc, width: bayerW, height: bayerH, state: storeLumaHistoryPipeline)
                enc.endEncoding()
                temporalRingCursor = (temporalRingCursor + 1) % temporalRingCapacity
                if temporalRingValidCount < temporalRingCapacity {
                    temporalRingValidCount += 1
                }
            }

            // Pass 5.5: Adaptive Unsharp Masking
            var postDenoiseTex = temporalOut
            if sharpnessStrength > 0.001, let sharpenTex = getOrCreateSharpenTexture(width: bayerW, height: bayerH),
               let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(unsharpPipeline)
                enc.setTexture(temporalOut, index: 0)
                enc.setTexture(sharpenTex, index: 1)
                var s = sharpnessStrength
                enc.setBytes(&s, length: MemoryLayout<Float>.stride, index: 0)
                dispatch(enc, width: bayerW, height: bayerH, state: unsharpPipeline)
                enc.endEncoding()
                postDenoiseTex = sharpenTex
            }

            // Pass 6: Log OETF
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(logOnlyPipeline)
                enc.setTexture(postDenoiseTex, index: 0)
                enc.setTexture(fusedOut, index: 1)
                var logParams = LogOnlyParams(
                    curveType: Int32(curveType.rawValue),
                    headroomScale: curveType == .linear ? 1.0 : headroomScale
                )
                enc.setBytes(&logParams, length: MemoryLayout<LogOnlyParams>.stride, index: 0)
                dispatch(enc, width: bayerW, height: bayerH, state: logOnlyPipeline)
                enc.endEncoding()
            }
            outputTex = fusedOut
        }

        // BGRA pixel buffer for recording (AVAssetWriter expects 32BGRA)
        var bgraOut: CVPixelBuffer?
        var bgraTex: MTLTexture?
        if encodeAsBGRA {
            bgraOut = getOrCreatePixelBuffer(width: encodeWidth, height: encodeHeight)
            if let bgraPB = bgraOut, let texCache = textureCache {
                var cvTexOut: CVMetalTexture?
                CVMetalTextureCacheCreateTextureFromImage(nil, texCache, bgraPB, nil, .bgra8Unorm, encodeWidth, encodeHeight, 0, &cvTexOut)
                if let cvTex = cvTexOut {
                    bgraTex = CVMetalTextureGetTexture(cvTex)
                }
            }
        }

        // Final crop and scale directly into target texture (eliminating redundant 2nd Lanczos pass!)
        let finalTex = cropToAspectAndScale(
            outputTex,
            targetWidth: encodeWidth,
            targetHeight: encodeHeight,
            destinationTexture: bgraTex,
            cb: commandBuffer
        ) ?? bgraTex ?? outputTex

        let finalTexBox = SendableBox(value: finalTex)
        let bgraOutBox = SendableBox(value: bgraOut)
        let completionBox = SendableBox(value: completion)
        commandBuffer.addCompletedHandler { cb in
#if DEBUG
            let ms = (CACurrentMediaTime() - t0) * 1000.0
            print("[MetalPipeline] frame time: \(String(format: "%.2f", ms)) ms")
#endif
            if let error = cb.error {
                print("[MetalPipeline] ERROR: Command buffer failed: \(error.localizedDescription)")
            }
            completionBox.value(finalTexBox.value, bgraOutBox.value)
        }
        commandBuffer.commit()
    }

    // MARK: - Preview-Only (lightweight capture path)

    /// Fast preview path: bin → DPC → demosaic → log → crop/scale.
    /// Skips spatial/chroma/temporal denoise entirely — useful for viewfinder
    /// preview where full denoise is unnecessary overhead. Frame times ~2-5ms.
    func processPreviewOnly(_ pixelBuffer: CVPixelBuffer,
                            encodeWidth: Int = 1920,
                            encodeHeight: Int = 1440,
                            encodeAsBGRA: Bool = false,
                            completion: @escaping (MTLTexture?, CVPixelBuffer?) -> Void) {
        let t0 = CACurrentMediaTime()
        let fullW = CVPixelBufferGetWidth(pixelBuffer)
        let fullH = CVPixelBufferGetHeight(pixelBuffer)
        guard fullW > 0, fullH > 0 else { completion(nil, nil); return }

        guard let fullBayer = makeRawTexture(from: pixelBuffer) else {
            print("[MetalPipeline] Failed to create input texture")
            completion(nil, nil); return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { completion(nil, nil); return }

        // Phase-preserving half reduction.
        var bayerIn: MTLTexture
        let bayerW: Int
        let bayerH: Int
        let halfW = (fullW / 2) & ~1
        let halfH = (fullH / 2) & ~1
        let canReduceRaw = (halfW >= encodeWidth && halfH >= encodeHeight) || fullW > 3000
        if canReduceRaw {
            guard let halfTex = getOrCreateBinTexture(width: halfW, height: halfH),
                  let enc = commandBuffer.makeComputeCommandEncoder() else { completion(nil, nil); return }
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

        // Defect pixel correction (skip at 4K).
        let is4K = encodeWidth >= 3840
        if !is4K && bayerW <= 3000 {
            guard let correctedBayerPass = getOrCreateCorrectedBayerTexture(width: bayerW, height: bayerH),
                  let encDPC = commandBuffer.makeComputeCommandEncoder() else { completion(nil, nil); return }
            encDPC.setComputePipelineState(defectPixelPipeline)
            encDPC.setTexture(bayerIn, index: 0)
            encDPC.setTexture(correctedBayerPass, index: 1)
            var debayerParams = DebayerParams(
                bayerPattern: bayerPattern,
                blackLevel: blackLevel,
                whiteLevel: max(whiteLevel, blackLevel + 1e-6),
                lscCoefficients: lscCoefficients
            )
            var defectNoiseParams = DefectPixelParams(shotCoeff: noiseShotCoeff, readCoeff: noiseReadCoeff)
            encDPC.setBytes(&debayerParams, length: MemoryLayout<DebayerParams>.stride, index: 0)
            encDPC.setBytes(&defectNoiseParams, length: MemoryLayout<DefectPixelParams>.stride, index: 1)
            dispatch(encDPC, width: bayerW, height: bayerH, state: defectPixelPipeline)
            encDPC.endEncoding()
            bayerIn = correctedBayerPass
        }

        // Fused Demosaic + LSC + WB + CCM + Log OETF in ONE single pass
        guard let logOut = getOrCreateFusedTexture(width: bayerW, height: bayerH) else { completion(nil, nil); return }
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(debayerFusedPipeline)
            enc.setTexture(bayerIn, index: 0)
            enc.setTexture(logOut, index: 1)
            var params = FusedParams(
                bayerPattern: bayerPattern,
                blackLevel: blackLevel,
                whiteLevel: max(whiteLevel, blackLevel + 1e-6),
                curveType: Int32(curveType.rawValue),
                wbGains: wbParams.gains,
                lscCoefficients: lscCoefficients,
                greenBalance: greenBalance,
                headroomScale: curveType == .linear ? 1.0 : headroomScale
            )
            enc.setBytes(&params, length: MemoryLayout<FusedParams>.stride, index: 0)
            enc.setBytes(&lscParams, length: MemoryLayout<LSCParams>.stride, index: 1)
            var cMatrix = wbParams.colorMatrix
            enc.setBytes(&cMatrix, length: MemoryLayout<simd_float3x3>.stride, index: 2)
            dispatch(enc, width: bayerW, height: bayerH, state: debayerFusedPipeline)
            enc.endEncoding()
        }

        var postLogTex = logOut
        if sharpnessStrength > 0.001, let sharpenTex = getOrCreateSharpenTexture(width: bayerW, height: bayerH),
           let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(unsharpPipeline)
            enc.setTexture(logOut, index: 0)
            enc.setTexture(sharpenTex, index: 1)
            var s = sharpnessStrength
            enc.setBytes(&s, length: MemoryLayout<Float>.stride, index: 0)
            dispatch(enc, width: bayerW, height: bayerH, state: unsharpPipeline)
            enc.endEncoding()
            postLogTex = sharpenTex
        }

        // BGRA output for recording (if requested)
        var bgraOut: CVPixelBuffer?
        var bgraTex: MTLTexture?
        if encodeAsBGRA {
            bgraOut = getOrCreatePixelBuffer(width: encodeWidth, height: encodeHeight)
            if let bgraPB = bgraOut, let texCache = textureCache {
                var cvTexOut: CVMetalTexture?
                CVMetalTextureCacheCreateTextureFromImage(nil, texCache, bgraPB, nil, .bgra8Unorm, encodeWidth, encodeHeight, 0, &cvTexOut)
                if let cvTex = cvTexOut {
                    bgraTex = CVMetalTextureGetTexture(cvTex)
                }
            }
        }

        let finalTex = cropToAspectAndScale(
            postLogTex,
            targetWidth: encodeWidth,
            targetHeight: encodeHeight,
            destinationTexture: bgraTex,
            cb: commandBuffer
        ) ?? bgraTex ?? postLogTex

        let finalTexBox = SendableBox(value: finalTex)
        let bgraOutBox = SendableBox(value: bgraOut)
        let completionBox = SendableBox(value: completion)
        commandBuffer.addCompletedHandler { _ in
#if DEBUG
            let ms = (CACurrentMediaTime() - t0) * 1000.0
            print("[MetalPipeline] preview frame time: \(String(format: "%.2f", ms)) ms")
#endif
            completionBox.value(finalTexBox.value, bgraOutBox.value)
        }
        commandBuffer.commit()
    }

    func scale(_ texture: MTLTexture, width: Int, height: Int, destinationTexture: MTLTexture? = nil, cb: MTLCommandBuffer) -> MTLTexture? {
        guard width > 0, height > 0 else { return nil }

        if let dst = destinationTexture {
            if texture.width == width && texture.height == height {
                if dst.pixelFormat != texture.pixelFormat {
                    // Ultra-fast 1-tap format conversion (1 read, 1 write per pixel, <1ms)
                    if let enc = cb.makeComputeCommandEncoder() {
                        enc.setComputePipelineState(convertFormatPipeline)
                        enc.setTexture(texture, index: 0)
                        enc.setTexture(dst, index: 1)
                        dispatch(enc, width: width, height: height, state: convertFormatPipeline)
                        enc.endEncoding()
                    }
                } else if let blit = cb.makeBlitCommandEncoder() {
                    blit.copy(
                        from: texture,
                        sourceSlice: 0, sourceLevel: 0,
                        sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                        sourceSize: MTLSize(width: width, height: height, depth: 1),
                        to: dst,
                        destinationSlice: 0, destinationLevel: 0,
                        destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
                    )
                    blit.endEncoding()
                }
                return dst
            }
            // Direct 1-pass Lanczos scaling & format conversion directly into target texture!
            scaler.encode(commandBuffer: cb, sourceTexture: texture, destinationTexture: dst)
            return dst
        }

        if texture.width == width && texture.height == height { return texture }
        guard let output = getOrCreateScaleTexture(width: width, height: height) else { return nil }
        scaler.encode(commandBuffer: cb, sourceTexture: texture, destinationTexture: output)
        return output
    }

    func cropToAspectAndScale(_ texture: MTLTexture, targetWidth: Int, targetHeight: Int, destinationTexture: MTLTexture? = nil, cb: MTLCommandBuffer) -> MTLTexture? {
        guard targetWidth > 0, targetHeight > 0 else { return nil }
        let srcW = texture.width
        let srcH = texture.height
        guard srcW > 0, srcH > 0 else { return nil }
 
        let srcAspect = Float(srcW) / Float(srcH)
        let dstAspect = Float(targetWidth) / Float(targetHeight)

        // Early exit: aspect matches within epsilon → just scale (no crop needed).
        if abs(srcAspect - dstAspect) < 0.001 {
            return scale(texture, width: targetWidth, height: targetHeight, destinationTexture: destinationTexture, cb: cb)
        }

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
            return scale(texture, width: targetWidth, height: targetHeight, destinationTexture: destinationTexture, cb: cb)
        }
 
        if cropW == srcW && cropH == srcH && srcW == targetWidth && srcH == targetHeight {
            if let dst = destinationTexture {
                return scale(texture, width: targetWidth, height: targetHeight, destinationTexture: dst, cb: cb)
            }
            return texture
        }
 
        let sourceForScale: MTLTexture
        if cropW == srcW && cropH == srcH {
            sourceForScale = texture
        } else {
            guard let cropped = getOrCreateCropTexture(width: cropW, height: cropH),
                  let blit = cb.makeBlitCommandEncoder() else {
                return scale(texture, width: targetWidth, height: targetHeight, destinationTexture: destinationTexture, cb: cb)
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
 
        return scale(sourceForScale, width: targetWidth, height: targetHeight, destinationTexture: destinationTexture, cb: cb)
    }

    func makeScopeData(from texture: MTLTexture,
                       sampleWidth: Int = 96,
                       sampleHeight: Int = 54,
                       completion: @escaping (ScopeData?) -> Void) {
        let width = max(16, sampleWidth)
        let height = max(16, sampleHeight)
        guard let output = getOrCreateScopeTexture(width: width, height: height),
              let cb = scopeCommandQueue.makeCommandBuffer() else {
            completion(nil)
            return
        }

        scaler.encode(commandBuffer: cb, sourceTexture: texture, destinationTexture: output)
        let outputBox = SendableBox(value: output)
        let completionBox = SendableBox(value: completion)
        cb.addCompletedHandler { _ in
            let componentsPerPixel = 4
            let bytesPerComponent = MemoryLayout<UInt16>.stride
            let bytesPerRow = width * componentsPerPixel * bytesPerComponent
            let totalElements = width * height * componentsPerPixel
            let pixels = [UInt16](unsafeUninitializedCapacity: totalElements) { buffer, initializedCount in
                if let base = buffer.baseAddress {
                    outputBox.value.getBytes(
                        base,
                        bytesPerRow: bytesPerRow,
                        from: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0
                    )
                    initializedCount = totalElements
                } else {
                    initializedCount = 0
                }
            }
            completionBox.value(ScopeData.make(fromHalfRGBA: pixels, width: width, height: height))
        }
        cb.commit()
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
        
        let pbBox = SendableBox(value: pb)
        let completionBox = SendableBox(value: completion)
        cb.addCompletedHandler { _ in
            completionBox.value(pbBox.value)
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
        guard width > 0, height > 0 else { return nil }

        let texture: MTLTexture
        if let pooled = pooledRawTex, pooledRawW == width, pooledRawH == height {
            texture = pooled
        } else {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .r16Unorm, width: width, height: height, mipmapped: false)
            desc.usage = [.shaderRead]
            desc.storageMode = .shared
            guard let newTex = device.makeTexture(descriptor: desc) else { return nil }
            pooledRawTex = newTex
            pooledRawW = width
            pooledRawH = height
            texture = newTex
        }

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

    private func makeRGTexture(width: Int, height: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg16Float, width: width, height: height, mipmapped: false)
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

    private func makeTexture2DArray(width: Int, height: Int, arrayLength: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor()
        desc.textureType = .type2DArray
        desc.pixelFormat = .rgba16Float
        desc.width = width
        desc.height = height
        desc.depth = 1
        desc.arrayLength = arrayLength
        desc.mipmapLevelCount = 1
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
