import Metal
import MetalKit
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
struct CropParams {
    var scaleX: Float
    var scaleY: Float
    var startX: Float
    var startY: Float
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
    private let convertYpCbCr10Pipeline: MTLComputePipelineState
    private let cropAndResamplePipeline: MTLComputePipelineState
    private let unsharpPipeline: MTLComputePipelineState
    private let lumaStatsPipeline: MTLComputePipelineState
    private let defectPixelPipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?

    // Global motion metric buffers (1 float per slot) used to skip temporal blending during motion.
    private let motionMetricBuffers: [MTLBuffer]

    // ── Texture pool (avoids per-frame allocation, triple-buffered for concurrent in-flight frames) ──
    private let poolSlotCount = 3
    private var pooledRawTex: [MTLTexture?] = [nil, nil, nil]
    private var pooledRawW: [Int] = [0, 0, 0]
    private var pooledRawH: [Int] = [0, 0, 0]
    private var pooledBinTex: [MTLTexture?] = [nil, nil, nil]
    private var pooledBinW: [Int] = [0, 0, 0]
    private var pooledBinH: [Int] = [0, 0, 0]
    private var pooledFusedTex: [MTLTexture?] = [nil, nil, nil]
    private var pooledFusedW: [Int] = [0, 0, 0]
    private var pooledFusedH: [Int] = [0, 0, 0]
    
    private var pooledLinearTex: [MTLTexture?] = [nil, nil, nil]
    private var pooledLinearW: [Int] = [0, 0, 0]
    private var pooledLinearH: [Int] = [0, 0, 0]
    
    private var pooledDenoisedTex: [MTLTexture?] = [nil, nil, nil]
    private var pooledDenoisedW: [Int] = [0, 0, 0]
    private var pooledDenoisedH: [Int] = [0, 0, 0]

    private var pooledSharpenTex: [MTLTexture?] = [nil, nil, nil]
    private var pooledSharpenW: [Int] = [0, 0, 0]
    private var pooledSharpenH: [Int] = [0, 0, 0]

    private var pooledChromaRawTex: [MTLTexture?] = [nil, nil, nil]
    private var pooledChromaRawW: [Int] = [0, 0, 0]
    private var pooledChromaRawH: [Int] = [0, 0, 0]
    private var pooledChromaDenoisedTex: [MTLTexture?] = [nil, nil, nil]
    private var pooledChromaDenoisedW: [Int] = [0, 0, 0]
    private var pooledChromaDenoisedH: [Int] = [0, 0, 0]
    private var pooledChromaMergedTex: [MTLTexture?] = [nil, nil, nil]
    private var pooledChromaMergedW: [Int] = [0, 0, 0]
    private var pooledChromaMergedH: [Int] = [0, 0, 0]

    private var pooledCorrectedBayerTex: [MTLTexture?] = [nil, nil, nil]
    private var pooledCorrectedBayerW: [Int] = [0, 0, 0]
    private var pooledCorrectedBayerH: [Int] = [0, 0, 0]

    private var pooledLumaStatsTex: [MTLTexture?] = [nil, nil, nil]
    private var pooledLumaStatsW: [Int] = [0, 0, 0]
    private var pooledLumaStatsH: [Int] = [0, 0, 0]

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
    private lazy var dummyStatsTex: MTLTexture? = {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Float, width: 1, height: 1, mipmapped: false)
        desc.usage = [.shaderRead]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }()

    private var pooledScaleTex: [MTLTexture?] = [nil, nil, nil]
    private var pooledScaleW: [Int] = [0, 0, 0]
    private var pooledScaleH: [Int] = [0, 0, 0]
    private var pooledScopeTex: MTLTexture?
    private var pooledScopeW: Int = 0
    private var pooledScopeH: Int = 0
    
    private var pixelBufferPool: CVPixelBufferPool?
    private var pixelBufferPoolW: Int = 0
    private var pixelBufferPoolH: Int = 0
    private var pixelBufferPoolFormat: OSType = 0

    var curveType: LogCurveType = .appleLog2 {
        didSet {
            headroomScale = LogCurve.defaultRMax(for: curveType)
        }
    }
    /// Scene reflectance headroom multiplier (1.0 for linear, 12.0 for Apple Log 2, 10.0 for S-Log3).
    var headroomScale: Float = LogCurve.defaultRMax(for: .appleLog2)
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
              let convertYpCbCr10Func = library.makeFunction(name: "convertRgbTo420YpCbCr10"),
              let lumaStatsFunc = library.makeFunction(name: "estimateLumaVariance"),
              let defectPixelFunc = library.makeFunction(name: "correctDefectPixelsBayer"),
              let storeLumaHistoryFunc = library.makeFunction(name: "storeLumaHistory"),
              let unsharpFunc = library.makeFunction(name: "unsharpMaskAdaptive"),
              let cropAndResampleFunc = library.makeFunction(name: "cropAndResampleBilinear") else {
            return nil
        }
        var mBuffers: [MTLBuffer] = []
        for _ in 0..<3 {
            guard let b = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared) else { return nil }
            mBuffers.append(b)
        }
        guard let scopeQueue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = queue
        self.scopeCommandQueue = scopeQueue
        self.motionMetricBuffers = mBuffers
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
            self.convertYpCbCr10Pipeline = try device.makeComputePipelineState(function: convertYpCbCr10Func)
            self.cropAndResamplePipeline = try device.makeComputePipelineState(function: cropAndResampleFunc)
            self.lumaStatsPipeline = try device.makeComputePipelineState(function: lumaStatsFunc)
            self.defectPixelPipeline = try device.makeComputePipelineState(function: defectPixelFunc)
            self.storeLumaHistoryPipeline = try device.makeComputePipelineState(function: storeLumaHistoryFunc)
        } catch {
            print("[MetalPipeline] Failed to create compute pipelines: \(error)")
            return nil
        }
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
    }

    // MARK: - Texture pool helpers

    private func getOrCreateBinTexture(width: Int, height: Int, slot: Int = 0) -> MTLTexture? {
        let i = slot % poolSlotCount
        if let tex = pooledBinTex[i], pooledBinW[i] == width, pooledBinH[i] == height {
            return tex
        }
        let tex = makeR16Texture(width: width, height: height)
        pooledBinTex[i] = tex
        pooledBinW[i] = width
        pooledBinH[i] = height
        return tex
    }

    private func getOrCreateFusedTexture(width: Int, height: Int, slot: Int = 0) -> MTLTexture? {
        let i = slot % poolSlotCount
        if let tex = pooledFusedTex[i], pooledFusedW[i] == width, pooledFusedH[i] == height {
            return tex
        }
        let tex = makePrivateTexture(width: width, height: height)
        pooledFusedTex[i] = tex
        pooledFusedW[i] = width
        pooledFusedH[i] = height
        return tex
    }

    private func getOrCreateLinearTexture(width: Int, height: Int, slot: Int = 0) -> MTLTexture? {
        let i = slot % poolSlotCount
        if let tex = pooledLinearTex[i], pooledLinearW[i] == width, pooledLinearH[i] == height {
            return tex
        }
        let tex = makePrivateTexture(width: width, height: height)
        pooledLinearTex[i] = tex
        pooledLinearW[i] = width
        pooledLinearH[i] = height
        return tex
    }

    private func getOrCreateDenoisedTexture(width: Int, height: Int, slot: Int = 0) -> MTLTexture? {
        let i = slot % poolSlotCount
        if let tex = pooledDenoisedTex[i], pooledDenoisedW[i] == width, pooledDenoisedH[i] == height {
            return tex
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderWrite, .shaderRead]
        desc.storageMode = .private
        
        guard let tex = device.makeTexture(descriptor: desc) else { return nil }
        pooledDenoisedTex[i] = tex
        pooledDenoisedW[i] = width
        pooledDenoisedH[i] = height
        return tex
    }

    private func getOrCreateSharpenTexture(width: Int, height: Int, slot: Int = 0) -> MTLTexture? {
        let i = slot % poolSlotCount
        if let tex = pooledSharpenTex[i], pooledSharpenW[i] == width, pooledSharpenH[i] == height {
            return tex
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderWrite, .shaderRead]
        desc.storageMode = .private
        guard let tex = device.makeTexture(descriptor: desc) else { return nil }
        pooledSharpenTex[i] = tex
        pooledSharpenW[i] = width
        pooledSharpenH[i] = height
        return tex
    }

    private func getOrCreateChromaRawTexture(width: Int, height: Int, slot: Int = 0) -> MTLTexture? {
        let i = slot % poolSlotCount
        if let tex = pooledChromaRawTex[i], pooledChromaRawW[i] == width, pooledChromaRawH[i] == height {
            return tex
        }
        let tex = makeRGTexture(width: width, height: height)
        pooledChromaRawTex[i] = tex
        pooledChromaRawW[i] = width
        pooledChromaRawH[i] = height
        return tex
    }

    private func getOrCreateChromaDenoisedTexture(width: Int, height: Int, slot: Int = 0) -> MTLTexture? {
        let i = slot % poolSlotCount
        if let tex = pooledChromaDenoisedTex[i], pooledChromaDenoisedW[i] == width, pooledChromaDenoisedH[i] == height {
            return tex
        }
        let tex = makeRGTexture(width: width, height: height)
        pooledChromaDenoisedTex[i] = tex
        pooledChromaDenoisedW[i] = width
        pooledChromaDenoisedH[i] = height
        return tex
    }

    private func getOrCreateChromaMergedTexture(width: Int, height: Int, slot: Int = 0) -> MTLTexture? {
        let i = slot % poolSlotCount
        if let tex = pooledChromaMergedTex[i], pooledChromaMergedW[i] == width, pooledChromaMergedH[i] == height {
            return tex
        }
        let tex = makePrivateTexture(width: width, height: height)
        pooledChromaMergedTex[i] = tex
        pooledChromaMergedW[i] = width
        pooledChromaMergedH[i] = height
        return tex
    }

    private func getOrCreateCorrectedBayerTexture(width: Int, height: Int, slot: Int = 0) -> MTLTexture? {
        let i = slot % poolSlotCount
        if let tex = pooledCorrectedBayerTex[i], pooledCorrectedBayerW[i] == width, pooledCorrectedBayerH[i] == height { return tex }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        let tex = device.makeTexture(descriptor: desc)
        pooledCorrectedBayerTex[i] = tex
        pooledCorrectedBayerW[i] = width
        pooledCorrectedBayerH[i] = height
        return tex
    }

    private func getOrCreateLumaStatsTexture(width: Int, height: Int, slot: Int = 0) -> MTLTexture? {
        let i = slot % poolSlotCount
        if let tex = pooledLumaStatsTex[i], pooledLumaStatsW[i] == width, pooledLumaStatsH[i] == height { return tex }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg16Float,
            width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        let tex = device.makeTexture(descriptor: desc)
        pooledLumaStatsTex[i] = tex
        pooledLumaStatsW[i] = width
        pooledLumaStatsH[i] = height
        return tex
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

        pooledRawTex = Array(repeating: nil, count: poolSlotCount)
        pooledRawW = Array(repeating: 0, count: poolSlotCount)
        pooledRawH = Array(repeating: 0, count: poolSlotCount)

        pooledBinTex = Array(repeating: nil, count: poolSlotCount)
        pooledBinW = Array(repeating: 0, count: poolSlotCount)
        pooledBinH = Array(repeating: 0, count: poolSlotCount)

        pooledFusedTex = Array(repeating: nil, count: poolSlotCount)
        pooledFusedW = Array(repeating: 0, count: poolSlotCount)
        pooledFusedH = Array(repeating: 0, count: poolSlotCount)

        pooledLinearTex = Array(repeating: nil, count: poolSlotCount)
        pooledLinearW = Array(repeating: 0, count: poolSlotCount)
        pooledLinearH = Array(repeating: 0, count: poolSlotCount)

        pooledDenoisedTex = Array(repeating: nil, count: poolSlotCount)
        pooledDenoisedW = Array(repeating: 0, count: poolSlotCount)
        pooledDenoisedH = Array(repeating: 0, count: poolSlotCount)

        pooledSharpenTex = Array(repeating: nil, count: poolSlotCount)
        pooledSharpenW = Array(repeating: 0, count: poolSlotCount)
        pooledSharpenH = Array(repeating: 0, count: poolSlotCount)

        pooledChromaRawTex = Array(repeating: nil, count: poolSlotCount)
        pooledChromaRawW = Array(repeating: 0, count: poolSlotCount)
        pooledChromaRawH = Array(repeating: 0, count: poolSlotCount)

        pooledChromaDenoisedTex = Array(repeating: nil, count: poolSlotCount)
        pooledChromaDenoisedW = Array(repeating: 0, count: poolSlotCount)
        pooledChromaDenoisedH = Array(repeating: 0, count: poolSlotCount)

        pooledChromaMergedTex = Array(repeating: nil, count: poolSlotCount)
        pooledChromaMergedW = Array(repeating: 0, count: poolSlotCount)
        pooledChromaMergedH = Array(repeating: 0, count: poolSlotCount)

        pooledCorrectedBayerTex = Array(repeating: nil, count: poolSlotCount)
        pooledCorrectedBayerW = Array(repeating: 0, count: poolSlotCount)
        pooledCorrectedBayerH = Array(repeating: 0, count: poolSlotCount)

        pooledLumaStatsTex = Array(repeating: nil, count: poolSlotCount)
        pooledLumaStatsW = Array(repeating: 0, count: poolSlotCount)
        pooledLumaStatsH = Array(repeating: 0, count: poolSlotCount)

        pooledScaleTex = Array(repeating: nil, count: poolSlotCount)
        pooledScaleW = Array(repeating: 0, count: poolSlotCount)
        pooledScaleH = Array(repeating: 0, count: poolSlotCount)

        pooledScopeTex = nil
        pooledScopeW = 0
        pooledScopeH = 0

        pixelBufferPool = nil
        pixelBufferPoolW = 0
        pixelBufferPoolH = 0
        pixelBufferPoolFormat = 0

        if let cache = textureCache {
            CVMetalTextureCacheFlush(cache, 0)
        }
    }

    private func getOrCreateScaleTexture(width: Int, height: Int, slot: Int = 0) -> MTLTexture? {
        let i = slot % poolSlotCount
        if let tex = pooledScaleTex[i], pooledScaleW[i] == width, pooledScaleH[i] == height {
            return tex
        }
        let tex = makePrivateTexture(width: width, height: height)
        pooledScaleTex[i] = tex
        pooledScaleW[i] = width
        pooledScaleH[i] = height
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

    private func getOrCreatePixelBuffer(width: Int, height: Int, format: OSType = kCVPixelFormatType_32BGRA) -> CVPixelBuffer? {
        if pixelBufferPool == nil || pixelBufferPoolW != width || pixelBufferPoolH != height || pixelBufferPoolFormat != format {
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: format,
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
            pixelBufferPoolFormat = format
        }

        guard let pixelBufferPool else { return nil }
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(nil, pixelBufferPool, &pixelBuffer)
        guard status == kCVReturnSuccess else { return nil }
        return pixelBuffer
    }

    /// Pre-allocates the CVPixelBufferPool and Metal texture pools ahead of recording so Frame 0 doesn't stall allocating on the fly.
    func prewarm(width: Int, height: Int, curveType: LogCurveType) {
        let format: OSType = (curveType == .linear)
            ? kCVPixelFormatType_32BGRA
            : kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        _ = getOrCreatePixelBuffer(width: width, height: height, format: format)
        let bW = pooledBinW[0] > 0 ? pooledBinW[0] : 2016
        let bH = pooledBinH[0] > 0 ? pooledBinH[0] : 1512
        for s in 0..<poolSlotCount {
            _ = getOrCreateBinTexture(width: bW, height: bH, slot: s)
            _ = getOrCreateCorrectedBayerTexture(width: bW, height: bH, slot: s)
            _ = getOrCreateLinearTexture(width: bW, height: bH, slot: s)
            _ = getOrCreateFusedTexture(width: width, height: height, slot: s)
            _ = getOrCreateScaleTexture(width: width, height: height, slot: s)
            _ = getOrCreateSharpenTexture(width: width, height: height, slot: s)
        }
    }

    /// Attaches standard NCLC color primaries, matrix, and transfer function metadata to pixel buffers.
    static func attachColorMetadata(to pixelBuffer: CVPixelBuffer, curveType: LogCurveType) {
        switch curveType {
        case .linear:
            CVBufferSetAttachment(pixelBuffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
            CVBufferSetAttachment(pixelBuffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
            CVBufferSetAttachment(pixelBuffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        case .appleLog2:
            CVBufferSetAttachment(pixelBuffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_2020, .shouldPropagate)
            CVBufferSetAttachment(pixelBuffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_2020, .shouldPropagate)
            CVBufferRemoveAttachment(pixelBuffer, kCVImageBufferTransferFunctionKey)
            if #available(iOS 17.2, *) {
                CVBufferSetAttachment(pixelBuffer, kCVImageBufferLogTransferFunctionKey, kCVImageBufferLogTransferFunction_AppleLog, .shouldPropagate)
            } else {
                CVBufferSetAttachment(pixelBuffer, "LogTransferFunction" as CFString, "com.apple.rec2020.apple-log" as CFString, .shouldPropagate)
            }
        case .sLog3Approx:
            CVBufferSetAttachment(pixelBuffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_2020, .shouldPropagate)
            CVBufferSetAttachment(pixelBuffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_2020, .shouldPropagate)
            CVBufferRemoveAttachment(pixelBuffer, kCVImageBufferTransferFunctionKey)
        }
    }

    /// Encodes an RGB texture into a recording CVPixelBuffer.
    /// For Linear: encodes to 8-bit BGRA (`kCVPixelFormatType_32BGRA`).
    /// For Apple Log 2 / S-Log3: encodes to true 10-bit Video Range BT.2020 YCbCr 4:2:0
    /// (`kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange`) via direct Metal kernel dispatch.
    /// Returns the pixel buffer along with CVMetalTextures that must be retained until GPU completion.
    private func encodeOutputPixelBuffer(
        from sourceTexture: MTLTexture,
        encodeWidth: Int,
        encodeHeight: Int,
        cb: MTLCommandBuffer
    ) -> (CVPixelBuffer?, [Any]) {
        if curveType == .linear {
            guard let pb = getOrCreatePixelBuffer(width: encodeWidth, height: encodeHeight, format: kCVPixelFormatType_32BGRA) else {
                return (nil, [])
            }
            if let unmanaged = CVPixelBufferGetIOSurface(pb) {
                let ioSurface = unmanaged.takeUnretainedValue()
                let desc = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .bgra8Unorm, width: encodeWidth, height: encodeHeight, mipmapped: false)
                desc.usage = [.shaderWrite]
                desc.storageMode = .shared
                if let bgraTex = device.makeTexture(descriptor: desc, iosurface: ioSurface, plane: 0),
                   let enc = cb.makeComputeCommandEncoder() {
                    enc.setComputePipelineState(convertFormatPipeline)
                    enc.setTexture(sourceTexture, index: 0)
                    enc.setTexture(bgraTex, index: 1)
                    dispatch(enc, width: encodeWidth, height: encodeHeight, state: convertFormatPipeline)
                    enc.endEncoding()

                    Self.attachColorMetadata(to: pb, curveType: curveType)
                    return (pb, [])
                }
            }

            guard let texCache = textureCache else { return (nil, []) }
            var cvTexOut: CVMetalTexture?
            let status = CVMetalTextureCacheCreateTextureFromImage(
                nil, texCache, pb, nil, .bgra8Unorm, encodeWidth, encodeHeight, 0, &cvTexOut)
            guard status == kCVReturnSuccess, let cvTex = cvTexOut,
                  let bgraTex = CVMetalTextureGetTexture(cvTex),
                  let enc = cb.makeComputeCommandEncoder() else {
                return (nil, [])
            }
            enc.setComputePipelineState(convertFormatPipeline)
            enc.setTexture(sourceTexture, index: 0)
            enc.setTexture(bgraTex, index: 1)
            dispatch(enc, width: encodeWidth, height: encodeHeight, state: convertFormatPipeline)
            enc.endEncoding()

            Self.attachColorMetadata(to: pb, curveType: curveType)
            return (pb, [cvTex])
        } else {
            guard let pb = getOrCreatePixelBuffer(width: encodeWidth, height: encodeHeight, format: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange) else {
                return (nil, [])
            }
            let yW = CVPixelBufferGetWidthOfPlane(pb, 0)
            let yH = CVPixelBufferGetHeightOfPlane(pb, 0)
            let uvW = CVPixelBufferGetWidthOfPlane(pb, 1)
            let uvH = CVPixelBufferGetHeightOfPlane(pb, 1)

            if let unmanaged = CVPixelBufferGetIOSurface(pb) {
                let ioSurface = unmanaged.takeUnretainedValue()
                let descY = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .r16Unorm, width: yW, height: yH, mipmapped: false)
                descY.usage = [.shaderWrite]
                descY.storageMode = .shared

                let descUV = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .rg16Unorm, width: uvW, height: uvH, mipmapped: false)
                descUV.usage = [.shaderWrite]
                descUV.storageMode = .shared

                if let yTex = device.makeTexture(descriptor: descY, iosurface: ioSurface, plane: 0),
                   let uvTex = device.makeTexture(descriptor: descUV, iosurface: ioSurface, plane: 1),
                   let enc = cb.makeComputeCommandEncoder() {
                    enc.setComputePipelineState(convertYpCbCr10Pipeline)
                    enc.setTexture(sourceTexture, index: 0)
                    enc.setTexture(yTex, index: 1)
                    enc.setTexture(uvTex, index: 2)
                    dispatch(enc, width: uvW, height: uvH, state: convertYpCbCr10Pipeline)
                    enc.endEncoding()

                    Self.attachColorMetadata(to: pb, curveType: curveType)
                    return (pb, [])
                }
            }

            guard let texCache = textureCache else { return (nil, []) }
            var cvYOut: CVMetalTexture?
            let statusY = CVMetalTextureCacheCreateTextureFromImage(
                nil, texCache, pb, nil, .r16Unorm, yW, yH, 0, &cvYOut)

            var cvUVOut: CVMetalTexture?
            let statusUV = CVMetalTextureCacheCreateTextureFromImage(
                nil, texCache, pb, nil, .rg16Unorm, uvW, uvH, 1, &cvUVOut)

            guard statusY == kCVReturnSuccess, statusUV == kCVReturnSuccess,
                  let cvY = cvYOut, let yTex = CVMetalTextureGetTexture(cvY),
                  let cvUV = cvUVOut, let uvTex = CVMetalTextureGetTexture(cvUV),
                  let enc = cb.makeComputeCommandEncoder() else {
                return (nil, [])
            }

            enc.setComputePipelineState(convertYpCbCr10Pipeline)
            enc.setTexture(sourceTexture, index: 0)
            enc.setTexture(yTex, index: 1)
            enc.setTexture(uvTex, index: 2)
            dispatch(enc, width: uvW, height: uvH, state: convertYpCbCr10Pipeline)
            enc.endEncoding()

            Self.attachColorMetadata(to: pb, curveType: curveType)
            return (pb, [cvY, cvUV])
        }
    }

    // MARK: - Main process (linear denoise path)

    /// Process RAW to log RGB using the linear denoise path.
    /// Uses CFA-preserving 2× reduction only when the reduced frame still covers the encode size.
    /// The completion handler is called on an internal Metal queue once the GPU work finishes.
    func process(_ pixelBuffer: CVPixelBuffer,
                 encodeWidth: Int = 1920,
                 encodeHeight: Int = 1440,
                 slot: Int = 0,
                 completion: @escaping (MTLTexture?) -> Void) {
        process(pixelBuffer, encodeWidth: encodeWidth, encodeHeight: encodeHeight, encodeAsBGRA: false, slot: slot) { texture, _ in
            completion(texture)
        }
    }

    func process(_ pixelBuffer: CVPixelBuffer,
                 encodeWidth: Int = 1920,
                 encodeHeight: Int = 1440,
                 encodeAsBGRA: Bool = false,
                 slot: Int = 0,
                 completion: @escaping (MTLTexture?, CVPixelBuffer?) -> Void) {
#if DEBUG
        let t0 = CACurrentMediaTime()
#endif
        let fullW = CVPixelBufferGetWidth(pixelBuffer)
        let fullH = CVPixelBufferGetHeight(pixelBuffer)
        guard fullW > 0, fullH > 0 else { completion(nil, nil); return }

        guard let fullBayer = makeRawTexture(from: pixelBuffer, slot: slot) else {
            print("[MetalPipeline] Failed to create input texture")
            completion(nil, nil); return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { completion(nil, nil); return }

        // Phase-preserving 2x2 CFA sensor binning (averages 4 identical-color photosites in each 4x4 block).
        // For Open Gate (1920x1440) and 1080p (1920x1080), half-resolution (2016x1512) exceeds the
        // container resolution (3.05 MP > 2.76 MP), so binning provides pristine 1:1 optical sampling,
        // boosts SNR by +6 dB (halving noise variance in Log shadows), eliminates Bayer moiré,
        // and maintains smooth real-time 30 fps playback.
        var bayerIn: MTLTexture
        let bayerW: Int
        let bayerH: Int
        let halfW = (fullW / 2) & ~1
        let halfH = (fullH / 2) & ~1
        let canReduceRaw = (halfW >= encodeWidth && halfH >= encodeHeight)
        if canReduceRaw {
            guard let halfTex = getOrCreateBinTexture(width: halfW, height: halfH, slot: slot),
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

        // Defect pixel correction: skipped when CFA binning is active because 2x2 averaging
        // naturally cancels isolated hot/stuck pixels without an extra GPU round-trip.
        if !canReduceRaw && bayerW <= 3000 {
            // Defect pixel correction operates on (possibly binned) Bayer data.
            guard let correctedBayerPass = getOrCreateCorrectedBayerTexture(width: bayerW, height: bayerH, slot: slot),
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

        let postLinearTex: MTLTexture
        var logAlreadyApplied = false

        if !runDenoise {
            // ── Fused Path: Demosaic + LSC + WB + CCM + Log OETF in ONE kernel ──
            // Uses debayerFusedLog to eliminate the separate applyLogOnly pass entirely,
            // saving one full-resolution rgba16Float read+write round-trip (~97.5 MB at 12.2 MP).
            // The fused kernel contains identical math to debayerWBLinear + applyLogOnly.
            guard let fusedOut = getOrCreateLinearTexture(width: bayerW, height: bayerH, slot: slot) else { completion(nil, nil); return }
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
            logAlreadyApplied = true
            postLinearTex = fusedOut
        } else {
            // ── Full Multi-Pass Denoise Path ──
            let chromaW = max(1, (bayerW + 1) / 2)
            let chromaH = max(1, (bayerH + 1) / 2)
            guard let linearOut = getOrCreateLinearTexture(width: bayerW, height: bayerH, slot: slot),
                  let chromaMergedOut = getOrCreateChromaMergedTexture(width: bayerW, height: bayerH, slot: slot),
                  let denoisedOut = getOrCreateDenoisedTexture(width: bayerW, height: bayerH, slot: slot),
                  let chromaRawOut = getOrCreateChromaRawTexture(width: chromaW, height: chromaH, slot: slot),
                  let chromaDenoisedOut = getOrCreateChromaDenoisedTexture(width: chromaW, height: chromaH, slot: slot) else { completion(nil, nil); return }

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

            // Pass 1.5: Per-pixel local-sigma guide. Skipped in previewFast or during thermal throttle.
            let useLocalSigma = processingQuality == .recordQuality && bayerW < 3000 && !isThermalThrottled
            let lumaStatsTex: MTLTexture? = useLocalSigma
                ? getOrCreateLumaStatsTexture(width: bayerW, height: bayerH, slot: slot)
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
            let motionMetricBuffer = motionMetricBuffers[slot % poolSlotCount]
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

            postLinearTex = temporalOut
        }

        // Final crop and scale into target aspect ratio & resolution
        let scaledTex = cropToAspectAndScale(
            postLinearTex,
            targetWidth: encodeWidth,
            targetHeight: encodeHeight,
            destinationTexture: nil,
            slot: slot,
            cb: commandBuffer
        ) ?? postLinearTex

        // Adaptive unsharp masking executed on target resolution post-crop/scale.
        // Avoids reading ~878 MB / writing ~97.5 MB per frame at 12.2 MP, reducing GPU latency
        // by up to 83% at 1080p, and prevents downscaling filter from blurring sharpened details.
        let sharpenedTex: MTLTexture
        if sharpnessStrength > 0.001, let sTex = getOrCreateSharpenTexture(width: scaledTex.width, height: scaledTex.height, slot: slot),
           let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(unsharpPipeline)
            enc.setTexture(scaledTex, index: 0)
            enc.setTexture(sTex, index: 1)
            var s = sharpnessStrength
            enc.setBytes(&s, length: MemoryLayout<Float>.stride, index: 0)
            dispatch(enc, width: scaledTex.width, height: scaledTex.height, state: unsharpPipeline)
            enc.endEncoding()
            sharpenedTex = sTex
        } else {
            sharpenedTex = scaledTex
        }

        let finalTex: MTLTexture
        if logAlreadyApplied {
            // Fused kernel already applied the Log OETF — skip the separate pass entirely.
            // This saves one full 12.2 MP rgba16Float read+write round-trip (~97.5 MB bandwidth).
            finalTex = sharpenedTex
        } else {
            // Apply Log OETF (with highlight shoulder) to the scaled scene-linear frame
            let outW = sharpenedTex.width
            let outH = sharpenedTex.height
            guard let finalLogTex = getOrCreateFusedTexture(width: outW, height: outH, slot: slot) else { completion(nil, nil); return }
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(logOnlyPipeline)
                enc.setTexture(sharpenedTex, index: 0)
                enc.setTexture(finalLogTex, index: 1)
                var logParams = LogOnlyParams(
                    curveType: Int32(curveType.rawValue),
                    headroomScale: curveType == .linear ? 1.0 : headroomScale
                )
                enc.setBytes(&logParams, length: MemoryLayout<LogOnlyParams>.stride, index: 0)
                dispatch(enc, width: outW, height: outH, state: logOnlyPipeline)
                enc.endEncoding()
            }
            finalTex = finalLogTex
        }

        var outputPB: CVPixelBuffer?
        var retainedCVTextures: [Any] = []
        if encodeAsBGRA {
            let encoded = encodeOutputPixelBuffer(
                from: finalTex,
                encodeWidth: encodeWidth,
                encodeHeight: encodeHeight,
                cb: commandBuffer
            )
            outputPB = encoded.0
            retainedCVTextures = encoded.1
        }

        let finalTexBox = SendableBox(value: finalTex)
        let outputPBBox = SendableBox(value: outputPB)
        let retainedTexturesBox = SendableBox(value: retainedCVTextures)
        let completionBox = SendableBox(value: completion)
        commandBuffer.addCompletedHandler { cb in
            _ = retainedTexturesBox.value
#if DEBUG
            let ms = (CACurrentMediaTime() - t0) * 1000.0
            print("[MetalPipeline] frame time: \(String(format: "%.2f", ms)) ms")
#endif
            if let error = cb.error {
                print("[MetalPipeline] ERROR: Command buffer failed: \(error.localizedDescription)")
            }
            completionBox.value(finalTexBox.value, outputPBBox.value)
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
                            slot: Int = 0,
                            completion: @escaping (MTLTexture?) -> Void) {
        let t0 = CACurrentMediaTime()
        let fullW = CVPixelBufferGetWidth(pixelBuffer)
        let fullH = CVPixelBufferGetHeight(pixelBuffer)
        guard fullW > 0, fullH > 0 else { completion(nil); return }

        guard let fullBayer = makeRawTexture(from: pixelBuffer, slot: slot) else {
            print("[MetalPipeline] Failed to create input texture")
            completion(nil); return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { completion(nil); return }

        // Phase-preserving half reduction.
        var bayerIn: MTLTexture
        let bayerW: Int
        let bayerH: Int
        let halfW = (fullW / 2) & ~1
        let halfH = (fullH / 2) & ~1
        let canReduceRaw = (halfW >= encodeWidth && halfH >= encodeHeight) || fullW > 3000
        if canReduceRaw {
            guard let halfTex = getOrCreateBinTexture(width: halfW, height: halfH, slot: slot),
                  let enc = commandBuffer.makeComputeCommandEncoder() else { completion(nil); return }
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

        // Defect pixel correction (skip on full-res sensor where hot pixels are averaged during downsampling).
        if bayerW <= 3000 {
            guard let correctedBayerPass = getOrCreateCorrectedBayerTexture(width: bayerW, height: bayerH, slot: slot),
                  let encDPC = commandBuffer.makeComputeCommandEncoder() else { completion(nil); return }
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

        // ── Fused Path: Demosaic + LSC + WB + CCM + Log OETF in ONE kernel ──
        guard let fusedOut = getOrCreateLinearTexture(width: bayerW, height: bayerH, slot: slot) else { completion(nil); return }
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

        // Final crop and scale into target aspect ratio & resolution
        let scaledTex = cropToAspectAndScale(
            fusedOut,
            targetWidth: encodeWidth,
            targetHeight: encodeHeight,
            destinationTexture: nil,
            slot: slot,
            cb: commandBuffer
        ) ?? fusedOut

        let finalTex: MTLTexture
        if sharpnessStrength > 0.001, let sharpenTex = getOrCreateSharpenTexture(width: scaledTex.width, height: scaledTex.height, slot: slot),
           let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(unsharpPipeline)
            enc.setTexture(scaledTex, index: 0)
            enc.setTexture(sharpenTex, index: 1)
            var s = sharpnessStrength
            enc.setBytes(&s, length: MemoryLayout<Float>.stride, index: 0)
            dispatch(enc, width: scaledTex.width, height: scaledTex.height, state: unsharpPipeline)
            enc.endEncoding()
            finalTex = sharpenTex
        } else {
            finalTex = scaledTex
        }

        let finalTexBox = SendableBox(value: finalTex)
        let completionBox = SendableBox(value: completion)
        commandBuffer.addCompletedHandler { _ in
#if DEBUG
            let ms = (CACurrentMediaTime() - t0) * 1000.0
            print("[MetalPipeline] preview frame time: \(String(format: "%.2f", ms)) ms")
#endif
            completionBox.value(finalTexBox.value)
        }
        commandBuffer.commit()
    }

    func scale(_ texture: MTLTexture, width: Int, height: Int, destinationTexture: MTLTexture? = nil, slot: Int = 0, cb: MTLCommandBuffer) -> MTLTexture? {
        return cropToAspectAndScale(texture, targetWidth: width, targetHeight: height, destinationTexture: destinationTexture, slot: slot, cb: cb)
    }

    func cropToAspectAndScale(_ texture: MTLTexture, targetWidth: Int, targetHeight: Int, destinationTexture: MTLTexture? = nil, slot: Int = 0, cb: MTLCommandBuffer) -> MTLTexture? {
        guard targetWidth > 0, targetHeight > 0 else { return nil }
        let srcW = texture.width
        let srcH = texture.height
        guard srcW > 0, srcH > 0 else { return nil }

        // Exact dimension match: zero-cost bypass (or fast format conversion / blit)
        if srcW == targetWidth && srcH == targetHeight {
            if let dst = destinationTexture {
                if dst.pixelFormat != texture.pixelFormat {
                    if let enc = cb.makeComputeCommandEncoder() {
                        enc.setComputePipelineState(convertFormatPipeline)
                        enc.setTexture(texture, index: 0)
                        enc.setTexture(dst, index: 1)
                        dispatch(enc, width: targetWidth, height: targetHeight, state: convertFormatPipeline)
                        enc.endEncoding()
                    }
                } else if let blit = cb.makeBlitCommandEncoder() {
                    blit.copy(
                        from: texture,
                        sourceSlice: 0, sourceLevel: 0,
                        sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                        sourceSize: MTLSize(width: targetWidth, height: targetHeight, depth: 1),
                        to: dst,
                        destinationSlice: 0, destinationLevel: 0,
                        destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
                    )
                    blit.endEncoding()
                }
                return dst
            }
            return texture
        }

        let srcAspect = Float(srcW) / Float(srcH)
        let dstAspect = Float(targetWidth) / Float(targetHeight)

        var cropW = Float(srcW)
        var cropH = Float(srcH)
        var originX: Float = 0
        var originY: Float = 0

        if srcAspect > dstAspect + 0.001 {
            cropW = (Float(srcH) * dstAspect).rounded(.toNearestOrEven)
            originX = max(0, (Float(srcW) - cropW) * 0.5)
        } else if srcAspect < dstAspect - 0.001 {
            cropH = (Float(srcW) / dstAspect).rounded(.toNearestOrEven)
            originY = max(0, (Float(srcH) - cropH) * 0.5)
        }

        // Direct 1-pass crop & hardware-filtered resample directly into target texture!
        // Eliminates intermediate crop blit copies (~146 MB memory bandwidth) and heavy
        // multi-tap Lanczos sinc separable convolutions, cutting frame time by ~20ms.
        let dst = destinationTexture ?? getOrCreateScaleTexture(width: targetWidth, height: targetHeight, slot: slot)
        guard let dst, let enc = cb.makeComputeCommandEncoder() else { return nil }

        let scaleX = cropW / Float(targetWidth)
        let scaleY = cropH / Float(targetHeight)
        let startX = originX + 0.5 * scaleX
        let startY = originY + 0.5 * scaleY

        enc.setComputePipelineState(cropAndResamplePipeline)
        enc.setTexture(texture, index: 0)
        enc.setTexture(dst, index: 1)
        var params = CropParams(scaleX: scaleX, scaleY: scaleY, startX: startX, startY: startY)
        enc.setBytes(&params, length: MemoryLayout<CropParams>.stride, index: 0)
        dispatch(enc, width: targetWidth, height: targetHeight, state: cropAndResamplePipeline)
        enc.endEncoding()

        return dst
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

        let scaleX = Float(texture.width) / Float(width)
        let scaleY = Float(texture.height) / Float(height)
        let startX = 0.5 * scaleX
        let startY = 0.5 * scaleY

        if let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(cropAndResamplePipeline)
            enc.setTexture(texture, index: 0)
            enc.setTexture(output, index: 1)
            var params = CropParams(scaleX: scaleX, scaleY: scaleY, startX: startX, startY: startY)
            enc.setBytes(&params, length: MemoryLayout<CropParams>.stride, index: 0)
            dispatch(enc, width: width, height: height, state: cropAndResamplePipeline)
            enc.endEncoding()
        }
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

    // MARK: - Texture helpers

    private func makeRawTexture(from pixelBuffer: CVPixelBuffer, slot: Int = 0) -> MTLTexture? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        // 1. Direct zero-copy IOSurface binding (instantaneous, bypasses CVMetalTextureCache overhead)
        if let unmanaged = CVPixelBufferGetIOSurface(pixelBuffer) {
            let ioSurface = unmanaged.takeUnretainedValue()
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .r16Unorm,
                width: width,
                height: height,
                mipmapped: false
            )
            desc.usage = [.shaderRead]
            desc.storageMode = .shared
            if let tex = device.makeTexture(descriptor: desc, iosurface: ioSurface, plane: 0) {
                return tex
            }
        }

        // 2. CVMetalTextureCache fallback
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
        return copyBayerToTexture(pixelBuffer, slot: slot)
    }

    private func copyBayerToTexture(_ pixelBuffer: CVPixelBuffer, slot: Int = 0) -> MTLTexture? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width > 0, height > 0 else { return nil }

        let i = slot % poolSlotCount
        let texture: MTLTexture
        if let pooled = pooledRawTex[i], pooledRawW[i] == width, pooledRawH[i] == height {
            texture = pooled
        } else {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .r16Unorm, width: width, height: height, mipmapped: false)
            desc.usage = [.shaderRead]
            desc.storageMode = .shared
            guard let newTex = device.makeTexture(descriptor: desc) else { return nil }
            pooledRawTex[i] = newTex
            pooledRawW[i] = width
            pooledRawH[i] = height
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
        // Optimal 16x16 2D tile (256 threads) on Apple Silicon maximizes GPU EU occupancy
        // and L1 texture cache locality while preventing register spilling.
        let tw: Int
        let th: Int
        if state.maxTotalThreadsPerThreadgroup >= 256 {
            tw = 16
            th = 16
        } else {
            tw = state.threadExecutionWidth
            th = max(1, state.maxTotalThreadsPerThreadgroup / tw)
        }
        let threadsPerGroup = MTLSize(width: tw, height: th, depth: 1)
        let groups = MTLSize(
            width: (width + tw - 1) / tw,
            height: (height + th - 1) / th,
            depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
    }
}
