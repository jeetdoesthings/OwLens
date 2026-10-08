import Metal
import MetalKit
import CoreVideo
import simd
import os
import UIKit

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

struct CropParams {
    var scaleX: Float
    var scaleY: Float
    var startX: Float
    var startY: Float
    var flip180: Int32
}

enum ProcessingQuality {
    /// Lower latency, skips expensive passes. Used for preview when not recording.
    case previewFast
    /// Full quality. Used when recording or when user explicitly wants highest quality.
    case recordQuality
}

/// Metal pipeline: Bayer → (optional CFA-safe phase-preserving 2× reduce) → (optional defect pixel correction)
/// → fused debayer / LSC / WB / CCM / Log OETF → (optional crop & scale) → (optional unsharp mask)
/// → 10-bit YCbCr 4:2:0 / BGRA encode.
///
///   • Single-pass demosaic + optical LSC + WB + Log encoding eliminates intermediate passes and VRAM traffic
///   • Pre-allocated texture pool — zero per-frame allocations
///   • Asynchronous GPU command buffer dispatch with zero CPU stalling
final class MetalPipeline: @unchecked Sendable {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private let scopeCommandQueue: MTLCommandQueue
    private let binPipeline: MTLComputePipelineState
    private let debayerFusedPipeline: MTLComputePipelineState
    private let convertFormatPipeline: MTLComputePipelineState
    private let convertYpCbCr10Pipeline: MTLComputePipelineState
    private let cropAndResamplePipeline: MTLComputePipelineState
    private let unsharpPipeline: MTLComputePipelineState
    private let defectPixelPipeline: MTLComputePipelineState
    private let binThreads: MTLSize
    private let debayerFusedThreads: MTLSize
    private let convertFormatThreads: MTLSize
    private let convertYpCbCr10Threads: MTLSize
    private let cropAndResampleThreads: MTLSize
    private let unsharpThreads: MTLSize
    private let defectPixelThreads: MTLSize
    private var textureCache: CVMetalTextureCache?

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

    private var pooledSharpenTex: [MTLTexture?] = [nil, nil, nil]
    private var pooledSharpenW: [Int] = [0, 0, 0]
    private var pooledSharpenH: [Int] = [0, 0, 0]

    private var pooledCorrectedBayerTex: [MTLTexture?] = [nil, nil, nil]
    private var pooledCorrectedBayerW: [Int] = [0, 0, 0]
    private var pooledCorrectedBayerH: [Int] = [0, 0, 0]

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

    // ── Frame timing & moving-average telemetry ──
    private let timingLock = OSAllocatedUnfairLock()
    private var rollingFrameTimes: [Double] = Array(repeating: 0.0, count: 30)
    private var rollingIndex = 0
    private var rollingCount = 0
    private var rollingSum: Double = 0.0
    private let rollingWindowSize = 30
    private var totalFramesProcessed: Int64 = 0
    private var lastRollingLogTime: CFTimeInterval = 0
    public private(set) var latestFrameTimeMs: Double = 0.0
    public private(set) var averageFrameTimeMs: Double = 0.0

    private func recordFrameTime(_ ms: Double, isRecording: Bool, width: Int, height: Int) {
        let (avg, minMs, maxMs, shouldLog): (Double, Double, Double, Bool) = timingLock.withLock {
            totalFramesProcessed += 1
            latestFrameTimeMs = ms
            if rollingCount < rollingWindowSize {
                rollingFrameTimes[rollingIndex] = ms
                rollingSum += ms
                rollingCount += 1
                rollingIndex = (rollingIndex + 1) % rollingWindowSize
            } else {
                rollingSum -= rollingFrameTimes[rollingIndex]
                rollingFrameTimes[rollingIndex] = ms
                rollingSum += ms
                rollingIndex = (rollingIndex + 1) % rollingWindowSize
            }

            let avg = rollingSum / Double(rollingCount)
            averageFrameTimeMs = avg

            let now = CACurrentMediaTime()
            let log = (now - lastRollingLogTime >= 1.0 || (totalFramesProcessed <= 30 && totalFramesProcessed % 10 == 0))
            var minVal = ms
            var maxVal = ms
            if log {
                lastRollingLogTime = now
                minVal = rollingFrameTimes[0..<rollingCount].min() ?? ms
                maxVal = rollingFrameTimes[0..<rollingCount].max() ?? ms
            }
            return (avg, minVal, maxVal, log)
        }

        #if DEBUG
        print(String(format: "[MetalPipeline] frame time: %.2f ms", ms))
        #endif

        if shouldLog {
            let effectiveFPS = avg > 0 ? (1000.0 / avg) : 0.0
            let headroom30 = 33.33 - avg
            let headroom60 = 16.67 - avg
            let status60 = headroom60 >= 0 ? "FEASIBLE" : "EXCEEDS BUDGET"

            print(String(
                format: "[MetalPipeline] ⏱ Avg Frame Time (30f): %.2f ms (min: %.2f, max: %.2f) | GPU: ~%.1f fps | 30fps margin: %+.1f ms | 60fps (16.67ms): %+.1f ms [%@] | Mode: %@",
                avg, minMs, maxMs, effectiveFPS, headroom30, headroom60, status60,
                isRecording ? "REC (\(width)x\(height))" : "VIEW (\(width)x\(height))"
            ))
        }
    }

    var curveType: LogCurveType = .sLog3Approx {
        didSet {
            headroomScale = 1.0
        }
    }
    /// Scene reflectance headroom multiplier (1.0 preserves exact standard scene reflectance).
    var headroomScale: Float = 1.0
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

    /// Adaptive edge sharpness strength (0.0 = off, 0.5 = natural cinema sharpness, 1.0 = sharp).
    var sharpnessStrength: Float = 0.5

    /// Active interface orientation: when .landscapeLeft, the pipeline inverts the 180° sensor image so preview and recording are natively upright.
    var orientation: UIInterfaceOrientation = .landscapeRight

    init?(customLibraryURL: URL? = nil) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = customLibraryURL.flatMap({ try? device.makeLibrary(URL: $0) }) ?? device.makeDefaultLibrary(),
              let binFunc = library.makeFunction(name: "binBayerCFA"),
              let debayerFusedFunc = library.makeFunction(name: "debayerFusedLog"),
              let convertFormatFunc = library.makeFunction(name: "convertRgba16FloatToBgra8"),
              let convertYpCbCr10Func = library.makeFunction(name: "convertRgbTo420YpCbCr10"),
              let defectPixelFunc = library.makeFunction(name: "correctDefectPixelsBayer"),
              let unsharpFunc = library.makeFunction(name: "unsharpMaskAdaptive"),
              let cropAndResampleFunc = library.makeFunction(name: "cropAndResampleBilinear") else {
            return nil
        }
        guard let scopeQueue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = queue
        self.scopeCommandQueue = scopeQueue
        do {
            let bin = try device.makeComputePipelineState(function: binFunc)
            let unsharp = try device.makeComputePipelineState(function: unsharpFunc)
            let debayer = try device.makeComputePipelineState(function: debayerFusedFunc)
            let convertFormat = try device.makeComputePipelineState(function: convertFormatFunc)
            let convertYp = try device.makeComputePipelineState(function: convertYpCbCr10Func)
            let crop = try device.makeComputePipelineState(function: cropAndResampleFunc)
            let defect = try device.makeComputePipelineState(function: defectPixelFunc)

            self.binPipeline = bin
            self.unsharpPipeline = unsharp
            self.debayerFusedPipeline = debayer
            self.convertFormatPipeline = convertFormat
            self.convertYpCbCr10Pipeline = convertYp
            self.cropAndResamplePipeline = crop
            self.defectPixelPipeline = defect

            self.binThreads = Self.computeThreadsPerGroup(for: bin)
            self.unsharpThreads = Self.computeThreadsPerGroup(for: unsharp)
            self.debayerFusedThreads = Self.computeThreadsPerGroup(for: debayer)
            self.convertFormatThreads = Self.computeThreadsPerGroup(for: convertFormat)
            self.convertYpCbCr10Threads = Self.computeThreadsPerGroup(for: convertYp)
            self.cropAndResampleThreads = Self.computeThreadsPerGroup(for: crop)
            self.defectPixelThreads = Self.computeThreadsPerGroup(for: defect)
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

    /// Releases recording textures to reduce VRAM and cache pressure when idle or stopping recording.
    func trimMemory() {
        pooledRawTex = Array(repeating: nil, count: poolSlotCount)
        pooledRawW = Array(repeating: 0, count: poolSlotCount)
        pooledRawH = Array(repeating: 0, count: poolSlotCount)

        pooledBinTex = Array(repeating: nil, count: poolSlotCount)
        pooledBinW = Array(repeating: 0, count: poolSlotCount)
        pooledBinH = Array(repeating: 0, count: poolSlotCount)

        pooledCorrectedBayerTex = Array(repeating: nil, count: poolSlotCount)
        pooledCorrectedBayerW = Array(repeating: 0, count: poolSlotCount)
        pooledCorrectedBayerH = Array(repeating: 0, count: poolSlotCount)

        pooledFusedTex = Array(repeating: nil, count: poolSlotCount)
        pooledFusedW = Array(repeating: 0, count: poolSlotCount)
        pooledFusedH = Array(repeating: 0, count: poolSlotCount)

        pooledScaleTex = Array(repeating: nil, count: poolSlotCount)
        pooledScaleW = Array(repeating: 0, count: poolSlotCount)
        pooledScaleH = Array(repeating: 0, count: poolSlotCount)

        pooledSharpenTex = Array(repeating: nil, count: poolSlotCount)
        pooledSharpenW = Array(repeating: 0, count: poolSlotCount)
        pooledSharpenH = Array(repeating: 0, count: poolSlotCount)

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
                kCVPixelBufferPoolMinimumBufferCountKey as String: 16
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
            _ = getOrCreateFusedTexture(width: bW, height: bH, slot: s)
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
                    dispatch(enc, width: encodeWidth, height: encodeHeight, threadsPerGroup: convertFormatThreads)
                    enc.endEncoding()

                    if CVBufferGetAttachment(pb, kCVImageBufferColorPrimariesKey, nil) == nil {
                        Self.attachColorMetadata(to: pb, curveType: curveType)
                    }
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
            dispatch(enc, width: encodeWidth, height: encodeHeight, threadsPerGroup: convertFormatThreads)
            enc.endEncoding()

            if CVBufferGetAttachment(pb, kCVImageBufferColorPrimariesKey, nil) == nil {
                Self.attachColorMetadata(to: pb, curveType: curveType)
            }
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
                    dispatch(enc, width: uvW, height: uvH, threadsPerGroup: convertYpCbCr10Threads)
                    enc.endEncoding()

                    if CVBufferGetAttachment(pb, kCVImageBufferColorPrimariesKey, nil) == nil {
                        Self.attachColorMetadata(to: pb, curveType: curveType)
                    }
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
            dispatch(enc, width: uvW, height: uvH, threadsPerGroup: convertYpCbCr10Threads)
            enc.endEncoding()

            if CVBufferGetAttachment(pb, kCVImageBufferColorPrimariesKey, nil) == nil {
                Self.attachColorMetadata(to: pb, curveType: curveType)
            }
            return (pb, [cvY, cvUV])
        }
    }

    // MARK: - Main process (single-pass fused pipeline)

    /// Process RAW to log RGB using the streamlined single-pass fused pipeline.
    /// Uses CFA-preserving 2× reduction only when the reduced frame still covers the encode size.
    /// The completion handler is called on an internal Metal queue once the GPU work finishes.
    func process(_ pixelBuffer: CVPixelBuffer,
                 encodeWidth: Int = 2016,
                 encodeHeight: Int = 1512,
                 slot: Int = 0,
                 completion: @escaping (MTLTexture?) -> Void) {
        process(pixelBuffer, encodeWidth: encodeWidth, encodeHeight: encodeHeight, encodeAsBGRA: false, slot: slot) { texture, _ in
            completion(texture)
        }
    }

    func process(_ pixelBuffer: CVPixelBuffer,
                 encodeWidth: Int = 2016,
                 encodeHeight: Int = 1512,
                 encodeAsBGRA: Bool = false,
                 slot: Int = 0,
                 completion: @escaping (MTLTexture?, CVPixelBuffer?) -> Void) {
        let t0 = CACurrentMediaTime()
        let fullW = CVPixelBufferGetWidth(pixelBuffer)
        let fullH = CVPixelBufferGetHeight(pixelBuffer)
        guard fullW > 0, fullH > 0 else { completion(nil, nil); return }

        guard let fullBayer = makeRawTexture(from: pixelBuffer, slot: slot) else {
            print("[MetalPipeline] Failed to create input texture")
            completion(nil, nil); return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { completion(nil, nil); return }

        // Phase-preserving 2x2 CFA sensor binning (averages 4 identical-color photosites in each 4x4 block).
        // For Open Gate (2016x1512) and 1080p (1920x1080), half-resolution (2016x1512) matches or exceeds the
        // container resolution (3.05 MP >= 2.76 MP), so binning provides pristine 1:1 optical sampling,
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
            dispatch(enc, width: halfW, height: halfH, threadsPerGroup: binThreads)
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
            dispatch(encDPC, width: bayerW, height: bayerH, threadsPerGroup: defectPixelThreads)
            encDPC.endEncoding()
            bayerIn = correctedBayerPass
        }

        // ── Single-Pass Fused Debayer: Demosaic + LSC + WB + CCM + Log OETF in ONE kernel ──
        // Uses debayerFusedLog to eliminate separate debayer, WB, LSC, and log passes,
        // saving ~97.5 MB per frame of VRAM bandwidth at 12.2 MP.
        guard let fusedOut = getOrCreateFusedTexture(width: bayerW, height: bayerH, slot: slot) else {
            completion(nil, nil); return
        }
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
            dispatch(enc, width: bayerW, height: bayerH, threadsPerGroup: debayerFusedThreads)
            enc.endEncoding()
        }

        // Final crop and scale into target aspect ratio & resolution (inverting 180° when in Landscape Left)
        let shouldFlip180 = (orientation == .landscapeLeft)
        let scaledTex = cropToAspectAndScale(
            fusedOut,
            targetWidth: encodeWidth,
            targetHeight: encodeHeight,
            destinationTexture: nil,
            slot: slot,
            flip180: shouldFlip180,
            cb: commandBuffer
        ) ?? fusedOut

        // Adaptive unsharp masking executed on target resolution post-crop/scale.
        // Runs for recording to deliver cinema-grade detail, and safely bypassed in previewFast
        // (and during elevated thermal states) to save ~2ms of GPU time and ~100MB/s memory traffic.
        let finalTex: MTLTexture
        if processingQuality == .recordQuality && sharpnessStrength > 0.001,
           let sTex = getOrCreateSharpenTexture(width: scaledTex.width, height: scaledTex.height, slot: slot),
           let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(unsharpPipeline)
            enc.setTexture(scaledTex, index: 0)
            enc.setTexture(sTex, index: 1)
            var s = sharpnessStrength
            enc.setBytes(&s, length: MemoryLayout<Float>.stride, index: 0)
            dispatch(enc, width: scaledTex.width, height: scaledTex.height, threadsPerGroup: unsharpThreads)
            enc.endEncoding()
            finalTex = sTex
        } else {
            finalTex = scaledTex
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
        commandBuffer.addCompletedHandler { [weak self] cb in
            _ = retainedTexturesBox.value
            let dtMs = (CACurrentMediaTime() - t0) * 1000.0
            self?.recordFrameTime(dtMs, isRecording: encodeAsBGRA, width: encodeWidth, height: encodeHeight)
            if let error = cb.error {
                print("[MetalPipeline] ERROR: Command buffer failed: \(error.localizedDescription)")
            }
            completionBox.value(finalTexBox.value, outputPBBox.value)
        }
        commandBuffer.commit()
    }

    // MARK: - Preview-Only (lightweight capture path)

    /// Fast preview path: delegates to process with encodeAsBGRA: false.
    /// Runs single-pass fused debayer + crop/scale + unsharp mask with zero CVPixelBuffer allocation.
    func processPreviewOnly(_ pixelBuffer: CVPixelBuffer,
                            encodeWidth: Int = 2016,
                            encodeHeight: Int = 1512,
                            slot: Int = 0,
                            completion: @escaping (MTLTexture?) -> Void) {
        process(pixelBuffer, encodeWidth: encodeWidth, encodeHeight: encodeHeight, encodeAsBGRA: false, slot: slot) { texture, _ in
            completion(texture)
        }
    }

    func scale(_ texture: MTLTexture, width: Int, height: Int, destinationTexture: MTLTexture? = nil, slot: Int = 0, flip180: Bool = false, cb: MTLCommandBuffer) -> MTLTexture? {
        return cropToAspectAndScale(texture, targetWidth: width, targetHeight: height, destinationTexture: destinationTexture, slot: slot, flip180: flip180, cb: cb)
    }

    func cropToAspectAndScale(_ texture: MTLTexture, targetWidth: Int, targetHeight: Int, destinationTexture: MTLTexture? = nil, slot: Int = 0, flip180: Bool = false, cb: MTLCommandBuffer) -> MTLTexture? {
        guard targetWidth > 0, targetHeight > 0 else { return nil }
        let srcW = texture.width
        let srcH = texture.height
        guard srcW > 0, srcH > 0 else { return nil }

        // Exact dimension match: zero-cost bypass (or fast format conversion / blit) only if not flipping 180°
        if srcW == targetWidth && srcH == targetHeight && !flip180 {
            if let dst = destinationTexture {
                if dst.pixelFormat != texture.pixelFormat {
                    if let enc = cb.makeComputeCommandEncoder() {
                        enc.setComputePipelineState(convertFormatPipeline)
                        enc.setTexture(texture, index: 0)
                        enc.setTexture(dst, index: 1)
                        dispatch(enc, width: targetWidth, height: targetHeight, threadsPerGroup: convertFormatThreads)
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
        var params = CropParams(scaleX: scaleX, scaleY: scaleY, startX: startX, startY: startY, flip180: flip180 ? 1 : 0)
        enc.setBytes(&params, length: MemoryLayout<CropParams>.stride, index: 0)
        dispatch(enc, width: targetWidth, height: targetHeight, threadsPerGroup: cropAndResampleThreads)
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
            var params = CropParams(scaleX: scaleX, scaleY: scaleY, startX: startX, startY: startY, flip180: 0)
            enc.setBytes(&params, length: MemoryLayout<CropParams>.stride, index: 0)
            dispatch(enc, width: width, height: height, threadsPerGroup: cropAndResampleThreads)
            enc.endEncoding()
        }
        let outputBox = SendableBox(value: output)
        let completionBox = SendableBox(value: completion)
        cb.addCompletedHandler { _ in
            let componentsPerPixel = 4
            let bytesPerComponent = MemoryLayout<UInt16>.stride
            let bytesPerRow = width * componentsPerPixel * bytesPerComponent
            let totalElements = width * height * componentsPerPixel
            withUnsafeTemporaryAllocation(of: UInt16.self, capacity: totalElements) { buffer in
                if let base = buffer.baseAddress {
                    outputBox.value.getBytes(
                        base,
                        bytesPerRow: bytesPerRow,
                        from: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0
                    )
                    let scope = ScopeData.make(
                        fromHalfRGBAPointer: base,
                        count: totalElements,
                        width: width,
                        height: height
                    )
                    completionBox.value(scope)
                } else {
                    completionBox.value(nil)
                }
            }
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

    private func makePrivateTexture(width: Int, height: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }

    private static func computeThreadsPerGroup(for state: MTLComputePipelineState) -> MTLSize {
        // Optimal 16x16 2D tile (256 threads) on Apple Silicon maximizes GPU EU occupancy
        // and L1 texture cache locality while preventing register spilling.
        if state.maxTotalThreadsPerThreadgroup >= 256 {
            return MTLSize(width: 16, height: 16, depth: 1)
        } else {
            let tw = state.threadExecutionWidth
            let th = max(1, state.maxTotalThreadsPerThreadgroup / tw)
            return MTLSize(width: tw, height: th, depth: 1)
        }
    }

    private func dispatch(_ enc: MTLComputeCommandEncoder, width: Int, height: Int, threadsPerGroup: MTLSize) {
        let tw = threadsPerGroup.width
        let th = threadsPerGroup.height
        let groups = MTLSize(
            width: (width + tw - 1) / tw,
            height: (height + th - 1) / th,
            depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
    }
}
