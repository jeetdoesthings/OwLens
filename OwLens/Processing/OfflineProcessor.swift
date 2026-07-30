import Foundation
import AVFoundation
import CoreVideo
import VideoToolbox

/// Processes a recorded .rawlite file through the full-quality Metal pipeline
/// with no real-time deadline, producing the final HEVC output.
///
/// Temporal ring buffer is preserved across frames for multi-frame denoising.
/// If frame index gaps are detected (dropped capture), the temporal history
/// is cleared to prevent blending across non-adjacent frames.
final class OfflineProcessor {
    private let metalPipeline: MetalPipeline
    private let processingQueue = DispatchQueue(label: "offline.processing.queue", qos: .default)

    var onProgress: ((Int, Int) -> Void)?
    var onComplete: ((URL) -> Void)?
    var onError: ((String) -> Void)?

    init(metalPipeline: MetalPipeline) {
        self.metalPipeline = metalPipeline
    }

    func process(rawLogURL: URL, audioURL: URL?, outputURL: URL,
                 encodeWidth: Int, encodeHeight: Int, targetFPS: Double) {
        processingQueue.async { [weak self] in
            self?.processInternal(rawLogURL: rawLogURL, audioURL: audioURL,
                                  outputURL: outputURL,
                                  encodeWidth: encodeWidth, encodeHeight: encodeHeight,
                                  targetFPS: targetFPS)
        }
    }

    private func processInternal(rawLogURL: URL, audioURL: URL?, outputURL: URL,
                                  encodeWidth: Int, encodeHeight: Int, targetFPS: Double) {
        guard let header = FrameLogReader.readHeader(from: rawLogURL) else {
            DispatchQueue.main.async { self.onError?("Invalid raw log") }
            return
        }
        let totalFrames = header.frameCount
        guard totalFrames > 0 else {
            DispatchQueue.main.async { self.onError?("No frames") }
            return
        }

        let fps = (abs(targetFPS - 30) < 0.5) ? 30.0 : 24.0
        print("[OfflineProcessor] \(totalFrames) frames @ \(Int(fps))fps binned \(header.width)x\(header.height)")

        guard let fh = try? FileHandle(forReadingFrom: rawLogURL) else {
            DispatchQueue.main.async { self.onError?("Cannot open raw log") }
            return
        }
        defer { fh.closeFile() }
        fh.seek(toFileOffset: 64)
        let frameSize = 128 + header.bytesPerRow * header.height

        // ── AVAssetWriter setup ──
        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mov) else {
            DispatchQueue.main.async { self.onError?("Cannot create writer") }
            return
        }

        let compression: [String: Any] = [
            AVVideoAverageBitRateKey: 100_000_000,
            kVTCompressionPropertyKey_ProfileLevel as String: kVTProfileLevel_HEVC_Main_AutoLevel,
            AVVideoExpectedSourceFrameRateKey: Int(fps),
            AVVideoMaxKeyFrameIntervalKey: Int(fps),
            AVVideoAllowFrameReorderingKey: false as NSNumber
        ]
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: encodeWidth,
            AVVideoHeightKey: encodeHeight,
            AVVideoCompressionPropertiesKey: compression,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
            ]
        ]
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        videoInput.expectsMediaDataInRealTime = false
        videoInput.mediaTimeScale = CMTimeScale(fps * 1000)
        guard writer.canAdd(videoInput) else { DispatchQueue.main.async { self.onError?("Cannot add video") }; return }
        writer.add(videoInput)

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: encodeWidth,
                kCVPixelBufferHeightKey as String: encodeHeight,
                kCVPixelBufferMetalCompatibilityKey as String: true
            ])

        // Optional audio from preview recording
        var audioInput: AVAssetWriterInput?
        var audioReader: AVAssetReader?
        var audioTrackOutput: AVAssetReaderTrackOutput?
        if let audioURL, FileManager.default.fileExists(atPath: audioURL.path) {
            let asset = AVAsset(url: audioURL)
            if let t = asset.tracks(withMediaType: .audio).first {
                let ai = AVAssetWriterInput(mediaType: .audio, outputSettings: nil)
                ai.expectsMediaDataInRealTime = false
                if writer.canAdd(ai) { writer.add(ai); audioInput = ai }
                if let reader = try? AVAssetReader(asset: asset) {
                    let o = AVAssetReaderTrackOutput(track: t, outputSettings: nil)
                    if reader.canAdd(o) { reader.add(o); audioTrackOutput = o; audioReader = reader }
                }
            }
        }

        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        audioReader?.startReading()

        // Pool for binned pixel buffers
        let poolAttrs = [kCVPixelBufferPixelFormatTypeKey: header.pixelFormat as OSType,
                         kCVPixelBufferWidthKey: header.width,
                         kCVPixelBufferHeightKey: header.height,
                         kCVPixelBufferMetalCompatibilityKey: true] as [String: Any]
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, nil, poolAttrs as CFDictionary, &pool)

        let sem = DispatchSemaphore(value: 0)
        var lastFrameIndex: UInt32 = UInt32.max
        var processedCount = 0
        let progressLock = NSLock()
        var lastReported = 0

        let report = { (cur: Int) in
            progressLock.lock()
            if cur - lastReported >= max(1, totalFrames / 100) { lastReported = cur }
            else { progressLock.unlock(); return }
            progressLock.unlock()
            DispatchQueue.main.async { self.onProgress?(cur, totalFrames) }
        }

        for i in 0..<totalFrames {
            autoreleasepool {
                let fd = fh.readData(ofLength: frameSize)
                guard fd.count == frameSize else { print("[OfflineProcessor] Truncated frame \(i)"); sem.signal(); return }

                let meta = FrameLogWriter.FrameMetadata(
                    blackLevel: fd.readF32(4), whiteLevel: fd.readF32(8),
                    iso: fd.readF32(12), exposureDuration: fd.readF64(16),
                    cfaPattern: Int32(bitPattern: fd.readU32(24)),
                    wbGains: SIMD3<Float>(fd.readF32(28), fd.readF32(32), fd.readF32(36)),
                    lscCoefficients: SIMD4<Float>(fd.readF32(40), fd.readF32(44), fd.readF32(48), fd.readF32(52)),
                    timestamp: fd.readF64(56))

                let frameIndex = fd.readU32(0)

                // ── Temporal gap detection ──
                // If frames were dropped mid-recording, the index jumps.
                // Clear temporal history to avoid blending non-adjacent frames.
                if lastFrameIndex != UInt32.max && frameIndex != lastFrameIndex + 1 {
                    print("[OfflineProcessor] Frame gap at \(i): index \(lastFrameIndex) → \(frameIndex) — clearing temporal")
                    metalPipeline.clearTemporalHistory()
                }
                lastFrameIndex = frameIndex

                // Create CVPixelBuffer from binned raw data
                let pixelData = fd.subdata(in: 128..<frameSize)
                var pb: CVPixelBuffer?
                if let pool { CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb) }
                if pb == nil {
                    CVPixelBufferCreate(nil, header.width, header.height,
                                       header.pixelFormat, poolAttrs as CFDictionary, &pb)
                }
                guard let pixelBuffer = pb else { sem.signal(); return }

                CVPixelBufferLockBaseAddress(pixelBuffer, [])
                if let base = CVPixelBufferGetBaseAddress(pixelBuffer) {
                    let bpr = CVPixelBufferGetBytesPerRow(pixelBuffer)
                    pixelData.withUnsafeBytes { src in
                        guard let s = src.baseAddress else { return }
                        if bpr == header.bytesPerRow {
                            memcpy(base, s, pixelData.count)
                        } else {
                            for y in 0..<header.height {
                                memcpy(base.advanced(by: y * bpr), s.advanced(by: y * header.bytesPerRow), min(bpr, header.bytesPerRow))
                            }
                        }
                    }
                }
                CVPixelBufferUnlockBaseAddress(pixelBuffer, [])

                // Configure pipeline
                metalPipeline.bayerPattern = meta.cfaPattern
                metalPipeline.blackLevel = meta.blackLevel
                metalPipeline.whiteLevel = meta.whiteLevel
                metalPipeline.iso = meta.iso
                metalPipeline.lscParams = simd4ToLSCParams(meta.lscCoefficients)
                metalPipeline.greenBalance = 1.0
                metalPipeline.wbParams = WhiteBalanceParams(gains: meta.wbGains, colorMatrix: matrix_identity_float3x3)
                metalPipeline.processingQuality = .recordQuality

                // Process through full pipeline (spatial + chroma + temporal + log)
                metalPipeline.process(pixelBuffer, encodeWidth: encodeWidth, encodeHeight: encodeHeight,
                                     encodeAsBGRA: true) { _, bgra in
                    defer { sem.signal() }
                    guard let bgra else { return }

                    while !videoInput.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.001) }
                    let pts = CMTime(value: CMTimeValue(processedCount * 1000), timescale: CMTimeScale(fps * 1000.0))
                    adaptor.append(bgra, withPresentationTime: pts)
                    processedCount += 1
                    report(processedCount)

                    // Drain audio
                    if let ai = audioInput, let ao = audioTrackOutput, let ar = audioReader, ar.status == .reading {
                        while ai.isReadyForMoreMediaData, let s = ao.copyNextSampleBuffer() { ai.append(s) }
                    }
                }
                sem.wait()
            }
        }

        videoInput.markAsFinished()
        if let ai = audioInput, let ao = audioTrackOutput, let ar = audioReader {
            while ar.status == .reading, let s = ao.copyNextSampleBuffer() {
                while !ai.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.001) }
                ai.append(s)
            }
            ai.markAsFinished()
        }

        let fs = DispatchSemaphore(value: 0)
        writer.finishWriting { fs.signal() }
        fs.wait()

        if writer.status == .completed {
            print("[OfflineProcessor] Completed: \(outputURL.lastPathComponent)")
            try? FileManager.default.removeItem(at: rawLogURL)
            if let a = audioURL, FileManager.default.fileExists(atPath: a.path) {
                try? FileManager.default.removeItem(at: a)
            }
            DispatchQueue.main.async { self.onComplete?(outputURL) }
        } else {
            print("[OfflineProcessor] Failed: \(String(describing: writer.error))")
            DispatchQueue.main.async { self.onError?(writer.error?.localizedDescription ?? "Unknown") }
        }
    }

    private func simd4ToLSCParams(_ c: SIMD4<Float>) -> LSCParams {
        LSCParams(radialR: c[0], radialG: (c[1]+c[2])*0.5, radialB: c[3],
                  radial4R: 0, radial4G: 0, radial4B: 0, azimuthR: 0, azimuthG: 0, azimuthB: 0)
    }
}
