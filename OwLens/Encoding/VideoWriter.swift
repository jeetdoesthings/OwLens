import AVFoundation
import CoreVideo
import VideoToolbox
import QuartzCore
import UIKit


/// AVAssetWriter — HEVC + optional AAC at a constant 24 fps.
///
/// Capture often delivers fewer real RAW frames than target. We still write a
/// true CFR timeline: missing slots **hold the last real frame**.
/// Result: file reports 24 fps with a wall-clock-aligned timeline.
final class VideoWriter: @unchecked Sendable {
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var audioInput: AVAssetWriterInput?
    private var metadataInput: AVAssetWriterInput?
    private var metadataFormatDesc: CMFormatDescription?
    let gyroRecorder = GyroMotionRecorder()
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var frameCount: Int64 = 0
    private var realFrameCount: Int64 = 0
    private var lastWrittenSlot: Int64 = -1
    private var width: Int = 0
    private var height: Int = 0
    private let targetFPS = CaptureFrameRate.fps24.rawValue
    private let frameDuration = CMTime(value: 1, timescale: 24)
    private var captureClock: CMClock?
    private var sessionStartTime: CMTime = .invalid
    private var lastVideoPTS: CMTime = .invalid
    private var lastAudioPTS: CMTime = .invalid
    private var hasStartedSession = false
    private var hasStartedWriterTimeline = false
    private var lastPixelBuffer: CVPixelBuffer?
    private var pendingAudioBuffers: [CMSampleBuffer] = []
    private let maxPendingAudioBuffers = 50
    private var prerollAudioBuffers: [CMSampleBuffer] = []
    private let maxPrerollAudioBuffers = 256
    private var curveType: LogCurveType = .sLog3Approx
    /// The codec the writer actually accepted. This can differ from the requested
    /// codec when VideoToolbox declines ProRes and we fall back to HEVC.
    private(set) var activeCodec: VideoCodecOption = .hevc
    private var expectedPixelFormat: OSType = 0
    private let lock = NSLock()

    var onLowDiskSpace: (@Sendable () -> Void)?
    var isRecording = false
    private(set) var droppedFrames: Int = 0
    private var backpressureEvents = 0

    /// Whether the writer is ready to accept a new frame without backpressure dropping.
    /// Check before queuing BGRA conversion + append.
    var isReady: Bool {
        lock.lock()
        defer { lock.unlock() }
        return isRecording && (videoInput?.isReadyForMoreMediaData ?? false)
    }

    func start(
        outputURL: URL,
        width: Int,
        height: Int,
        bitrate: Int = 100_000_000,
        captureClock: CMClock?,
        includeAudio: Bool = true,
        curveType: LogCurveType = .sLog3Approx,
        codec: VideoCodecOption = .hevc,
        orientation: UIInterfaceOrientation = .landscapeRight
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        try? FileManager.default.removeItem(at: outputURL)

        let fps = targetFPS
        let allowFrameReordering = DeviceCapabilities.chipTier(
            for: DeviceCapabilities.hardwareMachineIdentifier()
        ) != .a14

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
        self.width = width
        self.height = height
        self.captureClock = captureClock
        self.curveType = curveType

        var videoSettings: [String: Any] = [
            AVVideoWidthKey: width,
            AVVideoHeightKey: height
        ]

        if codec == .proRes422 || codec == .proRes422HQ {
            videoSettings[AVVideoCodecKey] = codec.avCodecType
        } else {
            let profileLevel: String
            switch curveType {
            case .linear:
                profileLevel = kVTProfileLevel_HEVC_Main_AutoLevel as String
            case .appleLog2, .sLog3Approx:
                profileLevel = kVTProfileLevel_HEVC_Main10_AutoLevel as String
            }

            let compression: [String: Any] = [
                AVVideoAverageBitRateKey: bitrate,
                kVTCompressionPropertyKey_ProfileLevel as String: profileLevel,
                kVTCompressionPropertyKey_RealTime as String: true as NSNumber,
                AVVideoExpectedSourceFrameRateKey: Int(fps),
                AVVideoAverageNonDroppableFrameRateKey: Int(fps),
                // Restore the lower-latency encode path on A14, where the RAW
                // burst and the encoder compete for real-time throughput.
                AVVideoMaxKeyFrameIntervalKey: Int(fps * 2),
                AVVideoAllowFrameReorderingKey: allowFrameReordering as NSNumber
            ]
            videoSettings[AVVideoCodecKey] = AVVideoCodecType.hevc
            videoSettings[AVVideoCompressionPropertiesKey] = compression
        }

        switch curveType {
        case .linear:
            videoSettings[AVVideoColorPropertiesKey] = [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
            ]
        case .appleLog2, .sLog3Approx:
            // AVAssetWriterInput strictly requires AVVideoTransferFunctionKey to be one of:
            // [ITU_R_2100_HLG, IEC_sRGB, SMPTE_ST_2084_PQ, Linear, ITU_R_709_2].
            // Passing custom or log transfer functions causes an NSInvalidArgumentException crash.
            // For log profiles, omitting
            // AVVideoColorPropertiesKey allows VideoToolbox to derive the exact 10-bit color
            // primaries, matrix, and Apple Log transfer function metadata directly from the
            // propagated CVPixelBuffer attachments (kCVImageBufferLogTransferFunctionKey).
            break
        }

        var resolvedCodec = codec
        var vInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        vInput.expectsMediaDataInRealTime = true
        vInput.mediaTimeScale = CMTimeScale(fps * 1000)
        vInput.transform = .identity

        // ProRes is intra-frame 10-bit 4:2:2. Feed it a true 10-bit 4:2:2 bi-planar buffer so
        // Apple's encoder never has to upsample a 4:2:0 source. HEVC log stays 10-bit 4:2:0;
        // Linear debug output remains 8-bit BGRA.
        let isProRes = (resolvedCodec == .proRes422 || resolvedCodec == .proRes422HQ)
        var pixelFormatType: OSType
        if isProRes {
            pixelFormatType = kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange
        } else if curveType == .linear {
            pixelFormatType = kCVPixelFormatType_32BGRA
        } else {
            pixelFormatType = kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        }

        var adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: vInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: pixelFormatType,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferMetalCompatibilityKey as String: true,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
            ])

        // Hardware ProRes encoder on iOS only supports 16:9 standard broadcast resolutions (1080p).
        // If writer.canAdd rejects ProRes, gracefully fall back to 10-bit HEVC so recording never fails.
        if !writer.canAdd(vInput) && (resolvedCodec == .proRes422 || resolvedCodec == .proRes422HQ) {
            print("[VideoWriter] ProRes (\(resolvedCodec.displayName)) rejected by VideoToolbox for \(width)x\(height). Falling back to HEVC 10-bit.")
            resolvedCodec = .hevc
            let profileLevel: String
            switch curveType {
            case .linear:
                profileLevel = kVTProfileLevel_HEVC_Main_AutoLevel as String
            case .appleLog2, .sLog3Approx:
                profileLevel = kVTProfileLevel_HEVC_Main10_AutoLevel as String
            }
            let compression: [String: Any] = [
                AVVideoAverageBitRateKey: bitrate,
                kVTCompressionPropertyKey_ProfileLevel as String: profileLevel,
                kVTCompressionPropertyKey_RealTime as String: true as NSNumber,
                AVVideoExpectedSourceFrameRateKey: Int(fps),
                AVVideoAverageNonDroppableFrameRateKey: Int(fps),
                AVVideoMaxKeyFrameIntervalKey: Int(fps * 2),
                AVVideoAllowFrameReorderingKey: allowFrameReordering as NSNumber
            ]
            var fallbackSettings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.hevc,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
                AVVideoCompressionPropertiesKey: compression
            ]
            if curveType == .linear {
                fallbackSettings[AVVideoColorPropertiesKey] = [
                    AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                    AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                    AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
                ]
            }
            let fallbackInput = AVAssetWriterInput(mediaType: .video, outputSettings: fallbackSettings)
            fallbackInput.expectsMediaDataInRealTime = true
            fallbackInput.mediaTimeScale = CMTimeScale(fps * 1000)
            fallbackInput.transform = .identity
            let fallbackPixelFormat = (curveType == .linear) ? kCVPixelFormatType_32BGRA : kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
            let fallbackAdaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: fallbackInput,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: fallbackPixelFormat,
                    kCVPixelBufferWidthKey as String: width,
                    kCVPixelBufferHeightKey as String: height,
                    kCVPixelBufferMetalCompatibilityKey as String: true,
                    kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
                ])
            vInput = fallbackInput
            adaptor = fallbackAdaptor
            pixelFormatType = fallbackPixelFormat
        }

        guard writer.canAdd(vInput) else {
            throw NSError(domain: "RawLogCam", code: 10, userInfo: [NSLocalizedDescriptionKey: "Cannot add video input (\(resolvedCodec.displayName))"])
        }
        writer.add(vInput)

        var aInput: AVAssetWriterInput?
        if includeAudio {
            let audioSettings: [String: Any] = [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: 48_000,
                AVNumberOfChannelsKey: 2,
                AVEncoderBitRateKey: 256_000
            ]
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
            input.expectsMediaDataInRealTime = true
            if writer.canAdd(input) {
                writer.add(input)
                aInput = input
            }
        }

        // Gyroflow motion logging & embedded CAMM metadata track
        gyroRecorder.start(orientation: orientation, fps: fps)
        var formatDesc: CMFormatDescription?
        let spec: [String: Any] = [
            kCMMetadataFormatDescriptionMetadataSpecificationKey_Identifier as String: "mdta/com.google.camm",
            kCMMetadataFormatDescriptionMetadataSpecificationKey_DataType as String: kCMMetadataBaseDataType_RawData as String
        ]
        let metaStatus = CMMetadataFormatDescriptionCreateWithMetadataSpecifications(
            allocator: kCFAllocatorDefault,
            metadataType: kCMMetadataFormatType_Boxed,
            metadataSpecifications: [spec] as CFArray,
            formatDescriptionOut: &formatDesc
        )
        var mInput: AVAssetWriterInput?
        if metaStatus == noErr, let formatDesc {
            let meta = AVAssetWriterInput(mediaType: .metadata, outputSettings: nil, sourceFormatHint: formatDesc)
            meta.expectsMediaDataInRealTime = true
            if writer.canAdd(meta) {
                writer.add(meta)
                mInput = meta
                self.metadataFormatDesc = formatDesc
            }
        }

        writer.metadata = Self.makeMetadataItems(curveType: curveType, codec: resolvedCodec, fps: fps, width: width, height: height)

        guard writer.startWriting() else {
            throw writer.error ?? NSError(domain: "RawLogCam", code: 11, userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter failed to start"])
        }

        self.assetWriter = writer
        self.activeCodec = resolvedCodec
        self.expectedPixelFormat = pixelFormatType
        self.videoInput = vInput
        self.audioInput = aInput
        self.metadataInput = mInput
        self.pixelBufferAdaptor = adaptor
        self.frameCount = 0
        self.realFrameCount = 0
        self.lastWrittenSlot = -1
        self.droppedFrames = 0
        self.backpressureEvents = 0
        self.sessionStartTime = .invalid
        self.lastVideoPTS = .invalid
        self.lastAudioPTS = .invalid
        self.hasStartedSession = false
        self.hasStartedWriterTimeline = false
        self.lastPixelBuffer = nil
        self.prerollAudioBuffers.removeAll()
        self.pendingAudioBuffers.removeAll()
        self.lastDiskCheckTime = 0
        self.cachedHasSpace = true
        self.isRecording = true

        print("[VideoWriter] CFR \(Int(fps))fps \(width)x\(height) codec=\(resolvedCodec.displayName) bitrate=\(bitrate) orientation=\(orientation.rawValue)")
    }

    private func drainPendingAudioBuffersLocked(trimFinalBuffer: Bool = false) {
        guard let input = audioInput, lastVideoPTS.isNumeric else { return }
        let videoEnd = CMTimeAdd(lastVideoPTS, frameDuration)
        var drainedCount = 0
        for buffer in pendingAudioBuffers {
            let pts = CMSampleBufferGetPresentationTimeStamp(buffer)
            let duration = CMSampleBufferGetDuration(buffer)
            let end = duration.isNumeric ? CMTimeAdd(pts, duration) : pts
            if pts >= videoEnd { break }
            if end > videoEnd {
                guard trimFinalBuffer, input.isReadyForMoreMediaData else { break }
                let count = CMSampleBufferGetNumSamples(buffer)
                let remaining = CMTimeSubtract(videoEnd, pts)
                let keep = Int((remaining.seconds / duration.seconds * Double(count)).rounded(.down))
                if keep > 0, keep < count {
                    var trimmed: CMSampleBuffer?
                    if CMSampleBufferCopySampleBufferForRange(
                        allocator: kCFAllocatorDefault, sampleBuffer: buffer,
                        sampleRange: CFRange(location: 0, length: keep), sampleBufferOut: &trimmed
                    ) == noErr, let trimmed {
                        _ = input.append(trimmed)
                    }
                }
                break
            }
            guard input.isReadyForMoreMediaData, input.append(buffer) else { break }
            drainedCount += 1
        }
        if drainedCount > 0 {
            pendingAudioBuffers.removeSubrange(0..<drainedCount)
        }
    }

    private func drainPrerollAudioBuffersLocked() {
        for buffer in prerollAudioBuffers {
            _ = enqueueRetimedAudioLocked(buffer)
        }
        prerollAudioBuffers.removeAll()
    }

    private var lastDiskCheckTime: CFTimeInterval = 0
    private var cachedHasSpace = true

    private func hasSufficientDiskSpace() -> Bool {
        let now = CACurrentMediaTime()
        if now - lastDiskCheckTime < 1.0 {
            return cachedHasSpace
        }
        lastDiskCheckTime = now
        guard let outputURL = assetWriter?.outputURL else { return true }
        do {
            let values = try outputURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            if let available = values.volumeAvailableCapacityForImportantUsage {
                // If less than 500 MB left, notify handler to stop recording safely and refuse further frames
                if available <= 500 * 1024 * 1024 {
                    onLowDiskSpace?()
                    cachedHasSpace = false
                    return false
                }
                cachedHasSpace = true
                return true
            }
        } catch {}
        cachedHasSpace = true
        return true
    }

    /// Append a real camera frame. Fills any missing CFR slots by holding last frame.
    @discardableResult
    func appendFrame(pixelBuffer: CVPixelBuffer, captureTime: CMTime) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard isRecording, captureTime.isNumeric,
              let adaptor = pixelBufferAdaptor,
              let input = videoInput,
              hasSufficientDiskSpace() else {
            droppedFrames += 1
            return false
        }

        let pbW = CVPixelBufferGetWidth(pixelBuffer)
        let pbH = CVPixelBufferGetHeight(pixelBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        if pbW != width || pbH != height || pixelFormat != expectedPixelFormat {
            print("[VideoWriter] Rejecting frame: got \(pbW)x\(pbH) format=\(pixelFormat), expected \(width)x\(height) format=\(expectedPixelFormat) for \(activeCodec.displayName)")
            droppedFrames += 1
            return false
        }

        // Color space & transfer characteristics are attached once at buffer creation by MetalPipeline.encodeOutputPixelBuffer.
        if CVBufferCopyAttachment(pixelBuffer, kCVImageBufferColorPrimariesKey, nil) == nil {
            MetalPipeline.attachColorMetadata(to: pixelBuffer, curveType: curveType)
        }

        if !hasStartedSession {
            // VideoToolbox initialization on Frame 0 can take 10-30ms.
            // Patient spin-wait (up to 150ms) ensures Frame 0 is NEVER dropped.
            if !input.isReadyForMoreMediaData { backpressureEvents += 1 }
            var spinAttempts = 0
            while !input.isReadyForMoreMediaData && spinAttempts < 50 {
                lock.unlock()
                usleep(3000) // 3ms sleep outside lock
                lock.lock()
                guard isRecording, videoInput != nil, pixelBufferAdaptor != nil else {
                    droppedFrames += 1
                    return false
                }
                spinAttempts += 1
            }

            guard input.isReadyForMoreMediaData else {
                print("[VideoWriter] VideoToolbox not ready for Frame 0 after 150ms")
                droppedFrames += 1
                return false
            }

            if !hasStartedWriterTimeline {
                assetWriter?.startSession(atSourceTime: .zero)
                hasStartedWriterTimeline = true
            }
            let firstPTS = CMTime(value: 0, timescale: frameDuration.timescale)
            guard adaptor.append(pixelBuffer, withPresentationTime: firstPTS) else {
                print("[VideoWriter] Failed to append Frame 0 to pixelBufferAdaptor: \(String(describing: assetWriter?.error))")
                droppedFrames += 1
                return false
            }

            // Frame 0 successfully committed at PTS .zero!
            hasStartedSession = true
            sessionStartTime = captureTime
            lastVideoPTS = firstPTS
            lastWrittenSlot = 0
            frameCount = 1
            realFrameCount = 1
            lastPixelBuffer = pixelBuffer

            if let captureClock {
                let hostTime = CMSyncConvertTime(captureTime, from: captureClock, to: CMClockGetHostTimeClock())
                if hostTime.isNumeric {
                    gyroRecorder.anchorSession(startHostTime: hostTime.seconds)
                }
            }
            appendGyroMetadataLocked(elapsedSeconds: 0)

            // DRAIN PREROLL AUDIO NOW: Video Frame 0 is committed at PTS .zero, so audio begins precisely at .zero
            drainPrerollAudioBuffersLocked()
            drainPendingAudioBuffersLocked()
            return true
        }

        let elapsed = CMTimeSubtract(captureTime, sessionStartTime)
        guard elapsed.isNumeric else {
            droppedFrames += 1
            return false
        }
        let targetSlot = CMTimeConvertScale(elapsed, timescale: frameDuration.timescale,
                                            method: .roundHalfAwayFromZero).value

        // If targetSlot has already been written, this frame is an intra-slot burst/duplicate.
        // Drop it so subsequent frames are NEVER shifted forward in time.
        if targetSlot <= lastWrittenSlot {
            droppedFrames += 1
            return false
        }

        // Drain any pending audio buffers now that video input might have made room
        drainPendingAudioBuffersLocked()

        // VideoToolbox backpressure handling:
        // Hardware encoder occasionally takes 1-4ms to complete a GOP slice.
        // Spin-wait briefly outside lock before discarding to eliminate spurious frame drops.
        if !input.isReadyForMoreMediaData { backpressureEvents += 1 }
        var spinAttempts = 0
        while !input.isReadyForMoreMediaData && spinAttempts < 15 {
            lock.unlock()
            usleep(2000) // 2ms sleep outside lock
            lock.lock()
            guard isRecording, videoInput != nil, pixelBufferAdaptor != nil else {
                droppedFrames += 1
                return false
            }
            spinAttempts += 1
        }

        // Fill every missing CFR slot by repeating the last frame. Capping this
        // at 0.5 seconds used to leave holes in the presentation timeline after
        // a RAW-capture stall, producing a variable-duration track while claiming
        // CFR. A long stall now costs temporal freshness, not timeline integrity.
        if let hold = lastPixelBuffer, targetSlot > lastWrittenSlot + 1 {
            var slotToFill = lastWrittenSlot + 1
            while slotToFill < targetSlot {
                if !input.isReadyForMoreMediaData { backpressureEvents += 1 }
                var holdSpin = 0
                while !input.isReadyForMoreMediaData && holdSpin < 5 {
                    lock.unlock()
                    usleep(1500)
                    lock.lock()
                    guard isRecording, videoInput != nil, pixelBufferAdaptor != nil else { return false }
                    holdSpin += 1
                }
                guard input.isReadyForMoreMediaData else {
                    droppedFrames += 1
                    return false
                }
                let holdPTS = CMTime(value: slotToFill, timescale: frameDuration.timescale)
                let holdElapsed = Double(slotToFill) / targetFPS
                if adaptor.append(hold, withPresentationTime: holdPTS) {
                    lastVideoPTS = holdPTS
                    appendGyroMetadataLocked(elapsedSeconds: holdElapsed)
                    lastWrittenSlot = slotToFill
                    frameCount = slotToFill + 1
                    slotToFill += 1
                } else {
                    droppedFrames += 1
                    return false
                }
            }
        }

        // Current real frame
        guard targetSlot == lastWrittenSlot + 1, input.isReadyForMoreMediaData else {
            droppedFrames += 1
            return false
        }

        let currentPTS = CMTime(value: targetSlot, timescale: frameDuration.timescale)
        let currentElapsed = Double(targetSlot) / targetFPS

        if adaptor.append(pixelBuffer, withPresentationTime: currentPTS) {
            lastVideoPTS = currentPTS
            appendGyroMetadataLocked(elapsedSeconds: currentElapsed)
            lastWrittenSlot = targetSlot
            frameCount = targetSlot + 1
            realFrameCount += 1
            lastPixelBuffer = pixelBuffer
            drainPendingAudioBuffersLocked()
            return true
        } else {
            droppedFrames += 1
            return false
        }
    }

    private func appendGyroMetadataLocked(elapsedSeconds: Double) {
        guard let mInput = metadataInput,
              let formatDesc = metadataFormatDesc else { return }

        let samples = gyroRecorder.drainSamples(upToElapsedSeconds: elapsedSeconds)
        for sample in samples {
            guard mInput.isReadyForMoreMediaData else { break }

            let cammData = sample.toCAMMData()
            var blockBuffer: CMBlockBuffer?
            let bbStatus = CMBlockBufferCreateWithMemoryBlock(
                allocator: kCFAllocatorDefault,
                memoryBlock: nil,
                blockLength: cammData.count,
                blockAllocator: kCFAllocatorDefault,
                customBlockSource: nil,
                offsetToData: 0,
                dataLength: cammData.count,
                flags: 0,
                blockBufferOut: &blockBuffer
            )
            guard bbStatus == noErr, let bb = blockBuffer else { continue }

            cammData.withUnsafeBytes { rawBuffer in
                if let ptr = rawBuffer.baseAddress {
                    CMBlockBufferReplaceDataBytes(
                        with: ptr,
                        blockBuffer: bb,
                        offsetIntoDestination: 0,
                        dataLength: cammData.count
                    )
                }
            }

            let samplePTS = CMTime(seconds: sample.timeMs / 1000.0, preferredTimescale: 24000)
            let duration = CMTime(value: 10, timescale: 1000)
            var timing = CMSampleTimingInfo(duration: duration, presentationTimeStamp: samplePTS, decodeTimeStamp: .invalid)

            var sampleBuffer: CMSampleBuffer?
            let sbStatus = CMSampleBufferCreateReady(
                allocator: kCFAllocatorDefault,
                dataBuffer: bb,
                formatDescription: formatDesc,
                sampleCount: 1,
                sampleTimingEntryCount: 1,
                sampleTimingArray: &timing,
                sampleSizeEntryCount: 1,
                sampleSizeArray: [cammData.count],
                sampleBufferOut: &sampleBuffer
            )
            guard sbStatus == noErr, let sampleBuffer else { continue }
            _ = mInput.append(sampleBuffer)
        }
    }

    @discardableResult
    func appendAudio(sampleBuffer: CMSampleBuffer) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard isRecording, audioInput != nil,
              CMSampleBufferDataIsReady(sampleBuffer),
              CMSampleBufferGetPresentationTimeStamp(sampleBuffer).isNumeric else { return false }

        if !hasStartedSession {
            if prerollAudioBuffers.count == maxPrerollAudioBuffers {
                prerollAudioBuffers.removeFirst()
            }
            prerollAudioBuffers.append(sampleBuffer)
            return true
        }

        drainPendingAudioBuffersLocked()
        return enqueueRetimedAudioLocked(sampleBuffer)
    }

    private func enqueueRetimedAudioLocked(_ source: CMSampleBuffer) -> Bool {
        guard audioInput != nil, sessionStartTime.isNumeric else { return false }
        var buffer = source
        let sourcePTS = CMSampleBufferGetPresentationTimeStamp(buffer)
        guard sourcePTS.isNumeric else { return false }

        if sourcePTS < sessionStartTime {
            // Preserve PCM that straddles frame zero by removing only samples
            // captured before the video anchor.
            let duration = CMSampleBufferGetDuration(buffer)
            let count = CMSampleBufferGetNumSamples(buffer)
            let offset = CMTimeSubtract(sessionStartTime, sourcePTS)
            guard duration.isNumeric, duration.seconds > 0, offset < duration, count > 0 else { return true }
            let samplesToTrim = Int((offset.seconds / duration.seconds * Double(count)).rounded(.up))
            guard samplesToTrim < count else { return true }
            var trimmed: CMSampleBuffer?
            guard CMSampleBufferCopySampleBufferForRange(
                allocator: kCFAllocatorDefault, sampleBuffer: buffer,
                sampleRange: CFRange(location: samplesToTrim, length: count - samplesToTrim),
                sampleBufferOut: &trimmed
            ) == noErr, let trimmed else { return false }
            buffer = trimmed
        }

        var count: CMItemCount = 0
        guard CMSampleBufferGetSampleTimingInfoArray(buffer, entryCount: 0, arrayToFill: nil,
                                                     entriesNeededOut: &count) == noErr, count > 0 else { return false }
        var timings = Array(repeating: CMSampleTimingInfo(duration: .invalid,
                            presentationTimeStamp: .invalid, decodeTimeStamp: .invalid), count: count)
        guard CMSampleBufferGetSampleTimingInfoArray(buffer, entryCount: count, arrayToFill: &timings,
                                                     entriesNeededOut: nil) == noErr else { return false }

        var previousPTS = lastAudioPTS
        for index in timings.indices {
            let pts = CMTimeSubtract(timings[index].presentationTimeStamp, sessionStartTime)
            guard pts.isNumeric, pts >= .zero, !previousPTS.isNumeric || pts > previousPTS else { return false }
            timings[index].presentationTimeStamp = pts
            if timings[index].decodeTimeStamp.isNumeric {
                timings[index].decodeTimeStamp = CMTimeSubtract(timings[index].decodeTimeStamp, sessionStartTime)
            }
            previousPTS = pts
        }

        var retimed: CMSampleBuffer?
        guard CMSampleBufferCreateCopyWithNewTiming(
            allocator: kCFAllocatorDefault, sampleBuffer: buffer,
            sampleTimingEntryCount: count, sampleTimingArray: &timings, sampleBufferOut: &retimed
        ) == noErr, let retimed else { return false }

        guard pendingAudioBuffers.count < maxPendingAudioBuffers else { return false }
        pendingAudioBuffers.append(retimed)
        lastAudioPTS = previousPTS
        drainPendingAudioBuffersLocked()
        return true
    }

    func finish(completion: @Sendable @escaping (URL?, Error?) -> Void) {
        lock.lock()
        guard isRecording else {
            lock.unlock()
            completion(nil, NSError(domain: "OwLens", code: 100, userInfo: [NSLocalizedDescriptionKey: "Recording was not active"]))
            return
        }

        isRecording = false

        // Tail freeze elimination:
        // Video track duration ends at lastVideoPTS + 1 nominal frame.
        // Drain pending audio buffers strictly up to videoEndTime and discard any trailing audio
        // that extends beyond videoEndTime so the audio track NEVER outlasts the video track
        // (which causes video players to freeze on the final frame).
        let videoEndTime: CMTime
        if lastVideoPTS.isValid {
            videoEndTime = CMTimeAdd(lastVideoPTS, frameDuration)
        } else {
            videoEndTime = .invalid
        }

        if videoEndTime.isNumeric {
            drainPendingAudioBuffersLocked(trimFinalBuffer: true)
        }
        pendingAudioBuffers.removeAll()
        prerollAudioBuffers.removeAll()

        let url = assetWriter?.outputURL
        isRecording = false
        sessionStartTime = .invalid
        lastVideoPTS = .invalid
        lastAudioPTS = .invalid
        captureClock = nil
        hasStartedSession = false
        lastPixelBuffer = nil
        lastWrittenSlot = -1
        let writer = assetWriter
        let vIn = videoInput
        let aIn = audioInput
        let mIn = metadataInput
        let total = frameCount
        let real = realFrameCount
        let drops = droppedFrames
        let stalls = backpressureEvents
        let fps = targetFPS
        let videoDurationSeconds = (videoEndTime.isValid && videoEndTime.seconds > 0) ? videoEndTime.seconds : (Double(total) / fps)

        appendGyroMetadataLocked(elapsedSeconds: videoDurationSeconds)

        if let writer = assetWriter, writer.status == .writing, videoEndTime.isValid {
            writer.endSession(atSourceTime: videoEndTime)
        }

        lock.unlock()

        // `writer` (AVAssetWriter) is non-Sendable; box it so the @Sendable
        // finishWriting callback can capture it. If there is no writer there is
        // nothing to finalize — report failure instead of silently never
        // invoking completion (which would hang the caller).
        guard let writer else {
            completion(nil, NSError(domain: "OwLens", code: 101, userInfo: [NSLocalizedDescriptionKey: "No active asset writer"]))
            return
        }

        if real == 0 {
            print("[VideoWriter] No real frames were appended to the video timeline.")
            if writer.status == .writing { writer.cancelWriting() }
            gyroRecorder.stop()
            if let url {
                try? FileManager.default.removeItem(at: url)
            }
            completion(nil, NSError(domain: "OwLens", code: 102, userInfo: [NSLocalizedDescriptionKey: "Recording ended with no frames"]))
            return
        }
        guard writer.status == .writing else {
            gyroRecorder.stop()
            completion(nil, writer.error ?? NSError(domain: "OwLens", code: 103, userInfo: [
                NSLocalizedDescriptionKey: "Video writer failed to finalize file"
            ]))
            return
        }
        vIn?.markAsFinished()
        aIn?.markAsFinished()
        mIn?.markAsFinished()
        let boxedWriter = SendableBox(value: writer)
        writer.finishWriting {
            let status = boxedWriter.value.status
            let duration = videoDurationSeconds
            let freshFPS = duration > 0 ? Double(real) / duration : 0
            print("[VideoWriter] Done. slots=\(total) unique=\(real) held=\(total - real) drops=\(drops) backpressure=\(stalls) freshFPS=\(freshFPS) \(String(format: "%.2f", duration))s @ \(Int(fps))fps status=\(String(describing: status))")
            let outputURL = boxedWriter.value.outputURL
            let gcsvData = self.gyroRecorder.exportGCSVData(videoFileName: outputURL.lastPathComponent)
            self.gyroRecorder.stop()
            if status != .completed {
                let err = boxedWriter.value.error ?? NSError(domain: "OwLens", code: 103, userInfo: [NSLocalizedDescriptionKey: "Video writer failed to finalize file"])
                print("[VideoWriter] Error: \(err)")
                completion(nil, err)
            } else {
                Self.patchMebxToCamm(at: outputURL)

                // 1. Always save dedicated .gcsv file to Documents/OwLens Gyro in Files app
                GyroMotionRecorder.saveGCSVFile(data: gcsvData, videoFileName: outputURL.lastPathComponent)

                // 2. Also write companion .gcsv alongside temporary video file
                let tempGcsvURL = outputURL.deletingPathExtension().appendingPathExtension("gcsv")
                try? gcsvData.write(to: tempGcsvURL, options: .atomic)

                completion(url, nil)
            }
        }
    }

    /// In-place patch to replace AVFoundation's default 'mebx' sample entry identifier
    /// in the metadata track's 'stsd' box with 'camm' (Google Camera Motion Metadata).
    /// This enables standard Gyroflow and DaVinci Resolve OpenFX plugin to natively detect
    /// and parse the embedded gyroscope stream with zero external sidecar files.
    static func patchMebxToCamm(at url: URL) {
        guard let file = try? FileHandle(forUpdating: url) else { return }
        defer { try? file.close() }

        func fourCC(_ text: String) -> UInt32 {
            text.utf8.reduce(0) { $0 << 8 | UInt32($1) }
        }

        func read(at offset: UInt64, count: Int) throws -> Data? {
            try file.seek(toOffset: offset)
            guard let bytes = try file.read(upToCount: count), bytes.count == count else { return nil }
            return bytes
        }

        struct Box {
            let type: UInt32
            let body: UInt64
            let end: UInt64
        }

        func box(at offset: UInt64, through limit: UInt64) throws -> Box? {
            guard offset <= limit, limit - offset >= 8,
                  let header = try read(at: offset, count: 8) else { return nil }
            let size32 = header.withUnsafeBytes { UInt32(bigEndian: $0.loadUnaligned(as: UInt32.self)) }
            let type = header.withUnsafeBytes { UInt32(bigEndian: $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self)) }
            let headerSize: UInt64 = size32 == 1 ? 16 : 8
            let size: UInt64
            if size32 == 1 {
                guard limit - offset >= headerSize, let bytes = try read(at: offset + 8, count: 8) else { return nil }
                size = bytes.withUnsafeBytes { UInt64(bigEndian: $0.loadUnaligned(as: UInt64.self)) }
            } else {
                size = size32 == 0 ? limit - offset : UInt64(size32)
            }
            guard size >= headerSize, size <= limit - offset else { return nil }
            return Box(type: type, body: offset + headerSize, end: offset + size)
        }

        let mebx = fourCC("mebx")
        func findEntry(from start: UInt64, through limit: UInt64, path: ArraySlice<UInt32>) throws -> UInt64? {
            if let child = path.first {
                var offset = start
                while let atom = try box(at: offset, through: limit) {
                    if atom.type == child,
                       let found = try findEntry(from: atom.body, through: atom.end, path: path.dropFirst()) {
                        return found
                    }
                    offset = atom.end
                }
            } else if limit - start >= 8, let header = try read(at: start, count: 8) {
                let count = header.withUnsafeBytes { UInt32(bigEndian: $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self)) }
                var offset = start + 8
                for _ in 0..<count {
                    guard let entry = try box(at: offset, through: limit) else { return nil }
                    if entry.type == mebx && entry.end - entry.body >= 8 { return offset + 4 }
                    offset = entry.end
                }
            }
            return nil
        }

        do {
            let size = try file.seekToEnd()
            let path = ["moov", "trak", "mdia", "minf", "stbl", "stsd"].map(fourCC)
            if let offset = try findEntry(from: 0, through: size, path: path[...]) {
                try file.seek(toOffset: offset)
                try file.write(contentsOf: Data("camm".utf8))
            }
        } catch {
            print("[VideoWriter] Could not tag CAMM metadata: \(error.localizedDescription)")
        }
    }

    private static func makeMetadataItems(
        curveType: LogCurveType,
        codec: VideoCodecOption = .hevc,
        fps: Double,
        width: Int,
        height: Int
    ) -> [AVMetadataItem] {
        var items: [AVMetadataItem] = []

        let gamut: String
        switch curveType {
        case .appleLog2:
            gamut = "Apple Wide Gamut"
        case .sLog3Approx:
            gamut = "ITU-R BT.2020"
        case .linear:
            gamut = "Rec.709"
        }
        let gamma = curveType.displayName

        // 1. Common Description
        let desc = AVMutableMetadataItem()
        desc.keySpace = .common
        desc.key = AVMetadataKey.commonKeyDescription as (NSCopying & NSObjectProtocol)
        desc.value = "\(gamma) · \(gamut) · \(codec.displayName)" as (NSCopying & NSObjectProtocol)
        items.append(desc)

        // 2. QuickTime User Data Description
        let qtDesc = AVMutableMetadataItem()
        qtDesc.keySpace = .quickTimeUserData
        qtDesc.key = AVMetadataKey.quickTimeUserDataKeyDescription as (NSCopying & NSObjectProtocol)
        qtDesc.value = "\(gamma) · \(gamut) · \(codec.displayName)" as (NSCopying & NSObjectProtocol)
        items.append(qtDesc)

        // 3. QuickTime Software
        let software = AVMutableMetadataItem()
        software.keySpace = .quickTimeUserData
        software.key = AVMetadataKey.quickTimeUserDataKeySoftware as (NSCopying & NSObjectProtocol)
        software.value = "OwLens Cinema Camera v2.1.6" as (NSCopying & NSObjectProtocol)
        items.append(software)

        // 4. QuickTime Make
        let make = AVMutableMetadataItem()
        make.keySpace = .quickTimeUserData
        make.key = AVMetadataKey.quickTimeUserDataKeyMake as (NSCopying & NSObjectProtocol)
        make.value = "Apple / \(gamma)" as (NSCopying & NSObjectProtocol)
        items.append(make)

        // 5. QuickTime Model / Profile
        let model = AVMutableMetadataItem()
        model.keySpace = .quickTimeUserData
        model.key = AVMetadataKey.quickTimeUserDataKeyModel as (NSCopying & NSObjectProtocol)
        model.value = "\(gamma) / \(gamut) 10-Bit" as (NSCopying & NSObjectProtocol)
        items.append(model)

        // 6. Detailed Comment for DaVinci Resolve / FCP NLEs
        let comment = AVMutableMetadataItem()
        comment.keySpace = .quickTimeUserData
        comment.key = AVMetadataKey.quickTimeUserDataKeyComment as (NSCopying & NSObjectProtocol)
        comment.value = "Color Space: \(gamut) | Gamma: \(gamma) | Frame Rate: \(Int(fps)) fps | Format: \(width)x\(height) \(codec.displayName)" as (NSCopying & NSObjectProtocol)
        items.append(comment)

        return items
    }
}
