import AVFoundation
import CoreVideo
import VideoToolbox
import QuartzCore
import UIKit


/// AVAssetWriter — HEVC + optional AAC at **constant** 24 or 30 fps.
///
/// Capture often delivers fewer real RAW frames than target. We still write a
/// true CFR timeline: missing slots **hold the last real frame**.
/// Result: file reports 24/30 fps, duration ≈ wall-clock, no “20 fps” metadata.
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
    private var width: Int = 0
    private var height: Int = 0
    private var targetFPS: Double = 24
    private var startHostTime: CFTimeInterval = 0
    private var audioReferenceTime: CMTime = .invalid
    private var sessionStartTime: CMTime = .invalid
    private var lastFrameCaptureTime: CMTime = .invalid
    private var lastFrameHostTime: CFTimeInterval = 0
    private var hasStartedSession = false
    private var lastPixelBuffer: CVPixelBuffer?
    private var pendingAudioBuffers: [CMSampleBuffer] = []
    private let maxPendingAudioBuffers = 50
    private var prerollAudioBuffers: [CMSampleBuffer] = []
    private let maxPrerollAudioBuffers = 20
    private var curveType: LogCurveType = .sLog3Approx
    private let lock = NSLock()

    var onLowDiskSpace: (@Sendable () -> Void)?
    var isRecording = false
    private(set) var droppedFrames: Int = 0

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
        targetFPS: Double = 24,
        includeAudio: Bool = true,
        curveType: LogCurveType = .sLog3Approx,
        codec: VideoCodecOption = .hevc,
        orientation: UIInterfaceOrientation = .landscapeRight
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        try? FileManager.default.removeItem(at: outputURL)

        // Only 24 / 30 supported for reliable CFR
        let fps = (abs(targetFPS - 30) < 0.5) ? 30.0 : 24.0

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
        self.width = width
        self.height = height
        self.targetFPS = fps
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
                AVVideoMaxKeyFrameIntervalKey: Int(fps),
                AVVideoAllowFrameReorderingKey: false as NSNumber
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
            // For log profiles (Apple Log 2 and S-Log3 in BT.2020 container), omitting
            // AVVideoColorPropertiesKey allows VideoToolbox to derive the exact 10-bit color
            // primaries, matrix, and Apple Log transfer function metadata directly from the
            // propagated CVPixelBuffer attachments (kCVImageBufferLogTransferFunctionKey).
            break
        }

        let vInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        vInput.expectsMediaDataInRealTime = true
        vInput.mediaTimeScale = CMTimeScale(fps * 1000)

        // If shooting in Landscape Left, rotate video track by 180° so video plays upright
        // in standard players (QuickTime, DaVinci Resolve, FCP) without upside-down playback.
        if orientation == .landscapeLeft {
            vInput.transform = CGAffineTransform(rotationAngle: .pi).translatedBy(x: -CGFloat(width), y: -CGFloat(height))
        } else {
            vInput.transform = .identity
        }

        // Use 10-bit bi-planar YCbCr for accurate LOG gradient recording.
        let pixelFormatType: OSType = (curveType == .linear)
            ? kCVPixelFormatType_32BGRA
            : kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: vInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: pixelFormatType,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferMetalCompatibilityKey as String: true,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
            ])

        guard writer.canAdd(vInput) else {
            throw NSError(domain: "RawLogCam", code: 10, userInfo: [NSLocalizedDescriptionKey: "Cannot add video input (\(codec.displayName))"])
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

        writer.metadata = Self.makeMetadataItems(curveType: curveType, fps: fps, width: width, height: height)

        guard writer.startWriting() else {
            throw writer.error ?? NSError(domain: "RawLogCam", code: 11, userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter failed to start"])
        }

        self.assetWriter = writer
        self.videoInput = vInput
        self.audioInput = aInput
        self.metadataInput = mInput
        self.pixelBufferAdaptor = adaptor
        self.frameCount = 0
        self.realFrameCount = 0
        self.droppedFrames = 0
        self.startHostTime = 0
        self.sessionStartTime = .invalid
        self.hasStartedSession = false
        self.lastPixelBuffer = nil
        self.audioReferenceTime = .invalid
        self.pendingAudioBuffers.removeAll()
        self.lastDiskCheckTime = 0
        self.cachedHasSpace = true
        self.lastFrameCaptureTime = .invalid
        self.lastFrameHostTime = 0
        self.isRecording = true

        print("[VideoWriter] CFR \(Int(fps))fps \(width)x\(height) codec=\(codec.displayName) bitrate=\(bitrate) orientation=\(orientation.rawValue)")
    }

    private func drainPendingAudioBuffersLocked() {
        guard let input = audioInput else { return }
        var drainedCount = 0
        for buffer in pendingAudioBuffers {
            if input.isReadyForMoreMediaData {
                _ = input.append(buffer)
                drainedCount += 1
            } else {
                break
            }
        }
        if drainedCount > 0 {
            pendingAudioBuffers.removeSubrange(0..<drainedCount)
        }
    }

    private func drainPrerollAudioBuffersLocked() {
        guard let input = audioInput, hasStartedSession, sessionStartTime.isValid else {
            prerollAudioBuffers.removeAll()
            return
        }
        let preroll = prerollAudioBuffers
        prerollAudioBuffers.removeAll()

        for sampleBuffer in preroll {
            var timingCount: CMItemCount = 0
            CMSampleBufferGetSampleTimingInfoArray(sampleBuffer, entryCount: 0, arrayToFill: nil, entriesNeededOut: &timingCount)
            guard timingCount > 0 else { continue }

            var timings = Array(repeating: CMSampleTimingInfo(
                duration: .invalid,
                presentationTimeStamp: .invalid,
                decodeTimeStamp: .invalid
            ), count: timingCount)
            CMSampleBufferGetSampleTimingInfoArray(sampleBuffer, entryCount: timingCount, arrayToFill: &timings, entriesNeededOut: &timingCount)

            let refTime = sessionStartTime
            if timings[0].duration.isValid {
                let bufferEnd = CMTimeAdd(timings[0].presentationTimeStamp, timings[0].duration)
                if CMTimeCompare(bufferEnd, refTime) <= 0 {
                    continue
                }
            }

            for i in 0..<timings.count {
                var pts = CMTimeSubtract(timings[i].presentationTimeStamp, refTime)
                if CMTimeCompare(pts, .zero) < 0 { pts = .zero }
                timings[i].presentationTimeStamp = pts
                if timings[i].decodeTimeStamp.isValid {
                    timings[i].decodeTimeStamp = pts
                }
            }

            var retimed: CMSampleBuffer?
            let status = CMSampleBufferCreateCopyWithNewTiming(
                allocator: kCFAllocatorDefault,
                sampleBuffer: sampleBuffer,
                sampleTimingEntryCount: timingCount,
                sampleTimingArray: &timings,
                sampleBufferOut: &retimed
            )
            guard status == noErr, let retimed else { continue }

            if input.isReadyForMoreMediaData && pendingAudioBuffers.isEmpty {
                _ = input.append(retimed)
            } else if pendingAudioBuffers.count < maxPendingAudioBuffers {
                pendingAudioBuffers.append(retimed)
            }
        }
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
    func appendFrame(pixelBuffer: CVPixelBuffer, captureTime: CMTime? = nil) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard isRecording,
              let adaptor = pixelBufferAdaptor,
              let input = videoInput,
              hasSufficientDiskSpace() else {
            droppedFrames += 1
            return false
        }

        let pbW = CVPixelBufferGetWidth(pixelBuffer)
        let pbH = CVPixelBufferGetHeight(pixelBuffer)
        if pbW != width || pbH != height {
            droppedFrames += 1
            return false
        }

        // Color space & transfer characteristics are attached once at buffer creation by MetalPipeline.encodeOutputPixelBuffer.
        if CVBufferGetAttachment(pixelBuffer, kCVImageBufferColorPrimariesKey, nil) == nil {
            MetalPipeline.attachColorMetadata(to: pixelBuffer, curveType: curveType)
        }

        let now = CACurrentMediaTime()

        if !hasStartedSession {
            startHostTime = now
            if let captureTime = captureTime, captureTime.isValid {
                sessionStartTime = captureTime
                lastFrameCaptureTime = captureTime
            }
            lastFrameHostTime = now
            assetWriter?.startSession(atSourceTime: .zero)
            hasStartedSession = true
            gyroRecorder.anchorSession(startHostTime: now)
            drainPrerollAudioBuffersLocked()
        } else if realFrameCount <= 1 {
            // Startup grace: If initial pipeline spin-up delayed the second frame,
            // re-anchor session start to prevent a burst of duplicate frames at file start.
            let deltaFromStart: Double
            if let captureTime = captureTime, captureTime.isValid, sessionStartTime.isValid {
                deltaFromStart = max(0, CMTimeSubtract(captureTime, sessionStartTime).seconds)
            } else {
                deltaFromStart = max(0, now - startHostTime)
            }
            if deltaFromStart > (1.5 / targetFPS) {
                if let captureTime = captureTime, captureTime.isValid {
                    sessionStartTime = captureTime
                    lastFrameCaptureTime = captureTime
                }
                startHostTime = now
                lastFrameHostTime = now
                gyroRecorder.anchorSession(startHostTime: now)
            }
        }

        // True CFR timeline pacing with jitter deadband:
        // By anchoring to cumulative elapsed time, we guarantee zero cumulative audio/video drift.
        // A held frame is ONLY injected if cumulative elapsed time indicates that an entire frame
        // period (>= 1.75 * interval) was genuinely dropped by camera hardware.
        // Minor 5-25ms timer/sensor jitters never trigger duplicate frames, eliminating stutter
        // while preserving rock-solid audio/video synchronization over arbitrarily long recordings.
        let elapsedSeconds: Double
        if let captureTime = captureTime, captureTime.isValid, sessionStartTime.isValid {
            elapsedSeconds = max(0, CMTimeSubtract(captureTime, sessionStartTime).seconds)
        } else {
            elapsedSeconds = max(0, now - startHostTime)
        }

        let expectedFrames = elapsedSeconds * targetFPS
        let cumulativeDrift = expectedFrames - Double(frameCount)

        if cumulativeDrift >= 1.75, let hold = lastPixelBuffer {
            let missedSlots = min(10, Int(cumulativeDrift))
            for _ in 0..<missedSlots {
                guard input.isReadyForMoreMediaData else {
                    droppedFrames += 1
                    break
                }
                if writeCFR(hold, index: frameCount, adaptor: adaptor) {
                    appendGyroMetadataLocked(index: frameCount, elapsedSeconds: elapsedSeconds)
                    frameCount += 1
                } else {
                    droppedFrames += 1
                    break
                }
            }
        }

        // Drain any pending audio buffers now that video input might have made room
        drainPendingAudioBuffersLocked()

        // VideoToolbox backpressure handling:
        // Hardware encoder occasionally takes 1-4ms to complete a GOP slice.
        // Spin-wait briefly outside lock before discarding to eliminate spurious frame drops.
        var spinAttempts = 0
        while !input.isReadyForMoreMediaData && spinAttempts < 5 {
            lock.unlock()
            usleep(2000) // 2ms sleep outside lock
            lock.lock()
            guard isRecording, videoInput != nil, pixelBufferAdaptor != nil else {
                droppedFrames += 1
                return false
            }
            spinAttempts += 1
        }

        // Current real frame
        guard input.isReadyForMoreMediaData else {
            lastPixelBuffer = pixelBuffer
            droppedFrames += 1
            return false
        }

        if writeCFR(pixelBuffer, index: frameCount, adaptor: adaptor) {
            appendGyroMetadataLocked(index: frameCount, elapsedSeconds: elapsedSeconds)
            frameCount += 1
            realFrameCount += 1
            lastPixelBuffer = pixelBuffer
            lastFrameHostTime = now
            if let captureTime = captureTime, captureTime.isValid {
                lastFrameCaptureTime = captureTime
            }
            return true
        } else {
            droppedFrames += 1
            return false
        }
    }

    private func appendGyroMetadataLocked(index: Int64, elapsedSeconds: Double) {
        guard let mInput = metadataInput,
              mInput.isReadyForMoreMediaData,
              let formatDesc = metadataFormatDesc,
              let sample = gyroRecorder.latestSample(at: elapsedSeconds) else { return }

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
        guard bbStatus == noErr, let bb = blockBuffer else { return }

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

        let pts = CMTime(value: index * 1000, timescale: CMTimeScale(targetFPS * 1000.0))
        let duration = CMTime(value: 1000, timescale: CMTimeScale(targetFPS * 1000.0))
        var timing = CMSampleTimingInfo(duration: duration, presentationTimeStamp: pts, decodeTimeStamp: .invalid)

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
        guard sbStatus == noErr, let sampleBuffer else { return }
        _ = mInput.append(sampleBuffer)
    }

    private func writeCFR(
        _ pixelBuffer: CVPixelBuffer,
        index: Int64,
        adaptor: AVAssetWriterInputPixelBufferAdaptor
    ) -> Bool {
        // Exact CFR: frame i at t = i / fps
        let pts = CMTime(value: index * 1000, timescale: CMTimeScale(targetFPS * 1000.0))
        return adaptor.append(pixelBuffer, withPresentationTime: pts)
    }

    @discardableResult
    func appendAudio(sampleBuffer: CMSampleBuffer) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard isRecording,
              let input = audioInput else {
            return false
        }
        guard CMSampleBufferDataIsReady(sampleBuffer) else { return false }

        // If session has not started yet (waiting for Frame 0), buffer pre-roll audio
        if !hasStartedSession || realFrameCount == 0 {
            if prerollAudioBuffers.count < maxPrerollAudioBuffers {
                prerollAudioBuffers.append(sampleBuffer)
            } else {
                prerollAudioBuffers.removeFirst()
                prerollAudioBuffers.append(sampleBuffer)
            }
            return true
        }

        // Drain previously queued audio buffers first if input is ready
        drainPendingAudioBuffersLocked()

        var singleTiming = CMSampleTimingInfo(
            duration: .invalid,
            presentationTimeStamp: .invalid,
            decodeTimeStamp: .invalid
        )
        var timingCount: CMItemCount = 0
        CMSampleBufferGetSampleTimingInfoArray(sampleBuffer, entryCount: 1, arrayToFill: &singleTiming, entriesNeededOut: &timingCount)
        guard timingCount > 0 else { return false }

        // Fast path: eliminate 50-100 heap array allocations/sec for standard 1-timing AVCapture audio buffers
        if timingCount == 1 {
            let baseTime = sessionStartTime.isValid ? sessionStartTime : audioReferenceTime
            if !baseTime.isValid {
                audioReferenceTime = singleTiming.presentationTimeStamp
            }
            let refTime = sessionStartTime.isValid ? sessionStartTime : audioReferenceTime

            if sessionStartTime.isValid, singleTiming.duration.isValid {
                let bufferEnd = CMTimeAdd(singleTiming.presentationTimeStamp, singleTiming.duration)
                if CMTimeCompare(bufferEnd, refTime) <= 0 {
                    return true
                }
            }

            var pts = CMTimeSubtract(singleTiming.presentationTimeStamp, refTime)
            if CMTimeCompare(pts, .zero) < 0 { pts = .zero }
            singleTiming.presentationTimeStamp = pts
            if singleTiming.decodeTimeStamp.isValid {
                singleTiming.decodeTimeStamp = pts
            }

            var retimed: CMSampleBuffer?
            let status = CMSampleBufferCreateCopyWithNewTiming(
                allocator: kCFAllocatorDefault,
                sampleBuffer: sampleBuffer,
                sampleTimingEntryCount: 1,
                sampleTimingArray: &singleTiming,
                sampleBufferOut: &retimed
            )
            guard status == noErr, let retimed else { return false }

            if input.isReadyForMoreMediaData && pendingAudioBuffers.isEmpty {
                return input.append(retimed)
            } else {
                if pendingAudioBuffers.count < maxPendingAudioBuffers {
                    pendingAudioBuffers.append(retimed)
                    return true
                } else {
                    pendingAudioBuffers.removeFirst()
                    pendingAudioBuffers.append(retimed)
                    return false
                }
            }
        }

        var timings = Array(repeating: CMSampleTimingInfo(
            duration: .invalid,
            presentationTimeStamp: .invalid,
            decodeTimeStamp: .invalid
        ), count: timingCount)
        CMSampleBufferGetSampleTimingInfoArray(sampleBuffer, entryCount: timingCount, arrayToFill: &timings, entriesNeededOut: &timingCount)

        // Anchor audio timeline to the first optical frame's capture time (sessionStartTime)
        // or fallback to first audio PTS. This ensures microsecond lip-sync alignment with video.
        let baseTime = sessionStartTime.isValid ? sessionStartTime : audioReferenceTime
        if !baseTime.isValid {
            audioReferenceTime = timings[0].presentationTimeStamp
        }
        let refTime = sessionStartTime.isValid ? sessionStartTime : audioReferenceTime

        // If this audio buffer finished completely before session start, skip pre-roll
        if sessionStartTime.isValid, timings[0].duration.isValid {
            let bufferEnd = CMTimeAdd(timings[0].presentationTimeStamp, timings[0].duration)
            if CMTimeCompare(bufferEnd, refTime) <= 0 {
                return true
            }
        }

        for i in 0..<timings.count {
            var pts = CMTimeSubtract(timings[i].presentationTimeStamp, refTime)
            if CMTimeCompare(pts, .zero) < 0 {
                pts = .zero
            }
            timings[i].presentationTimeStamp = pts
            if timings[i].decodeTimeStamp.isValid {
                timings[i].decodeTimeStamp = pts
            }
        }

        var retimed: CMSampleBuffer?
        let status = CMSampleBufferCreateCopyWithNewTiming(
            allocator: kCFAllocatorDefault,
            sampleBuffer: sampleBuffer,
            sampleTimingEntryCount: timings.count,
            sampleTimingArray: &timings,
            sampleBufferOut: &retimed
        )
        guard status == noErr, let retimed else { return false }

        if input.isReadyForMoreMediaData && pendingAudioBuffers.isEmpty {
            return input.append(retimed)
        } else {
            // Buffer temporarily during video encode backpressure to avoid sample gaps/static
            if pendingAudioBuffers.count < maxPendingAudioBuffers {
                pendingAudioBuffers.append(retimed)
                return true
            } else {
                // Buffer capacity reached (long stall): drop oldest to maintain real-time queue
                pendingAudioBuffers.removeFirst()
                pendingAudioBuffers.append(retimed)
                return false
            }
        }
    }

    func finish(completion: @Sendable @escaping (URL?, Error?) -> Void) {
        lock.lock()
        guard isRecording else {
            lock.unlock()
            completion(nil, NSError(domain: "OwLens", code: 100, userInfo: [NSLocalizedDescriptionKey: "Recording was not active"]))
            return
        }

        // Bounded pad to wall clock: pad hold frames to match audio duration cleanly
        if let hold = lastPixelBuffer,
           let adaptor = pixelBufferAdaptor,
           let vIn = videoInput {
            let elapsed = max(0, CACurrentMediaTime() - startHostTime)
            let targetCount = min(Int64((elapsed * targetFPS).rounded()), frameCount + 15)
            while frameCount < targetCount && vIn.isReadyForMoreMediaData {
                if writeCFR(hold, index: frameCount, adaptor: adaptor) {
                    frameCount += 1
                } else {
                    break
                }
            }
        }

        // Drain any remaining buffered audio
        drainPendingAudioBuffersLocked()
        pendingAudioBuffers.removeAll()
        prerollAudioBuffers.removeAll()

        let url = assetWriter?.outputURL
        isRecording = false
        audioReferenceTime = .invalid
        sessionStartTime = .invalid
        hasStartedSession = false
        lastPixelBuffer = nil
        let writer = assetWriter
        let vIn = videoInput
        let aIn = audioInput
        let mIn = metadataInput
        let total = frameCount
        let real = realFrameCount
        let drops = droppedFrames
        let fps = targetFPS
        lock.unlock()

        vIn?.markAsFinished()
        aIn?.markAsFinished()
        mIn?.markAsFinished()

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
            writer.cancelWriting()
            gyroRecorder.stop()
            if let url {
                try? FileManager.default.removeItem(at: url)
            }
            completion(nil, NSError(domain: "OwLens", code: 102, userInfo: [NSLocalizedDescriptionKey: "Recording ended with no frames"]))
            return
        }
        let boxedWriter = SendableBox(value: writer)
        writer.finishWriting {
            let status = boxedWriter.value.status
            let duration = Double(total) / fps
            print("[VideoWriter] Done. timeline=\(total) real=\(real) holds=\(total - real) drops=\(drops) \(String(format: "%.2f", duration))s @ \(Int(fps))fps status=\(String(describing: status))")
            let outputURL = boxedWriter.value.outputURL
            let gcsvData = self.gyroRecorder.exportGCSVData(videoFileName: outputURL.lastPathComponent)
            self.gyroRecorder.stop()
            if status == .failed {
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
    private static func patchMebxToCamm(at url: URL) {
        guard let fileHandle = try? FileHandle(forUpdating: url) else { return }
        defer { try? fileHandle.close() }

        let fileSize = fileHandle.seekToEndOfFile()
        guard fileSize > 64 else { return }

        let scanLength = min(fileSize, 5 * 1024 * 1024)
        let scanOffset = fileSize - scanLength
        fileHandle.seek(toFileOffset: scanOffset)
        let buffer = fileHandle.readData(ofLength: Int(scanLength))

        let mebxBytes = Data("mebx".utf8)
        let cammBytes = Data("camm".utf8)
        let stsdBytes = Data("stsd".utf8)

        if let stsdRange = buffer.range(of: stsdBytes) {
            let searchWindow = Range(uncheckedBounds: (
                lower: stsdRange.upperBound,
                upper: min(buffer.count, stsdRange.upperBound + 64)
            ))
            if let mebxRange = buffer.range(of: mebxBytes, options: [], in: searchWindow) {
                let filePatchOffset = scanOffset + UInt64(mebxRange.lowerBound)
                fileHandle.seek(toFileOffset: filePatchOffset)
                fileHandle.write(cammBytes)
                print("[VideoWriter] Successfully patched stsd entry 'mebx' -> 'camm' at offset \(filePatchOffset)")
                return
            }
        }

        if let mebxRange = buffer.range(of: mebxBytes) {
            let filePatchOffset = scanOffset + UInt64(mebxRange.lowerBound)
            fileHandle.seek(toFileOffset: filePatchOffset)
            fileHandle.write(cammBytes)
            print("[VideoWriter] Fallback patched 'mebx' -> 'camm' at offset \(filePatchOffset)")
        }
    }

    private static func makeMetadataItems(
        curveType: LogCurveType,
        fps: Double,
        width: Int,
        height: Int
    ) -> [AVMetadataItem] {
        var items: [AVMetadataItem] = []

        // 1. Common Description
        let desc = AVMutableMetadataItem()
        desc.keySpace = .common
        desc.key = AVMetadataKey.commonKeyDescription as (NSCopying & NSObjectProtocol)
        desc.value = "\(curveType.displayName) · S-Gamut3.Cine · 10-Bit" as (NSCopying & NSObjectProtocol)
        items.append(desc)

        // 2. QuickTime User Data Description
        let qtDesc = AVMutableMetadataItem()
        qtDesc.keySpace = .quickTimeUserData
        qtDesc.key = AVMetadataKey.quickTimeUserDataKeyDescription as (NSCopying & NSObjectProtocol)
        qtDesc.value = "\(curveType.displayName) · S-Gamut3.Cine · 10-Bit" as (NSCopying & NSObjectProtocol)
        items.append(qtDesc)

        // 3. QuickTime Software
        let software = AVMutableMetadataItem()
        software.keySpace = .quickTimeUserData
        software.key = AVMetadataKey.quickTimeUserDataKeySoftware as (NSCopying & NSObjectProtocol)
        software.value = "OwLens Cinema Camera v1.6" as (NSCopying & NSObjectProtocol)
        items.append(software)

        // 4. QuickTime Make
        let make = AVMutableMetadataItem()
        make.keySpace = .quickTimeUserData
        make.key = AVMetadataKey.quickTimeUserDataKeyMake as (NSCopying & NSObjectProtocol)
        make.value = "Apple / Sony S-Log3" as (NSCopying & NSObjectProtocol)
        items.append(make)

        // 5. QuickTime Model / Profile
        let model = AVMutableMetadataItem()
        model.keySpace = .quickTimeUserData
        model.key = AVMetadataKey.quickTimeUserDataKeyModel as (NSCopying & NSObjectProtocol)
        model.value = "S-Log3 / S-Gamut3.Cine 10-Bit" as (NSCopying & NSObjectProtocol)
        items.append(model)

        // 6. Detailed Comment for DaVinci Resolve / FCP NLEs
        let comment = AVMutableMetadataItem()
        comment.keySpace = .quickTimeUserData
        comment.key = AVMetadataKey.quickTimeUserDataKeyComment as (NSCopying & NSObjectProtocol)
        comment.value = "Color Space: Sony S-Gamut3.Cine | Gamma: Sony S-Log3 | Frame Rate: \(Int(fps)) fps | Format: \(width)x\(height) 10-Bit HEVC" as (NSCopying & NSObjectProtocol)
        items.append(comment)

        return items
    }
}
