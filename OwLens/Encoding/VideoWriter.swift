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
    private var targetFPS: Double = 24
    private var startHostTime: CFTimeInterval = 0
    private var audioReferenceTime: CMTime = .invalid
    private var sessionStartTime: CMTime = .invalid
    private var lastVideoPTS: CMTime = .invalid
    private var lastAudioPTS: CMTime = .invalid
    private var lastAudioEndPTS: CMTime = .invalid
    private var lastFrameCaptureTime: CMTime = .invalid
    private var lastFrameHostTime: CFTimeInterval = 0
    private var hasStartedSession = false
    private var lastPixelBuffer: CVPixelBuffer?
    private var pendingAudioBuffers: [CMSampleBuffer] = []
    private let maxPendingAudioBuffers = 50
    private var prerollAudioBuffers: [CMSampleBuffer] = []
    private let maxPrerollAudioBuffers = 60
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

        // Target frame rate (defaults to cinema 24 fps)
        let fps = targetFPS > 0 ? targetFPS : 24.0

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

        // Frames are encoded natively upright by MetalPipeline (with 180° sensor
        // inversion applied when shooting in Landscape Left). We preserve track transform identity
        // so all players (QuickTime, DaVinci Resolve on Mac/Win, Premiere, VLC, FCP)
        // play the true raster upright without depending on container matrix interpretation.
        vInput.transform = .identity

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
        self.lastWrittenSlot = -1
        self.droppedFrames = 0
        self.startHostTime = 0
        self.sessionStartTime = .invalid
        self.lastVideoPTS = .invalid
        self.lastAudioPTS = .invalid
        self.lastAudioEndPTS = .invalid
        self.hasStartedSession = false
        self.lastPixelBuffer = nil
        self.audioReferenceTime = .invalid
        self.prerollAudioBuffers.removeAll()
        self.pendingAudioBuffers.removeAll()
        self.lastDiskCheckTime = 0
        self.cachedHasSpace = true
        self.lastFrameCaptureTime = .invalid
        self.lastFrameHostTime = 0
        self.isRecording = true

        print("[VideoWriter] CFR \(Int(fps))fps \(width)x\(height) codec=\(codec.displayName) bitrate=\(bitrate) orientation=\(orientation.rawValue)")
    }

    private func drainPendingAudioBuffersLocked(upTo maxPTS: CMTime? = nil) {
        guard let input = audioInput else { return }
        var drainedCount = 0
        for buffer in pendingAudioBuffers {
            let pts = CMSampleBufferGetPresentationTimeStamp(buffer)
            if let maxPTS = maxPTS, maxPTS.isValid, CMTimeCompare(pts, maxPTS) > 0 {
                break
            }
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
        guard let input = audioInput, hasStartedSession, sessionStartTime.isValid, realFrameCount > 0 else {
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
            } else if CMTimeCompare(timings[0].presentationTimeStamp, refTime) < 0 {
                continue
            }

            for i in 0..<timings.count {
                let rawPTS = CMTimeSubtract(timings[i].presentationTimeStamp, refTime)
                let bufferDur = timings[i].duration.isValid ? timings[i].duration : CMTime(value: 1024, timescale: 48000)
                let pts: CMTime
                if self.lastAudioEndPTS.isValid {
                    let delta = CMTimeGetSeconds(CMTimeSubtract(rawPTS, self.lastAudioEndPTS))
                    if abs(delta) < 0.050 {
                        pts = self.lastAudioEndPTS
                    } else if CMTimeCompare(rawPTS, self.lastAudioEndPTS) < 0 {
                        pts = self.lastAudioEndPTS
                    } else {
                        pts = rawPTS
                    }
                } else {
                    // Initial audio buffer: anchor strictly to max(.zero, rawPTS)
                    // If buffer starts within 25ms of Frame 0, snap precisely to .zero
                    let rawSec = CMTimeGetSeconds(rawPTS)
                    if rawSec <= 0.025 {
                        pts = .zero
                    } else {
                        pts = rawPTS
                    }
                }

                timings[i].presentationTimeStamp = pts
                self.lastAudioPTS = pts
                self.lastAudioEndPTS = CMTimeAdd(pts, bufferDur)
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
            // VideoToolbox initialization on Frame 0 can take 10-30ms.
            // Patient spin-wait (up to 150ms) ensures Frame 0 is NEVER dropped.
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

            // Start the asset writer session at source time .zero
            assetWriter?.startSession(atSourceTime: .zero)
            let firstPTS = CMTime(value: 0, timescale: CMTimeScale(round(targetFPS * 1000.0)))
            guard adaptor.append(pixelBuffer, withPresentationTime: firstPTS) else {
                print("[VideoWriter] Failed to append Frame 0 to pixelBufferAdaptor: \(String(describing: assetWriter?.error))")
                droppedFrames += 1
                return false
            }

            // Frame 0 successfully committed at PTS .zero!
            hasStartedSession = true
            startHostTime = now
            if let captureTime = captureTime, captureTime.isValid {
                sessionStartTime = captureTime
                lastFrameCaptureTime = captureTime
            } else {
                sessionStartTime = CMTime(seconds: now, preferredTimescale: 24000)
                lastFrameCaptureTime = sessionStartTime
            }
            lastFrameHostTime = now
            lastVideoPTS = firstPTS
            lastWrittenSlot = 0
            frameCount = 1
            realFrameCount = 1
            lastPixelBuffer = pixelBuffer

            let anchorTime = (captureTime != nil && captureTime!.isValid) ? CMTimeGetSeconds(captureTime!) : now
            gyroRecorder.anchorSession(startHostTime: anchorTime)
            appendGyroMetadataLocked(elapsedSeconds: 0)

            // DRAIN PREROLL AUDIO NOW: Video Frame 0 is committed at PTS .zero, so audio begins precisely at .zero
            drainPrerollAudioBuffersLocked()
            drainPendingAudioBuffersLocked()
            return true
        }

        // Calculate wall-clock elapsed time since optical session start
        let elapsed: Double
        if let captureTime = captureTime, captureTime.isValid, sessionStartTime.isValid {
            let optical = CMTimeGetSeconds(CMTimeSubtract(captureTime, sessionStartTime))
            let wallClock = now - startHostTime
            // Protect against clock-domain jumps or un-anchored capture timestamps
            if optical >= 0 && abs(optical - wallClock) < 1.0 {
                elapsed = optical
            } else {
                elapsed = max(0, wallClock)
            }
        } else {
            elapsed = max(0, now - startHostTime)
        }

        // Strict Constant Frame Rate (CFR):
        // Each frame s is mathematically presented at t = s / targetFPS.
        // targetSlot is the nearest 24.000 FPS grid slot corresponding to elapsed optical time.
        let targetSlot = Int64(round(elapsed * targetFPS))

        // If targetSlot has already been written, this frame is an intra-slot burst/duplicate.
        // Drop it so subsequent frames are NEVER shifted forward in time.
        if targetSlot <= lastWrittenSlot {
            lastPixelBuffer = pixelBuffer
            droppedFrames += 1
            return false
        }

        // Drain any pending audio buffers now that video input might have made room
        drainPendingAudioBuffersLocked()

        // VideoToolbox backpressure handling:
        // Hardware encoder occasionally takes 1-4ms to complete a GOP slice.
        // Spin-wait briefly outside lock before discarding to eliminate spurious frame drops.
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

        // Fill any missing CFR slots by repeating last frame (up to 12 frames max / 0.5s).
        // This ensures the video container maintains strict CFR 24.0000 FPS without VFR warnings,
        // while guaranteeing that audio and video remain in microsecond sync over long recordings!
        if let hold = lastPixelBuffer, targetSlot > lastWrittenSlot + 1 {
            let holdLimit = min(targetSlot, lastWrittenSlot + 13)
            var slotToFill = lastWrittenSlot + 1
            while slotToFill < holdLimit {
                var holdSpin = 0
                while !input.isReadyForMoreMediaData && holdSpin < 5 {
                    lock.unlock()
                    usleep(1500)
                    lock.lock()
                    guard isRecording, videoInput != nil, pixelBufferAdaptor != nil else { break }
                    holdSpin += 1
                }
                guard input.isReadyForMoreMediaData else {
                    droppedFrames += 1
                    break
                }
                let holdPTS = CMTime(value: slotToFill * 1000, timescale: CMTimeScale(round(targetFPS * 1000.0)))
                let holdElapsed = Double(slotToFill) / targetFPS
                if adaptor.append(hold, withPresentationTime: holdPTS) {
                    lastVideoPTS = holdPTS
                    appendGyroMetadataLocked(elapsedSeconds: holdElapsed)
                    lastWrittenSlot = slotToFill
                    frameCount = slotToFill + 1
                    slotToFill += 1
                } else {
                    droppedFrames += 1
                    break
                }
            }
        }

        // Current real frame
        guard input.isReadyForMoreMediaData else {
            lastPixelBuffer = pixelBuffer
            droppedFrames += 1
            return false
        }

        let currentPTS = CMTime(value: targetSlot * 1000, timescale: CMTimeScale(round(targetFPS * 1000.0)))
        let currentElapsed = Double(targetSlot) / targetFPS

        if adaptor.append(pixelBuffer, withPresentationTime: currentPTS) {
            lastVideoPTS = currentPTS
            appendGyroMetadataLocked(elapsedSeconds: currentElapsed)
            lastWrittenSlot = targetSlot
            frameCount = targetSlot + 1
            realFrameCount += 1
            lastPixelBuffer = pixelBuffer
            lastFrameHostTime = now
            if let captureTime = captureTime, captureTime.isValid {
                lastFrameCaptureTime = captureTime
            }
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

            if sessionStartTime.isValid {
                if singleTiming.duration.isValid {
                    let bufferEnd = CMTimeAdd(singleTiming.presentationTimeStamp, singleTiming.duration)
                    if CMTimeCompare(bufferEnd, refTime) <= 0 {
                        return true
                    }
                } else if CMTimeCompare(singleTiming.presentationTimeStamp, refTime) < 0 {
                    return true
                }
            }

            let rawPTS = CMTimeSubtract(singleTiming.presentationTimeStamp, refTime)
            let bufferDur = singleTiming.duration.isValid ? singleTiming.duration : CMTime(value: 1024, timescale: 48000)
            let pts: CMTime
            if lastAudioEndPTS.isValid {
                let delta = CMTimeGetSeconds(CMTimeSubtract(rawPTS, lastAudioEndPTS))
                if abs(delta) < 0.050 {
                    pts = lastAudioEndPTS
                } else if CMTimeCompare(rawPTS, lastAudioEndPTS) < 0 {
                    pts = lastAudioEndPTS
                } else {
                    pts = rawPTS
                }
            } else {
                let rawSec = CMTimeGetSeconds(rawPTS)
                if rawSec <= 0.025 {
                    pts = .zero
                } else {
                    pts = rawPTS
                }
            }
            singleTiming.presentationTimeStamp = pts
            lastAudioPTS = pts
            lastAudioEndPTS = CMTimeAdd(pts, bufferDur)
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

        // If this audio buffer finished or started before session start, skip pre-roll
        if sessionStartTime.isValid {
            if timings[0].duration.isValid {
                let bufferEnd = CMTimeAdd(timings[0].presentationTimeStamp, timings[0].duration)
                if CMTimeCompare(bufferEnd, refTime) <= 0 {
                    return true
                }
            } else if CMTimeCompare(timings[0].presentationTimeStamp, refTime) < 0 {
                return true
            }
        }

        for i in 0..<timings.count {
            let rawPTS = CMTimeSubtract(timings[i].presentationTimeStamp, refTime)
            let bufferDur = timings[i].duration.isValid ? timings[i].duration : CMTime(value: 1024, timescale: 48000)
            let pts: CMTime
            if lastAudioEndPTS.isValid {
                let delta = CMTimeGetSeconds(CMTimeSubtract(rawPTS, lastAudioEndPTS))
                if abs(delta) < 0.050 {
                    pts = lastAudioEndPTS
                } else if CMTimeCompare(rawPTS, lastAudioEndPTS) < 0 {
                    pts = lastAudioEndPTS
                } else {
                    pts = rawPTS
                }
            } else {
                let rawSec = CMTimeGetSeconds(rawPTS)
                if rawSec <= 0.025 {
                    pts = .zero
                } else {
                    pts = rawPTS
                }
            }
            timings[i].presentationTimeStamp = pts
            lastAudioPTS = pts
            lastAudioEndPTS = CMTimeAdd(pts, bufferDur)
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

        isRecording = false

        // Tail freeze elimination:
        // Video track duration ends at lastVideoPTS + 1 nominal frame.
        // Drain pending audio buffers strictly up to videoEndTime and discard any trailing audio
        // that extends beyond videoEndTime so the audio track NEVER outlasts the video track
        // (which causes video players to freeze on the final frame).
        let nominalFrameDuration = CMTime(value: 1000, timescale: CMTimeScale(targetFPS * 1000.0))
        let videoEndTime: CMTime
        if lastVideoPTS.isValid {
            videoEndTime = CMTimeAdd(lastVideoPTS, nominalFrameDuration)
        } else if frameCount > 0 {
            videoEndTime = CMTime(value: frameCount * 1000, timescale: CMTimeScale(targetFPS * 1000.0))
        } else {
            videoEndTime = .invalid
        }

        if let _ = audioInput, videoEndTime.isValid {
            drainPendingAudioBuffersLocked(upTo: videoEndTime)
        }
        pendingAudioBuffers.removeAll()
        prerollAudioBuffers.removeAll()

        let url = assetWriter?.outputURL
        isRecording = false
        audioReferenceTime = .invalid
        sessionStartTime = .invalid
        lastVideoPTS = .invalid
        lastAudioPTS = .invalid
        lastAudioEndPTS = .invalid
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
        let fps = targetFPS
        let videoDurationSeconds = (videoEndTime.isValid && videoEndTime.seconds > 0) ? videoEndTime.seconds : (Double(total) / fps)

        appendGyroMetadataLocked(elapsedSeconds: videoDurationSeconds)

        if let writer = assetWriter, writer.status == .writing, videoEndTime.isValid {
            writer.endSession(atSourceTime: videoEndTime)
        }

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
            let duration = videoDurationSeconds
            print("[VideoWriter] Done. frames=\(real) drops=\(drops) \(String(format: "%.2f", duration))s @ \(Int(fps))fps status=\(String(describing: status))")
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

        let mebxBytes = Data("mebx".utf8)
        let cammBytes = Data("camm".utf8)
        let stsdBytes = Data("stsd".utf8)

        var regionsToScan: [(offset: UInt64, length: Int)] = []
        let headLength = Int(min(fileSize, 2 * 1024 * 1024))
        regionsToScan.append((offset: 0, length: headLength))

        let tailLength = Int(min(fileSize, 8 * 1024 * 1024))
        let tailOffset = fileSize - UInt64(tailLength)
        if tailOffset > UInt64(headLength) {
            regionsToScan.append((offset: tailOffset, length: tailLength))
        }

        for region in regionsToScan {
            fileHandle.seek(toFileOffset: region.offset)
            let buffer = fileHandle.readData(ofLength: region.length)
            var searchStart = 0

            while searchStart < buffer.count {
                guard let stsdRange = buffer.range(of: stsdBytes, options: [], in: searchStart..<buffer.count) else {
                    break
                }
                let windowEnd = min(buffer.count, stsdRange.upperBound + 64)
                let searchWindow = stsdRange.upperBound..<windowEnd
                if let mebxRange = buffer.range(of: mebxBytes, options: [], in: searchWindow) {
                    let filePatchOffset = region.offset + UInt64(mebxRange.lowerBound)
                    fileHandle.seek(toFileOffset: filePatchOffset)
                    fileHandle.write(cammBytes)
                    print("[VideoWriter] Successfully patched stsd entry 'mebx' -> 'camm' at offset \(filePatchOffset)")
                    return
                }
                searchStart = stsdRange.upperBound
            }
        }
    }

    private static func makeMetadataItems(
        curveType: LogCurveType,
        fps: Double,
        width: Int,
        height: Int
    ) -> [AVMetadataItem] {
        var items: [AVMetadataItem] = []

        let gamut: String
        switch curveType {
        case .appleLog2:
            gamut = "ITU-R BT.2020"
        case .sLog3Approx:
            gamut = "Sony S-Gamut3.Cine"
        case .linear:
            gamut = "Rec.709"
        }
        let gamma = curveType.displayName

        // 1. Common Description
        let desc = AVMutableMetadataItem()
        desc.keySpace = .common
        desc.key = AVMetadataKey.commonKeyDescription as (NSCopying & NSObjectProtocol)
        desc.value = "\(gamma) · \(gamut) · 10-Bit" as (NSCopying & NSObjectProtocol)
        items.append(desc)

        // 2. QuickTime User Data Description
        let qtDesc = AVMutableMetadataItem()
        qtDesc.keySpace = .quickTimeUserData
        qtDesc.key = AVMetadataKey.quickTimeUserDataKeyDescription as (NSCopying & NSObjectProtocol)
        qtDesc.value = "\(gamma) · \(gamut) · 10-Bit" as (NSCopying & NSObjectProtocol)
        items.append(qtDesc)

        // 3. QuickTime Software
        let software = AVMutableMetadataItem()
        software.keySpace = .quickTimeUserData
        software.key = AVMetadataKey.quickTimeUserDataKeySoftware as (NSCopying & NSObjectProtocol)
        software.value = "OwLens Cinema Camera v2.1.5" as (NSCopying & NSObjectProtocol)
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
        comment.value = "Color Space: \(gamut) | Gamma: \(gamma) | Frame Rate: \(Int(fps)) fps | Format: \(width)x\(height) 10-Bit HEVC" as (NSCopying & NSObjectProtocol)
        items.append(comment)

        return items
    }
}
