import AVFoundation
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
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var frameCount: Int64 = 0
    private var realFrameCount: Int64 = 0
    private var width: Int = 0
    private var height: Int = 0
    private var targetFPS: Double = 24
    private var startHostTime: CFTimeInterval = 0
    private var audioReferenceTime: CMTime = .invalid
    private var sessionStartTime: CMTime = .invalid
    private var hasStartedSession = false
    private var lastPixelBuffer: CVPixelBuffer?
    private var pendingAudioBuffers: [CMSampleBuffer] = []
    private let maxPendingAudioBuffers = 50
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
        case .sLog3Approx, .appleLog2:
            // For log curves (Apple Log and S-Log3 in BT.2020 container), omitting explicit
            // AVVideoColorPropertiesKey from outputSettings allows VideoToolbox to derive
            // the exact 10-bit track tagging (BT.2020 primaries + matrix, plus Apple Log transfer
            // function where applicable) directly from the CVPixelBuffer attachments.
            // This prevents S-Log3 from being falsely tagged with Rec.709 transfer function.
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
                kCVPixelBufferMetalCompatibilityKey as String: true
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
                AVNumberOfChannelsKey: 1,
                AVEncoderBitRateKey: 128_000
            ]
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
            input.expectsMediaDataInRealTime = true
            if writer.canAdd(input) {
                writer.add(input)
                aInput = input
            }
        }

        guard writer.startWriting() else {
            throw writer.error ?? NSError(domain: "RawLogCam", code: 11, userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter failed to start"])
        }

        self.assetWriter = writer
        self.videoInput = vInput
        self.audioInput = aInput
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

    private func hasSufficientDiskSpace() -> Bool {
        guard let outputURL = assetWriter?.outputURL else { return true }
        do {
            let values = try outputURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            if let available = values.volumeAvailableCapacityForImportantUsage {
                // If less than 500 MB left, notify handler to stop recording safely and refuse further frames
                if available <= 500 * 1024 * 1024 {
                    onLowDiskSpace?()
                    return false
                }
                return true
            }
        } catch {}
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

        // Attach color space & transfer characteristics to pixel buffer for VideoToolbox encoding
        MetalPipeline.attachColorMetadata(to: pixelBuffer, curveType: curveType)

        let now = CACurrentMediaTime()

        if !hasStartedSession {
            startHostTime = now
            if let captureTime = captureTime, captureTime.isValid {
                sessionStartTime = captureTime
            }
            assetWriter?.startSession(atSourceTime: .zero)
            hasStartedSession = true
        }

        var elapsedSeconds: Double
        if let captureTime = captureTime, captureTime.isValid, sessionStartTime.isValid {
            elapsedSeconds = max(0, CMTimeSubtract(captureTime, sessionStartTime).seconds)
        } else {
            elapsedSeconds = max(0, now - startHostTime)
        }

        // Startup grace: If startup delays (pool allocation, camera mode lock, or initial pipeline spin-up)
        // caused a gap after the very first frame or before the second frame, do NOT inject a flurry of hold frames!
        // Hold frames are only for bridging mid-stream frame drops, not for startup stalls.
        // If continuous motion has not been established yet (realFrameCount <= 1) and elapsedSeconds exceeds
        // 1.5 frame durations, re-anchor sessionStartTime and startHostTime to this frame, resetting elapsedSeconds to 0.
        if realFrameCount <= 1 && elapsedSeconds > (1.5 / targetFPS) {
            if let captureTime = captureTime, captureTime.isValid {
                sessionStartTime = captureTime
            }
            startHostTime = now
            elapsedSeconds = 0.0
        }

        // How many CFR frames should exist by this capture time.
        // Tolerates up to 0.40 frame duration of optical timestamp jitter before inserting a hold frame,
        // preventing premature duplicate frames from normal sensor readout latency variations.
        let frameSlot = Int64(floor(elapsedSeconds * targetFPS + 0.40))
        let wallTargetCount = frameSlot + 1
        // Always advance at least one slot for this real frame
        let targetCount = max(frameCount + 1, wallTargetCount)

        // Hold last real frame for skipped slots to maintain strict CFR duration
        if let hold = lastPixelBuffer {
            let maxHold = min(targetCount - 1, frameCount + 10)
            while frameCount < maxHold {
                guard input.isReadyForMoreMediaData else {
                    droppedFrames += 1
                    break
                }
                if writeCFR(hold, index: frameCount, adaptor: adaptor) {
                    frameCount += 1
                } else {
                    droppedFrames += 1
                    break
                }
            }
        }

        // Drain any pending audio buffers now that video input might have made room
        drainPendingAudioBuffersLocked()

        // Current real frame
        guard input.isReadyForMoreMediaData else {
            lastPixelBuffer = pixelBuffer
            droppedFrames += 1
            return false
        }

        if writeCFR(pixelBuffer, index: frameCount, adaptor: adaptor) {
            frameCount += 1
            realFrameCount += 1
            lastPixelBuffer = pixelBuffer
            return true
        } else {
            droppedFrames += 1
            return false
        }
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
              let input = audioInput,
              hasStartedSession,
              realFrameCount >= 2 else {
            return false
        }
        guard CMSampleBufferDataIsReady(sampleBuffer) else { return false }

        // Drain previously queued audio buffers first if input is ready
        drainPendingAudioBuffersLocked()

        var timingCount: CMItemCount = 0
        CMSampleBufferGetSampleTimingInfoArray(sampleBuffer, entryCount: 0, arrayToFill: nil, entriesNeededOut: &timingCount)
        guard timingCount > 0 else { return false }

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

    func finish(completion: @Sendable @escaping (URL?) -> Void) {
        lock.lock()
        guard isRecording else {
            lock.unlock()
            completion(nil)
            return
        }

        // Bounded pad to wall clock: pad at most 4 hold frames to match audio duration cleanly
        if let hold = lastPixelBuffer,
           let adaptor = pixelBufferAdaptor,
           let vIn = videoInput {
            let elapsed = max(0, CACurrentMediaTime() - startHostTime)
            let targetCount = min(Int64((elapsed * targetFPS).rounded()), frameCount + 4)
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

        let url = assetWriter?.outputURL
        isRecording = false
        audioReferenceTime = .invalid
        sessionStartTime = .invalid
        hasStartedSession = false
        lastPixelBuffer = nil
        let writer = assetWriter
        let vIn = videoInput
        let aIn = audioInput
        let total = frameCount
        let real = realFrameCount
        let drops = droppedFrames
        let fps = targetFPS
        lock.unlock()

        vIn?.markAsFinished()
        aIn?.markAsFinished()

        // `writer` (AVAssetWriter) is non-Sendable; box it so the @Sendable
        // finishWriting callback can capture it. If there is no writer there is
        // nothing to finalize — report failure instead of silently never
        // invoking completion (which would hang the caller).
        guard let writer else {
            completion(nil)
            return
        }

        if real == 0 {
            print("[VideoWriter] No real frames were appended to the video timeline.")
            writer.cancelWriting()
            if let url {
                try? FileManager.default.removeItem(at: url)
            }
            completion(nil)
            return
        }
        let boxedWriter = SendableBox(value: writer)
        writer.finishWriting {
            let status = boxedWriter.value.status
            let duration = Double(total) / fps
            print("[VideoWriter] Done. timeline=\(total) real=\(real) holds=\(total - real) drops=\(drops) \(String(format: "%.2f", duration))s @ \(Int(fps))fps status=\(String(describing: status))")
            if status == .failed {
                print("[VideoWriter] Error: \(String(describing: boxedWriter.value.error))")
                completion(nil)
            } else {
                completion(url)
            }
        }
    }
}
