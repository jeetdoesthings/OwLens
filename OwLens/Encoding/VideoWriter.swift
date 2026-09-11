import AVFoundation
import VideoToolbox
import QuartzCore

/// Wraps a non-Sendable reference so it can be captured by a `@Sendable` closure
/// (e.g. `AVAssetWriter.finishWriting`'s callback). `AVAssetWriter` is not
/// `Sendable`, but here it is retired before this callback runs and is not used
/// concurrently elsewhere, so `@unchecked` isolation is sound.
private final class SendableAssetWriter: @unchecked Sendable {
    let value: AVAssetWriter
    init(_ value: AVAssetWriter) { self.value = value }
}

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
    private var hasStartedSession = false
    private var lastPixelBuffer: CVPixelBuffer?
    private var pendingAudioBuffers: [CMSampleBuffer] = []
    private let maxPendingAudioBuffers = 50
    private let lock = NSLock()

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
        curveType: LogCurveType = .sLog3Approx
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

        let compression: [String: Any] = [
            AVVideoAverageBitRateKey: bitrate,
            kVTCompressionPropertyKey_ProfileLevel as String: kVTProfileLevel_HEVC_Main_AutoLevel,
            AVVideoExpectedSourceFrameRateKey: Int(fps),
            AVVideoAverageNonDroppableFrameRateKey: Int(fps),
            AVVideoMaxKeyFrameIntervalKey: Int(fps),
            AVVideoAllowFrameReorderingKey: false as NSNumber
        ]

        let colorProperties: [String: Any]
        switch curveType {
        case .linear:
            colorProperties = [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
            ]
        case .appleLog2, .sLog3Approx:
            colorProperties = [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_2020,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_2020
            ]
        }

        var videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: compression,
            AVVideoColorPropertiesKey: colorProperties
        ]

        let vInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        vInput.expectsMediaDataInRealTime = true
        vInput.mediaTimeScale = CMTimeScale(fps * 1000)

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: vInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferMetalCompatibilityKey as String: true
            ])

        guard writer.canAdd(vInput) else {
            throw NSError(domain: "RawLogCam", code: 10, userInfo: [NSLocalizedDescriptionKey: "Cannot add video input (HEVC)"])
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
        writer.startSession(atSourceTime: .zero)

        self.assetWriter = writer
        self.videoInput = vInput
        self.audioInput = aInput
        self.pixelBufferAdaptor = adaptor
        self.frameCount = 0
        self.realFrameCount = 0
        self.droppedFrames = 0
        self.startHostTime = CACurrentMediaTime()
        self.hasStartedSession = true
        self.lastPixelBuffer = nil
        self.audioReferenceTime = .invalid
        self.pendingAudioBuffers.removeAll()
        self.isRecording = true

        print("[VideoWriter] CFR \(Int(fps))fps \(width)x\(height) codec=HEVC bitrate=\(bitrate)")
    }

    private func drainPendingAudioBuffersLocked() {
        guard let input = audioInput else { return }
        while !pendingAudioBuffers.isEmpty && input.isReadyForMoreMediaData {
            let next = pendingAudioBuffers.removeFirst()
            _ = input.append(next)
        }
    }

    /// Append a real camera frame. Fills any missing CFR slots by holding last frame.
    @discardableResult
    func appendFrame(pixelBuffer: CVPixelBuffer) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard isRecording,
              let adaptor = pixelBufferAdaptor,
              let input = videoInput else {
            return false
        }

        let pbW = CVPixelBufferGetWidth(pixelBuffer)
        let pbH = CVPixelBufferGetHeight(pixelBuffer)
        if pbW != width || pbH != height {
            droppedFrames += 1
            return false
        }

        let now = CACurrentMediaTime()

        if !hasStartedSession {
            startHostTime = now
            assetWriter?.startSession(atSourceTime: .zero)
            hasStartedSession = true
        }

        let elapsed = max(0, now - startHostTime)
        // How many CFR frames should exist by this wall time (0-based next index)
        // e.g. at t=1.0s @ 24fps -> need frames 0..23 written (count 24) -> targetCount = 24
        let wallTargetCount = Int64(floor(elapsed * targetFPS + 1e-9)) + 1
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
              hasStartedSession else {
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

        // Use the first audio sample's PTS as reference to keep all retiming
        // in the audio clock domain (not host time / CACurrentMediaTime).
        // This prevents drift from mismatched clock domains.
        let firstPTS = timings[0].presentationTimeStamp
        if CMTimeCompare(audioReferenceTime, .invalid) == 0 {
            audioReferenceTime = firstPTS
        }

        for i in 0..<timings.count {
            var pts = CMTimeSubtract(timings[i].presentationTimeStamp, audioReferenceTime)
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
        let boxedWriter = SendableAssetWriter(writer)
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

    var estimatedDuration: Double {
        lock.lock()
        defer { lock.unlock() }
        guard startHostTime > 0 else { return 0 }
        return CACurrentMediaTime() - startHostTime
    }
}
