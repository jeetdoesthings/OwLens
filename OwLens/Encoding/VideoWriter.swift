import AVFoundation
import VideoToolbox
import QuartzCore

/// AVAssetWriter — HEVC + optional AAC at **constant** 24 or 30 fps.
///
/// Capture often delivers fewer real RAW frames than target. We still write a
/// true CFR timeline: missing slots **hold the last real frame**.
/// Result: file reports 24/30 fps, duration ≈ wall-clock, no “20 fps” metadata.
final class VideoWriter {
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
    private var hasStartedSession = false
    private var lastPixelBuffer: CVPixelBuffer?
    private let lock = NSLock()

    var isRecording = false
    private(set) var droppedFrames: Int = 0

    func start(
        outputURL: URL,
        width: Int,
        height: Int,
        bitrate: Int = 100_000_000,
        targetFPS: Double = 24,
        includeAudio: Bool = true
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
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: compression,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
            ]
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

        self.assetWriter = writer
        self.videoInput = vInput
        self.audioInput = aInput
        self.pixelBufferAdaptor = adaptor
        self.frameCount = 0
        self.realFrameCount = 0
        self.droppedFrames = 0
        self.startHostTime = 0
        self.hasStartedSession = false
        self.lastPixelBuffer = nil
        self.isRecording = true

        print("[VideoWriter] CFR \(Int(fps))fps \(width)x\(height) codec=HEVC bitrate=\(bitrate)")
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
            guard input.isReadyForMoreMediaData else {
                droppedFrames += 1
                lastPixelBuffer = pixelBuffer
                return false
            }
            if writeCFR(pixelBuffer, index: 0, adaptor: adaptor) {
                lastPixelBuffer = pixelBuffer
                frameCount = 1
                realFrameCount = 1
                return true
            }
            droppedFrames += 1
            return false
        }

        let elapsed = max(0, now - startHostTime)
        // How many CFR frames should exist by this wall time (0-based next index)
        // e.g. at t=1.0s @ 24fps → need frames 0..23 written (count 24) → targetCount = 24
        let wallTargetCount = Int64(floor(elapsed * targetFPS + 1e-9)) + 1
        // Always advance at least one slot for this real frame
        let targetCount = max(frameCount + 1, wallTargetCount)

        var wroteAny = false

        // Hold last real frame for skipped slots
        if let hold = lastPixelBuffer {
            while frameCount < targetCount - 1 {
                guard input.isReadyForMoreMediaData else {
                    droppedFrames += 1
                    break
                }
                if writeCFR(hold, index: frameCount, adaptor: adaptor) {
                    frameCount += 1
                    wroteAny = true
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
            return wroteAny
        }
        if writeCFR(pixelBuffer, index: frameCount, adaptor: adaptor) {
            frameCount += 1
            realFrameCount += 1
            lastPixelBuffer = pixelBuffer
            return true
        }
        droppedFrames += 1
        lastPixelBuffer = pixelBuffer
        return wroteAny
    }

    /// Pad to full wall-clock duration at target FPS so file length matches shoot time.
    private func padToWallClockLocked() {
        guard hasStartedSession,
              let hold = lastPixelBuffer,
              let adaptor = pixelBufferAdaptor,
              let input = videoInput else { return }

        let elapsed = max(0, CACurrentMediaTime() - startHostTime)
        // Round to nearest frame so 10.0s @ 24 → exactly 240 frames
        let targetCount = Int64((elapsed * targetFPS).rounded())
        while frameCount < targetCount {
            guard input.isReadyForMoreMediaData else { break }
            if writeCFR(hold, index: frameCount, adaptor: adaptor) {
                frameCount += 1
            } else {
                break
            }
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
              startHostTime > 0 else {
            return false
        }
        guard input.isReadyForMoreMediaData else { return false }
        guard CMSampleBufferDataIsReady(sampleBuffer) else { return false }

        var timingCount: CMItemCount = 0
        CMSampleBufferGetOutputSampleTimingInfoArray(sampleBuffer, entryCount: 0, arrayToFill: nil, entriesNeededOut: &timingCount)
        guard timingCount > 0 else { return false }

        var timings = Array(repeating: CMSampleTimingInfo(
            duration: .invalid,
            presentationTimeStamp: .invalid,
            decodeTimeStamp: .invalid
        ), count: timingCount)
        CMSampleBufferGetOutputSampleTimingInfoArray(sampleBuffer, entryCount: timingCount, arrayToFill: &timings, entriesNeededOut: &timingCount)

        let startCMTime = CMTime(seconds: startHostTime, preferredTimescale: 48_000)
        
        for i in 0..<timings.count {
            timings[i].presentationTimeStamp = CMTimeSubtract(timings[i].presentationTimeStamp, startCMTime)
            if timings[i].decodeTimeStamp.isValid {
                timings[i].decodeTimeStamp = timings[i].presentationTimeStamp
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
        return input.append(retimed)
    }

    func finish(completion: @escaping (URL?) -> Void) {
        lock.lock()
        guard isRecording else {
            lock.unlock()
            completion(nil)
            return
        }

        padToWallClockLocked()

        let url = assetWriter?.outputURL
        isRecording = false
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
        writer?.finishWriting {
            let duration = Double(total) / fps
            let status = writer?.status
            print("[VideoWriter] Done. timeline=\(total) real=\(real) holds=\(total - real) drops=\(drops) \(String(format: "%.2f", duration))s @ \(Int(fps))fps status=\(String(describing: status))")
            if status == .failed {
                print("[VideoWriter] Error: \(String(describing: writer?.error))")
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
