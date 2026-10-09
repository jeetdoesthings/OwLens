import Foundation
import CoreVideo
import os

/// Thread-safe ring buffer for incoming RAW frames.
/// Decouples capture rate from processing rate — if the Metal pipeline
/// falls behind, oldest frames are silently dropped rather than
/// back-pressuring the capture loop / exploding RAM.
final class RawFrameBuffer {
    private let capacity: Int
    private var buffer: [RawFrameData?]
    private var writeIndex = 0
    private var readIndex = 0
    private var count = 0
    private let lock = OSAllocatedUnfairLock()
    private var _recordingDroppedCount: Int = 0

    /// Returns the number of real recording frames dropped due to buffer overrun.
    /// Preview frames skipped to maintain real-time UI are never counted here.
    var recordingDroppedCount: Int {
        lock.withLock { _recordingDroppedCount }
    }

    var droppedCount: Int {
        lock.withLock { _recordingDroppedCount }
    }

    func resetRecordingDrops() {
        lock.withLock {
            _recordingDroppedCount = 0
        }
    }

    init(capacity: Int = 16) {
        self.capacity = max(1, capacity)
        self.buffer = Array(repeating: nil, count: self.capacity)
    }

    /// Enqueue a new raw frame. Overwrites oldest if full.
    func enqueue(_ frame: RawFrameData) {
        lock.withLock {
            if count == capacity {
                // Drop oldest
                let old = buffer[readIndex]
                if old?.isRecordingFrame == true {
                    _recordingDroppedCount += 1
                }
                buffer[readIndex] = nil
                readIndex = (readIndex + 1) % capacity
                count -= 1
            }

            buffer[writeIndex] = frame
            writeIndex = (writeIndex + 1) % capacity
            count += 1
        }
    }

    /// Dequeue the oldest available frame. Returns nil if empty.
    func dequeue() -> RawFrameData? {
        lock.withLock {
            guard count > 0 else { return nil }

            let frame = buffer[readIndex]
            buffer[readIndex] = nil
            readIndex = (readIndex + 1) % capacity
            count -= 1
            return frame
        }
    }

    /// Drop backlog; return only the newest frame (lowest preview/encode latency).
    func dequeueLatest() -> RawFrameData? {
        lock.withLock {
            guard count > 0 else { return nil }

            let dropped = count - 1
            if dropped > 0 {
                for i in 0..<dropped {
                    let old = buffer[(readIndex + i) % capacity]
                    if old?.isRecordingFrame == true {
                        _recordingDroppedCount += 1
                    }
                    buffer[(readIndex + i) % capacity] = nil
                }
                readIndex = (readIndex + dropped) % capacity
                count -= dropped
            }

            let frame = buffer[readIndex]
            buffer[readIndex] = nil
            readIndex = (readIndex + 1) % capacity
            count -= 1
            return frame
        }
    }

    var currentCount: Int {
        lock.withLock { count }
    }

    var isFull: Bool {
        lock.withLock { count == capacity }
    }

    func flush() {
        lock.withLock {
            buffer = Array(repeating: nil, count: capacity)
            writeIndex = 0
            readIndex = 0
            count = 0
            _recordingDroppedCount = 0
        }
    }

    /// Check if any unread frame in the buffer belongs to an active recording session.
    func hasRecordingFrames() -> Bool {
        lock.withLock {
            guard count > 0 else { return false }
            for i in 0..<count {
                if buffer[(readIndex + i) % capacity]?.isRecordingFrame == true {
                    return true
                }
            }
            return false
        }
    }
}
