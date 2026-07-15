import Foundation
import CoreVideo

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
    private let lock = NSLock()
    private var _droppedCount: Int = 0

    var droppedCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return _droppedCount
    }

    init(capacity: Int = 3) {
        self.capacity = max(1, capacity)
        self.buffer = Array(repeating: nil, count: self.capacity)
    }

    /// Enqueue a new raw frame. Overwrites oldest if full.
    func enqueue(_ frame: RawFrameData) {
        lock.lock()
        defer { lock.unlock() }

        if count == capacity {
            // Drop oldest
            buffer[readIndex] = nil
            readIndex = (readIndex + 1) % capacity
            count -= 1
            _droppedCount += 1
        }

        buffer[writeIndex] = frame
        writeIndex = (writeIndex + 1) % capacity
        count += 1
    }

    /// Dequeue the oldest available frame. Returns nil if empty.
    func dequeue() -> RawFrameData? {
        lock.lock()
        defer { lock.unlock() }

        guard count > 0 else { return nil }

        let frame = buffer[readIndex]
        buffer[readIndex] = nil
        readIndex = (readIndex + 1) % capacity
        count -= 1
        return frame
    }

    /// Drop backlog; return only the newest frame (lowest preview/encode latency).
    func dequeueLatest() -> RawFrameData? {
        lock.lock()
        defer { lock.unlock() }

        guard count > 0 else { return nil }

        while count > 1 {
            buffer[readIndex] = nil
            readIndex = (readIndex + 1) % capacity
            count -= 1
            _droppedCount += 1
        }

        let frame = buffer[readIndex]
        buffer[readIndex] = nil
        readIndex = (readIndex + 1) % capacity
        count -= 1
        return frame
    }

    var currentCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    var isFull: Bool {
        lock.lock()
        defer { lock.unlock() }
        return count == capacity
    }

    func flush() {
        lock.lock()
        defer { lock.unlock() }
        buffer = Array(repeating: nil, count: capacity)
        writeIndex = 0
        readIndex = 0
        count = 0
    }
}
