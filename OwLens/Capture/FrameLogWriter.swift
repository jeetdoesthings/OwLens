import Foundation
import CoreVideo
import AVFoundation

/// Append-only binary log for Bayer pixel buffers + per-frame metadata.
/// Written at binned resolution (~2016x1512) for ~5.8 MB/frame at 24fps.
///
/// Format: [header 64B] [frame 0: metadata 128B + pixel data] [frame 1: …]
/// Header: magic "OWL\0" 4B | version U32 | pixelFormat U32 |
///         width U32 | height U32 | bytesPerRow U32 | frameCount U32 | reserved 40B
///
/// Per-frame metadata (128B):
///   frameIndex U32 | blackLevel F32 | whiteLevel F32 | iso F32 |
///   exposureDuration F64 | cfaPattern S32 |
///   wbGainsR F32 | wbGainsG F32 | wbGainsB F32 |
///   lsc0 F32 | lsc1 F32 | lsc2 F32 | lsc3 F32 |
///   timestamp F64 | reserved 60B
final class FrameLogWriter {
    private let fileHandle: FileHandle
    private let headerSize: UInt64 = 64
    private let metadataSize: UInt64 = 128
    private var frameCount: UInt32 = 0
    private let bytesPerRow: Int
    private let framePixelSize: Int  // total bytes of pixel data per frame

    let width: Int
    let height: Int

    struct FrameMetadata {
        let blackLevel: Float
        let whiteLevel: Float
        let iso: Float
        let exposureDuration: Double
        let cfaPattern: Int32
        let wbGains: SIMD3<Float>
        let lscCoefficients: SIMD4<Float>
        let timestamp: Double
    }

    init?(url: URL, width: Int, height: Int, bytesPerRow: Int, pixelFormat: OSType) {
        self.width = width
        self.height = height
        self.bytesPerRow = bytesPerRow
        self.framePixelSize = bytesPerRow * height

        let fm = FileManager.default
        if fm.fileExists(atPath: url.path) {
            try? fm.removeItem(at: url)
        }
        guard fm.createFile(atPath: url.path, contents: nil, attributes: nil),
              let fh = try? FileHandle(forWritingTo: url) else {
            print("[FrameLogWriter] Failed to create file at \(url.path)")
            return nil
        }
        self.fileHandle = fh

        var header = Data(count: Int(headerSize))
        header.replaceSubrange(0..<4, with: "OWL\0".data(using: .ascii)!)
        header.storeU32(1, at: 4)
        header.storeU32(UInt32(pixelFormat), at: 8)
        header.storeU32(UInt32(width), at: 12)
        header.storeU32(UInt32(height), at: 16)
        header.storeU32(UInt32(bytesPerRow), at: 20)
        header.storeU32(0, at: 24)
        fileHandle.write(header)
        print("[FrameLogWriter] Created \(width)x\(height) bpr=\(bytesPerRow)")
    }

    /// Append one frame of raw pixel data + metadata.
    /// `pixelData` must be `height * bytesPerRow` bytes (caller must provide already-binned data).
    func appendFrame(pixelData: Data, metadata: FrameMetadata) {
        var meta = Data(count: Int(metadataSize))
        meta.storeU32(frameCount, at: 0)
        meta.storeF32(metadata.blackLevel, at: 4)
        meta.storeF32(metadata.whiteLevel, at: 8)
        meta.storeF32(metadata.iso, at: 12)
        meta.storeF64(metadata.exposureDuration, at: 16)
        meta.storeS32(metadata.cfaPattern, at: 24)
        meta.storeF32(metadata.wbGains.x, at: 28)
        meta.storeF32(metadata.wbGains.y, at: 32)
        meta.storeF32(metadata.wbGains.z, at: 36)
        meta.storeF32(metadata.lscCoefficients[0], at: 40)
        meta.storeF32(metadata.lscCoefficients[1], at: 44)
        meta.storeF32(metadata.lscCoefficients[2], at: 48)
        meta.storeF32(metadata.lscCoefficients[3], at: 52)
        meta.storeF64(metadata.timestamp, at: 56)
        fileHandle.write(meta)
        fileHandle.write(pixelData)
        frameCount += 1
    }

    /// Append a full-res CVPixelBuffer by binning it on CPU using the same
    /// phase-preserving pattern as the Metal binBayerCFA kernel.
    func appendBinned(pixelBuffer: CVPixelBuffer, metadata: FrameMetadata) -> Bool {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return false }

        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)
        let srcBPR = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let src = base.assumingMemoryBound(to: UInt16.self)

        let dstW = width
        let dstH = height
        var binned = [UInt16](repeating: 0, count: dstW * dstH)

        // Phase-preserving bin: each output pixel (x,y) samples src(2x + (x&1), 2y + (y&1)).
        // This keeps the Bayer RGGB phase intact.
        for y in 0..<dstH {
            let sy = min(2 * y + (y & 1), srcH - 1)
            let dstRow = y * dstW
            for x in 0..<dstW {
                let sx = min(2 * x + (x & 1), srcW - 1)
                binned[dstRow + x] = src[sy * (srcBPR / 2) + sx]
            }
        }

        let pixelData = Data(bytes: binned, count: dstW * dstH * 2)
        appendFrame(pixelData: pixelData, metadata: metadata)
        return true
    }

    func close() {
        var countData = Data(count: 4)
        countData.storeU32(frameCount, at: 0)
        fileHandle.seek(toFileOffset: 24)
        fileHandle.write(countData)
        fileHandle.closeFile()
        print("[FrameLogWriter] Closed: \(frameCount) frames")
    }

    var totalFrames: Int { Int(frameCount) }
}

// MARK: - Binary helpers

extension Data {
    mutating func storeU32(_ value: UInt32, at offset: Int) {
        var v = value
        Swift.withUnsafeBytes(of: &v) { dest in
            replaceSubrange(offset..<offset+4, with: dest)
        }
    }
    mutating func storeS32(_ value: Int32, at offset: Int) {
        var v = value
        Swift.withUnsafeBytes(of: &v) { dest in
            replaceSubrange(offset..<offset+4, with: dest)
        }
    }
    mutating func storeF32(_ value: Float, at offset: Int) {
        var v = value
        Swift.withUnsafeBytes(of: &v) { dest in
            replaceSubrange(offset..<offset+4, with: dest)
        }
    }
    mutating func storeF64(_ value: Double, at offset: Int) {
        var v = value
        Swift.withUnsafeBytes(of: &v) { dest in
            replaceSubrange(offset..<offset+8, with: dest)
        }
    }
    func readU32(_ offset: Int) -> UInt32 {
        guard offset + 4 <= count else { return 0 }
        return withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
    }
    func readF32(_ offset: Int) -> Float {
        guard offset + 4 <= count else { return 0 }
        return withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: Float.self) }
    }
    func readF64(_ offset: Int) -> Double {
        guard offset + 8 <= count else { return 0 }
        return withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: Double.self) }
    }
}

// MARK: - Reader

struct FrameLogReader {
    struct Header {
        let width: Int; let height: Int; let bytesPerRow: Int
        let pixelFormat: OSType; let frameCount: Int
    }
    struct FrameRecord {
        let metadata: FrameLogWriter.FrameMetadata
        let pixelData: Data
    }

    static func readHeader(from url: URL) -> Header? {
        guard let fh = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { fh.closeFile() }
        let d = fh.readData(ofLength: 64)
        guard d.count == 64, String(data: d[0..<4], encoding: .ascii) == "OWL\0" else { return nil }
        return Header(width: Int(d.readU32(12)), height: Int(d.readU32(16)),
                      bytesPerRow: Int(d.readU32(20)), pixelFormat: OSType(d.readU32(8)),
                      frameCount: Int(d.readU32(24)))
    }

    /// Enumerate frames via callback. Returns early on gap-detection failure.
    static func enumerateFrames(url: URL, callback: (FrameRecord, Int, Int) -> Bool) {
        guard let header = readHeader(from: url),
              let fh = try? FileHandle(forReadingFrom: url) else { return }
        defer { fh.closeFile() }
        fh.seek(toFileOffset: 64)
        let frameSize = 128 + header.bytesPerRow * header.height

        for i in 0..<header.frameCount {
            let d = fh.readData(ofLength: frameSize)
            guard d.count == frameSize else { return }
            let meta = FrameLogWriter.FrameMetadata(
                blackLevel: d.readF32(4), whiteLevel: d.readF32(8),
                iso: d.readF32(12), exposureDuration: d.readF64(16),
                cfaPattern: Int32(bitPattern: d.readU32(24)),
                wbGains: SIMD3<Float>(d.readF32(28), d.readF32(32), d.readF32(36)),
                lscCoefficients: SIMD4<Float>(d.readF32(40), d.readF32(44), d.readF32(48), d.readF32(52)),
                timestamp: d.readF64(56))
            let pixelData = d.subdata(in: 128..<frameSize)
            let cont = callback(FrameRecord(metadata: meta, pixelData: pixelData), i, header.frameCount)
            if !cont { return }
        }
    }
}
