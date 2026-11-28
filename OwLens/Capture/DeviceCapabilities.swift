import AVFoundation
import Foundation

#if canImport(Darwin)
import Darwin
#endif

// MARK: - Chip tier & Bayer

/// Coarse SoC generation for default fps / resolution guidance.
enum ChipTier: String, CaseIterable {
    case a12a13 = "A12/A13"
    case a14 = "A14"
    case a15 = "A15"
    case a16Plus = "A16+"
    case unknown = "Unknown"
}

/// CFA pattern ids matching Debayer.metal / CaptureController.
enum BayerPatternID: Int32, CaseIterable {
    case rggb = 0
    case grbg = 1
    case gbrg = 2
    case bggr = 3

    var label: String {
        switch self {
        case .rggb: return "RGGB"
        case .grbg: return "GRBG"
        case .gbrg: return "GBRG"
        case .bggr: return "BGGR"
        }
    }
}

/// Output codec choice
enum VideoCodecOption: String, CaseIterable, Identifiable {
    case hevc = "hevc"
    case proRes422 = "proRes422"
    case proRes422HQ = "proRes422HQ"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .hevc: return "HEVC 10-bit"
        case .proRes422: return "ProRes 422"
        case .proRes422HQ: return "ProRes 422 HQ"
        }
    }

    var avCodecType: AVVideoCodecType {
        switch self {
        case .hevc: return .hevc
        case .proRes422: return .proRes422
        case .proRes422HQ: return .proRes422HQ
        }
    }

    /// Detect ProRes from the actual hardware encoder, then verify the exact
    /// 1080p writer configuration. Model-name heuristics and `canAdd` alone
    /// have both produced false positives on some devices.
    static let isProResSupported: Bool = {
        guard #available(iOS 15.0, *) else { return false }
        // AVAssetWriter must be given a writable file URL; "/dev/null" makes the initializer
        // fail on iOS and silently disables ProRes for every device. Use a unique temp file.
        let probeURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("owlens_prores_probe_\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: probeURL) }
        guard let writer = try? AVAssetWriter(outputURL: probeURL, fileType: .mov) else { return false }
        let testInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.proRes422,
            AVVideoWidthKey: 1920,
            AVVideoHeightKey: 1080
        ])
        return writer.canAdd(testInput)
    }()

    static var availableCases: [VideoCodecOption] {
        if isProResSupported {
            return [.hevc, .proRes422, .proRes422HQ]
        } else {
            return [.hevc]
        }
    }
}

// MARK: - LSCCoefficients

/// Per-device lens shading correction coefficients from DNG / calibration.
struct LSCCoefficients: Sendable {
    let radialR: Float
    let radialG: Float
    let radialB: Float
    let radial4R: Float
    let radial4G: Float
    let radial4B: Float
    let azimuthR: Float
    let azimuthG: Float
    let azimuthB: Float

    /// Full 4-term initializer.
    init(
        radialR: Float, radialG: Float, radialB: Float,
        radial4R: Float = 0, radial4G: Float = 0, radial4B: Float = 0,
        azimuthR: Float = 0, azimuthG: Float = 0, azimuthB: Float = 0
    ) {
        self.radialR = radialR
        self.radialG = radialG
        self.radialB = radialB
        self.radial4R = radial4R
        self.radial4G = radial4G
        self.radial4B = radial4B
        self.azimuthR = azimuthR
        self.azimuthG = azimuthG
        self.azimuthB = azimuthB
    }

    var asLSCParams: LSCParams {
        LSCParams(
            radialR: radialR, radialG: radialG, radialB: radialB,
            radial4R: radial4R, radial4G: radial4G, radial4B: radial4B,
            azimuthR: azimuthR, azimuthG: azimuthG, azimuthB: azimuthB
        )
    }
}

extension LSCCoefficients: Codable {}

// MARK: - DNGGainMap

/// Factory 2D per-CFA-plane lens shading correction map parsed from Apple DNG `OpcodeList2` (`GainMap`).
/// Channels per grid point are interleaved in `(R, Gr, Gb, B)` order.
struct DNGGainMap: Sendable, Codable, Equatable {
    let width: Int
    let height: Int
    /// Interleaved `(R, Gr, Gb, B)` gains; count == width * height * 4.
    let rgbaGains: [Float]
    /// Affine transform mapping normalized sensor UV `(0..1, 0..1)` to GainMap texture UV:
    /// `texUV = sensorUV * SIMD2(uvScaleX, uvScaleY) + SIMD2(uvOffsetX, uvOffsetY)`.
    let uvScaleX: Float
    let uvScaleY: Float
    let uvOffsetX: Float
    let uvOffsetY: Float
    /// Stable identifier for GPU texture caching.
    let cacheID: String
    /// Diagnostic summary (grid dims and corner gains).
    let summary: String
}

/// Pure-Swift Adobe DNG TIFF / `OpcodeList2` (`GainMap`, Opcode ID 9) parser.
enum DNGGainMapParser {
    private struct ParsedGainMapOpcode {
        let top: UInt32
        let left: UInt32
        let bottom: UInt32
        let right: UInt32
        let plane: UInt32
        let planes: UInt32
        let rowPitch: UInt32
        let colPitch: UInt32
        let mapPointsV: Int
        let mapPointsH: Int
        let mapSpacingV: Double
        let mapSpacingH: Double
        let mapOriginV: Double
        let mapOriginH: Double
        let mapPlanes: Int
        let gains: [Float] // row-major [v][h][p]
    }

    private struct IFDScanResult {
        var imageWidth: UInt32?
        var imageHeight: UInt32?
        var activeArea: [UInt32]?
        var cfaPattern4: [UInt8]?
        var opcodeList1: Data?
        var opcodeList2: Data?
        var opcodeList3: Data?
        var subIFDOffsets: [UInt32] = []
        var nextIFDOffset: UInt32 = 0
    }

    /// Extract a factory `GainMap` from raw DNG file bytes (`AVCapturePhoto.fileDataRepresentation()`).
    static func extract(
        fromDNGData data: Data,
        cfaPattern: Int32,
        rawWidth: Int,
        rawHeight: Int,
        cacheID: String
    ) -> DNGGainMap? {
        guard data.count >= 8 else { return nil }
        let b0 = data[data.startIndex]
        let b1 = data[data.startIndex + 1]
        let littleEndian: Bool
        if b0 == 0x49 && b1 == 0x49 {
            littleEndian = true
        } else if b0 == 0x4D && b1 == 0x4D {
            littleEndian = false
        } else {
            return nil
        }

        let magic = readUInt16(data, offset: 2, littleEndian: littleEndian)
        guard magic == 42 else { return nil }
        let ifd0 = readUInt32(data, offset: 4, littleEndian: littleEndian)
        guard ifd0 > 0 else { return nil }

        var queue: [UInt32] = [ifd0]
        var visited = Set<UInt32>()
        var radialFallback: DNGGainMap?

        while !queue.isEmpty && visited.count < 16 {
            let offset = queue.removeFirst()
            guard offset >= 8, !visited.contains(offset) else { continue }
            visited.insert(offset)

            guard let ifd = parseIFD(data, offset: Int(offset), littleEndian: littleEndian) else {
                continue
            }
            for sub in ifd.subIFDOffsets where sub > 0 && !visited.contains(sub) {
                queue.append(sub)
            }
            if ifd.nextIFDOffset > 0 && !visited.contains(ifd.nextIFDOffset) {
                queue.append(ifd.nextIFDOffset)
            }

            let resolvedCFA = resolveCFAPattern(ifd.cfaPattern4) ?? cfaPattern
            let resolvedW = rawWidth > 0 ? rawWidth : Int(ifd.imageWidth ?? 0)
            let resolvedH = rawHeight > 0 ? rawHeight : Int(ifd.imageHeight ?? 0)

            for listData in [ifd.opcodeList2, ifd.opcodeList1, ifd.opcodeList3].compactMap({ $0 }) {
                if let map = parseOpcodeList(
                    listData,
                    cfaPattern: resolvedCFA,
                    rawWidth: resolvedW,
                    rawHeight: resolvedH,
                    activeArea: ifd.activeArea,
                    cacheID: cacheID
                ) {
                    return map
                }
            }
            if radialFallback == nil, let list = ifd.opcodeList3,
               let width = ifd.imageWidth, let height = ifd.imageHeight {
                radialFallback = radialVignetteMap(list, storedWidth: Int(width), storedHeight: Int(height),
                                                   rawWidth: resolvedW, rawHeight: resolvedH,
                                                   activeArea: ifd.activeArea, cacheID: cacheID)
            }
        }
        return radialFallback
    }

    // OpcodeList3 operates on the demosaiced active image. Represent the radial
    // brightness function on the same sensor-space grid used by the Metal path.
    private static func radialVignetteMap(
        _ list: Data, storedWidth: Int, storedHeight: Int,
        rawWidth: Int, rawHeight: Int, activeArea: [UInt32]?, cacheID: String
    ) -> DNGGainMap? {
        guard list.count >= 4, storedWidth > 0, storedHeight > 0 else { return nil }
        let count = Int(readUInt32BE(list, offset: 0))
        guard (1...64).contains(count) else { return nil }
        var cursor = 4
        var parameters: [Double]?
        for _ in 0..<count {
            guard cursor + 16 <= list.count else { return nil }
            let id = readUInt32BE(list, offset: cursor)
            let size = Int(readUInt32BE(list, offset: cursor + 12))
            cursor += 16
            guard size <= list.count - cursor else { return nil }
            if id == 3 {
                guard size == 56, parameters == nil else { return nil }
                parameters = (0..<7).map { readDoubleBE(list, offset: cursor + $0 * 8) }
            }
            cursor += size
        }
        guard let coefficients = parameters, coefficients.allSatisfy(\.isFinite),
              (0...1).contains(coefficients[5]), (0...1).contains(coefficients[6]) else { return nil }

        let area = activeArea?.map(Int.init) ?? [0, 0, storedHeight, storedWidth]
        guard area.count == 4, area[0] < area[2], area[1] < area[3],
              area[2] <= storedHeight, area[3] <= storedWidth else { return nil }
        let width = area[3] - area[1], height = area[2] - area[0]
        let originX: Int, originY: Int
        if rawWidth == width, rawHeight == height {
            originX = 0
            originY = 0
        } else if rawWidth == storedWidth, rawHeight == storedHeight {
            originX = area[1]
            originY = area[0]
        } else {
            return nil
        }

        let centerX = Double(width) * coefficients[5]
        let centerY = Double(height) * coefficients[6]
        let radius = hypot(max(centerX, Double(width) - centerX), max(centerY, Double(height) - centerY))
        let side = 65
        var gains = [Float]()
        gains.reserveCapacity(side * side * 4)
        for row in 0..<side {
            for column in 0..<side {
                let x = min(Double(width), max(0, Double(column * rawWidth) / Double(side - 1) - Double(originX)))
                let y = min(Double(height), max(0, Double(row * rawHeight) / Double(side - 1) - Double(originY)))
                let r = hypot(x - centerX, y - centerY) / radius
                let r2 = r * r
                let gain = 1 + coefficients.prefix(5).reversed().reduce(0.0) { r2 * ($0 + $1) }
                guard gain.isFinite, gain >= 0.1, gain <= 16 else { return nil }
                for _ in 0..<4 { gains.append(Float(gain)) }
            }
        }
        let scale = Float(side - 1) / Float(side)
        let offset = 0.5 / Float(side)
        return DNGGainMap(width: side, height: side, rgbaGains: gains,
                          uvScaleX: scale, uvScaleY: scale, uvOffsetX: offset, uvOffsetY: offset,
                          cacheID: cacheID, summary: "DNG FixVignetteRadial • peak \(gains.max() ?? 1)×")
    }

    /// Parse an Adobe DNG `OpcodeList` binary blob (always Big-Endian per DNG spec) for `GainMap` opcodes (ID 9).
    static func parseOpcodeList(
        _ opcodeData: Data,
        cfaPattern: Int32,
        rawWidth: Int,
        rawHeight: Int,
        activeArea: [UInt32]? = nil,
        cacheID: String
    ) -> DNGGainMap? {
        guard opcodeData.count >= 4 else { return nil }
        let opcodeCount = Int(readUInt32BE(opcodeData, offset: 0))
        guard opcodeCount > 0, opcodeCount <= 64 else { return nil }

        var cursor = 4
        var gainMaps: [ParsedGainMapOpcode] = []

        for _ in 0..<opcodeCount {
            guard cursor + 16 <= opcodeData.count else { break }
            let opcodeID = readUInt32BE(opcodeData, offset: cursor)
            // skip dngVersion (+4) and flags (+8)
            let paramBytes = Int(readUInt32BE(opcodeData, offset: cursor + 12))
            cursor += 16
            guard paramBytes >= 0, cursor + paramBytes <= opcodeData.count else { break }
            let payload = opcodeData.subdata(in: (opcodeData.startIndex + cursor)..<(opcodeData.startIndex + cursor + paramBytes))
            cursor += paramBytes

            if opcodeID == 9, let gm = parseGainMapOpcode(payload) {
                gainMaps.append(gm)
            }
        }

        guard !gainMaps.isEmpty else { return nil }
        return assembleGainMap(
            opcodes: gainMaps,
            cfaPattern: cfaPattern,
            rawWidth: rawWidth,
            rawHeight: rawHeight,
            activeArea: activeArea,
            cacheID: cacheID
        )
    }

    private static func parseGainMapOpcode(_ payload: Data) -> ParsedGainMapOpcode? {
        guard payload.count >= 76 else { return nil }
        let top = readUInt32BE(payload, offset: 0)
        let left = readUInt32BE(payload, offset: 4)
        let bottom = readUInt32BE(payload, offset: 8)
        let right = readUInt32BE(payload, offset: 12)
        let plane = readUInt32BE(payload, offset: 16)
        let planes = readUInt32BE(payload, offset: 20)
        let rowPitch = max(1, readUInt32BE(payload, offset: 24))
        let colPitch = max(1, readUInt32BE(payload, offset: 28))
        let mapPointsV = Int(readUInt32BE(payload, offset: 32))
        let mapPointsH = Int(readUInt32BE(payload, offset: 36))
        let mapSpacingV = readDoubleBE(payload, offset: 40)
        let mapSpacingH = readDoubleBE(payload, offset: 48)
        let mapOriginV = readDoubleBE(payload, offset: 56)
        let mapOriginH = readDoubleBE(payload, offset: 64)
        let mapPlanes = Int(readUInt32BE(payload, offset: 72))

        guard mapPointsV >= 2, mapPointsH >= 2, mapPointsV <= 256, mapPointsH <= 256,
              mapPlanes >= 1, mapPlanes <= 4,
              bottom > top, right > left else {
            return nil
        }

        let floatCount = mapPointsV * mapPointsH * mapPlanes
        guard payload.count >= 76 + floatCount * 4 else { return nil }

        var gains = [Float]()
        gains.reserveCapacity(floatCount)
        for i in 0..<floatCount {
            let val = readFloatBE(payload, offset: 76 + i * 4)
            guard val.isFinite, val >= 0.25, val <= 32.0 else { return nil }
            gains.append(val)
        }

        return ParsedGainMapOpcode(
            top: top, left: left, bottom: bottom, right: right,
            plane: plane, planes: planes, rowPitch: rowPitch, colPitch: colPitch,
            mapPointsV: mapPointsV, mapPointsH: mapPointsH,
            mapSpacingV: mapSpacingV, mapSpacingH: mapSpacingH,
            mapOriginV: mapOriginV, mapOriginH: mapOriginH,
            mapPlanes: mapPlanes, gains: gains
        )
    }

    private static func assembleGainMap(
        opcodes: [ParsedGainMapOpcode],
        cfaPattern: Int32,
        rawWidth: Int,
        rawHeight: Int,
        activeArea: [UInt32]?,
        cacheID: String
    ) -> DNGGainMap? {
        guard let base = opcodes.first else { return nil }
        let width = base.mapPointsH
        let height = base.mapPointsV
        let count = width * height

        let activeTop = (activeArea?.count ?? 0) >= 4 ? activeArea![0] : 0
        let activeLeft = (activeArea?.count ?? 0) >= 4 ? activeArea![1] : 0

        var planeR: [Float]?
        var planeGr: [Float]?
        var planeGb: [Float]?
        var planeB: [Float]?

        func extractPlane(_ op: ParsedGainMapOpcode, planeIdx: Int) -> [Float] {
            if op.mapPointsH == width && op.mapPointsV == height {
                var out = [Float](repeating: 1.0, count: count)
                for v in 0..<height {
                    for h in 0..<width {
                        out[v * width + h] = op.gains[(v * op.mapPointsH + h) * op.mapPlanes + planeIdx]
                    }
                }
                return out
            }
            var out = [Float](repeating: 1.0, count: count)
            for v in 0..<height {
                let fv = Float(v) / Float(max(1, height - 1)) * Float(op.mapPointsV - 1)
                let v0 = min(op.mapPointsV - 1, max(0, Int(fv)))
                let v1 = min(op.mapPointsV - 1, v0 + 1)
                let tv = fv - Float(v0)
                for h in 0..<width {
                    let fh = Float(h) / Float(max(1, width - 1)) * Float(op.mapPointsH - 1)
                    let h0 = min(op.mapPointsH - 1, max(0, Int(fh)))
                    let h1 = min(op.mapPointsH - 1, h0 + 1)
                    let th = fh - Float(h0)
                    let g00 = op.gains[(v0 * op.mapPointsH + h0) * op.mapPlanes + planeIdx]
                    let g10 = op.gains[(v0 * op.mapPointsH + h1) * op.mapPlanes + planeIdx]
                    let g01 = op.gains[(v1 * op.mapPointsH + h0) * op.mapPlanes + planeIdx]
                    let g11 = op.gains[(v1 * op.mapPointsH + h1) * op.mapPlanes + planeIdx]
                    let topVal = g00 + (g10 - g00) * th
                    let botVal = g01 + (g11 - g01) * th
                    out[v * width + h] = topVal + (botVal - topVal) * tv
                }
            }
            return out
        }

        for op in opcodes {
            if op.rowPitch == 2 && op.colPitch == 2 && op.mapPlanes == 1 {
                let x = Int(activeLeft + op.left)
                let y = Int(activeTop + op.top)
                let xEven = ((x ^ Int(cfaPattern)) & 1) == 0
                let yEven = ((y ^ Int(cfaPattern >> 1)) & 1) == 0
                let extracted = extractPlane(op, planeIdx: 0)
                if xEven && yEven {
                    planeR = extracted
                } else if !xEven && yEven {
                    planeGr = extracted
                } else if xEven && !yEven {
                    planeGb = extracted
                } else {
                    planeB = extracted
                }
            } else if op.mapPlanes == 1 {
                let extracted = extractPlane(op, planeIdx: 0)
                if op.planes == 1 && op.plane < 4 {
                    switch op.plane {
                    case 0: if planeR == nil { planeR = extracted }
                    case 1: if planeGr == nil { planeGr = extracted }
                    case 2: if planeGb == nil { planeGb = extracted }
                    default: if planeB == nil { planeB = extracted }
                    }
                } else {
                    if planeR == nil { planeR = extracted }
                    if planeGr == nil { planeGr = extracted }
                    if planeGb == nil { planeGb = extracted }
                    if planeB == nil { planeB = extracted }
                }
            } else if op.mapPlanes == 3 {
                planeR = extractPlane(op, planeIdx: 0)
                let g = extractPlane(op, planeIdx: 1)
                planeGr = g
                planeGb = g
                planeB = extractPlane(op, planeIdx: 2)
            } else if op.mapPlanes >= 4 {
                planeR = extractPlane(op, planeIdx: 0)
                planeGr = extractPlane(op, planeIdx: 1)
                planeGb = extractPlane(op, planeIdx: 2)
                planeB = extractPlane(op, planeIdx: 3)
            }
        }

        // Fill any missing Bayer green/color planes safely.
        let fallbackG = planeGr ?? planeGb ?? planeR ?? planeB
        guard let gAnchor = fallbackG else { return nil }
        let finalGr = planeGr ?? planeGb ?? gAnchor
        let finalGb = planeGb ?? planeGr ?? gAnchor
        let finalR = planeR ?? gAnchor
        let finalB = planeB ?? gAnchor

        var rgba = [Float](repeating: 1.0, count: count * 4)
        for i in 0..<count {
            rgba[i * 4 + 0] = finalR[i]
            rgba[i * 4 + 1] = finalGr[i]
            rgba[i * 4 + 2] = finalGb[i]
            rgba[i * 4 + 3] = finalB[i]
        }

        // Compute affine UV transform mapping sensor UV [0, 1] -> GainMap texel center UV.
        let macroLeft = Double(activeLeft + (base.left & ~1))
        let macroTop = Double(activeTop + (base.top & ~1))
        let macroRight = Double(activeLeft + base.right)
        let macroBottom = Double(activeTop + base.bottom)
        let boxW = max(1.0, macroRight - macroLeft)
        let boxH = max(1.0, macroBottom - macroTop)
        let rW = rawWidth > 0 ? Double(rawWidth) : boxW
        let rH = rawHeight > 0 ? Double(rawHeight) : boxH
        let spacingH = base.mapSpacingH > 1e-9 ? base.mapSpacingH : (1.0 / Double(width - 1))
        let spacingV = base.mapSpacingV > 1e-9 ? base.mapSpacingV : (1.0 / Double(height - 1))

        let scaleX = Float((rW / boxW) / (spacingH * Double(width)))
        let scaleY = Float((rH / boxH) / (spacingV * Double(height)))
        let offsetX = Float((0.5 + (-macroLeft / boxW - base.mapOriginH) / spacingH) / Double(width))
        let offsetY = Float((0.5 + (-macroTop / boxH - base.mapOriginV) / spacingV) / Double(height))

        let centerIdx = (height / 2) * width + (width / 2)
        let cornerIndices = [0, width - 1, (height - 1) * width, height * width - 1]
        var maxCornerR: Float = 1.0
        var maxCornerG: Float = 1.0
        var maxCornerB: Float = 1.0
        for ci in cornerIndices {
            maxCornerR = max(maxCornerR, finalR[ci])
            maxCornerG = max(maxCornerG, 0.5 * (finalGr[ci] + finalGb[ci]))
            maxCornerB = max(maxCornerB, finalB[ci])
        }
        let cR = finalR[centerIdx]
        let cG = 0.5 * (finalGr[centerIdx] + finalGb[centerIdx])
        let cB = finalB[centerIdx]
        let summary = String(
            format: "%dx%d (%d op) center=(%.2f,%.2f,%.2f) cornerMax=(%.2f,%.2f,%.2f)",
            width, height, opcodes.count, cR, cG, cB, maxCornerR, maxCornerG, maxCornerB
        )

        return DNGGainMap(
            width: width,
            height: height,
            rgbaGains: rgba,
            uvScaleX: scaleX,
            uvScaleY: scaleY,
            uvOffsetX: offsetX,
            uvOffsetY: offsetY,
            cacheID: cacheID,
            summary: summary
        )
    }

    private static func parseIFD(_ data: Data, offset: Int, littleEndian: Bool) -> IFDScanResult? {
        guard offset + 2 <= data.count else { return nil }
        let entryCount = Int(readUInt16(data, offset: offset, littleEndian: littleEndian))
        guard entryCount > 0, entryCount <= 256, offset + 2 + entryCount * 12 + 4 <= data.count else {
            return nil
        }

        var result = IFDScanResult()
        for i in 0..<entryCount {
            let entryOff = offset + 2 + i * 12
            let tag = readUInt16(data, offset: entryOff, littleEndian: littleEndian)
            let type = readUInt16(data, offset: entryOff + 2, littleEndian: littleEndian)
            let count = Int(readUInt32(data, offset: entryOff + 4, littleEndian: littleEndian))
            let valFieldOff = entryOff + 8

            switch tag {
            case 0x0100: // ImageWidth
                result.imageWidth = readFirstIntTag(data, type: type, count: count, valFieldOff: valFieldOff, littleEndian: littleEndian)
            case 0x0101: // ImageLength
                result.imageHeight = readFirstIntTag(data, type: type, count: count, valFieldOff: valFieldOff, littleEndian: littleEndian)
            case 0x014A: // SubIFDs
                if count > 0 && count <= 16 {
                    result.subIFDOffsets = readUInt32Array(data, type: type, count: count, valFieldOff: valFieldOff, littleEndian: littleEndian)
                }
            case 0xC68D: // ActiveArea
                if count == 4 {
                    result.activeArea = readUInt32Array(data, type: type, count: 4, valFieldOff: valFieldOff, littleEndian: littleEndian)
                }
            case 0x828E: // CFAPattern
                if let bytes = readTagBytes(data, type: type, count: count, valFieldOff: valFieldOff, littleEndian: littleEndian), bytes.count >= 4 {
                    result.cfaPattern4 = Array(bytes.prefix(4))
                }
            case 0xC740: // OpcodeList1
                result.opcodeList1 = readTagBytes(data, type: type, count: count, valFieldOff: valFieldOff, littleEndian: littleEndian)
            case 0xC741: // OpcodeList2
                result.opcodeList2 = readTagBytes(data, type: type, count: count, valFieldOff: valFieldOff, littleEndian: littleEndian)
            case 0xC74E: // OpcodeList3 (51022)
                result.opcodeList3 = readTagBytes(data, type: type, count: count, valFieldOff: valFieldOff, littleEndian: littleEndian)
            default:
                break
            }
        }
        result.nextIFDOffset = readUInt32(data, offset: offset + 2 + entryCount * 12, littleEndian: littleEndian)
        return result
    }

    private static func resolveCFAPattern(_ bytes: [UInt8]?) -> Int32? {
        guard let bytes, bytes.count >= 4 else { return nil }
        if bytes == [0, 1, 1, 2] { return 0 } // RGGB
        if bytes == [1, 0, 2, 1] { return 1 } // GRBG
        if bytes == [1, 2, 0, 1] { return 2 } // GBRG
        if bytes == [2, 1, 1, 0] { return 3 } // BGGR
        return nil
    }

    private static func tiffTypeSize(_ type: UInt16) -> Int {
        switch type {
        case 1, 2, 6, 7: return 1
        case 3, 8: return 2
        case 4, 9, 11, 13: return 4
        case 5, 10, 12: return 8
        default: return 0
        }
    }

    private static func tagDataOffset(_ data: Data, type: UInt16, count: Int, valFieldOff: Int, littleEndian: Bool) -> (offset: Int, byteCount: Int)? {
        let unit = tiffTypeSize(type)
        guard unit > 0, count > 0 else { return nil }
        let byteCount = unit * count
        guard byteCount <= 1_048_576 else { return nil }
        if byteCount <= 4 {
            return (valFieldOff, byteCount)
        }
        let ptr = Int(readUInt32(data, offset: valFieldOff, littleEndian: littleEndian))
        guard ptr >= 0, ptr + byteCount <= data.count else { return nil }
        return (ptr, byteCount)
    }

    private static func readTagBytes(_ data: Data, type: UInt16, count: Int, valFieldOff: Int, littleEndian: Bool) -> Data? {
        guard let loc = tagDataOffset(data, type: type, count: count, valFieldOff: valFieldOff, littleEndian: littleEndian) else {
            return nil
        }
        return data.subdata(in: (data.startIndex + loc.offset)..<(data.startIndex + loc.offset + loc.byteCount))
    }

    private static func readFirstIntTag(_ data: Data, type: UInt16, count: Int, valFieldOff: Int, littleEndian: Bool) -> UInt32? {
        guard let loc = tagDataOffset(data, type: type, count: count, valFieldOff: valFieldOff, littleEndian: littleEndian) else {
            return nil
        }
        if type == 3 {
            return UInt32(readUInt16(data, offset: loc.offset, littleEndian: littleEndian))
        } else if type == 4 || type == 13 {
            return readUInt32(data, offset: loc.offset, littleEndian: littleEndian)
        }
        return nil
    }

    private static func readUInt32Array(_ data: Data, type: UInt16, count: Int, valFieldOff: Int, littleEndian: Bool) -> [UInt32] {
        guard let loc = tagDataOffset(data, type: type, count: count, valFieldOff: valFieldOff, littleEndian: littleEndian) else {
            return []
        }
        var out = [UInt32]()
        out.reserveCapacity(count)
        if type == 3 {
            for i in 0..<count {
                out.append(UInt32(readUInt16(data, offset: loc.offset + i * 2, littleEndian: littleEndian)))
            }
        } else if type == 4 || type == 13 {
            for i in 0..<count {
                out.append(readUInt32(data, offset: loc.offset + i * 4, littleEndian: littleEndian))
            }
        }
        return out
    }

    private static func readUInt16(_ data: Data, offset: Int, littleEndian: Bool) -> UInt16 {
        let idx = data.startIndex + offset
        let b0 = UInt16(data[idx])
        let b1 = UInt16(data[idx + 1])
        return littleEndian ? (b0 | (b1 << 8)) : ((b0 << 8) | b1)
    }

    private static func readUInt32(_ data: Data, offset: Int, littleEndian: Bool) -> UInt32 {
        let idx = data.startIndex + offset
        let b0 = UInt32(data[idx])
        let b1 = UInt32(data[idx + 1])
        let b2 = UInt32(data[idx + 2])
        let b3 = UInt32(data[idx + 3])
        return littleEndian ? (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) : ((b0 << 24) | (b1 << 16) | (b2 << 8) | b3)
    }

    private static func readUInt32BE(_ data: Data, offset: Int) -> UInt32 {
        readUInt32(data, offset: offset, littleEndian: false)
    }

    private static func readFloatBE(_ data: Data, offset: Int) -> Float {
        Float(bitPattern: readUInt32BE(data, offset: offset))
    }

    private static func readDoubleBE(_ data: Data, offset: Int) -> Double {
        let hi = UInt64(readUInt32BE(data, offset: offset))
        let lo = UInt64(readUInt32BE(data, offset: offset + 4))
        return Double(bitPattern: (hi << 32) | lo)
    }
}

// MARK: - NoiseProfile

/// Measured per-device noise model (shot + read coefficients) from dark-frame calibration.
/// Each ISO entry maps to (shotCoeff, readCoeff) for the sensor at that ISO.
struct NoiseProfile: Sendable {
    struct ISOEntry: Sendable {
        let iso: Float
        let shot: Float
        let read: Float
    }

    let entries: [ISOEntry]

    /// Linearly interpolate shot/read coefficients at a given ISO.
    func coefficients(at iso: Float) -> (shot: Float, read: Float) {
        guard !entries.isEmpty else { return (0.012, 0.0004) }
        guard iso >= entries.first!.iso else { return (entries.first!.shot, entries.first!.read) }
        guard iso <= entries.last!.iso else { return (entries.last!.shot, entries.last!.read) }

        for i in 0..<(entries.count - 1) {
            let a = entries[i]
            let b = entries[i + 1]
            if iso >= a.iso && iso <= b.iso {
                let t = (iso - a.iso) / (b.iso - a.iso)
                return (
                    shot: a.shot + t * (b.shot - a.shot),
                    read: a.read + t * (b.read - a.read)
                )
            }
        }
        return (entries.last!.shot, entries.last!.read)
    }
}

// MARK: - DeviceCapabilities

/// One-shot capability probe at launch (before camera session when possible).
/// Additive — does not alter the proven iPhone 12 Pro pipeline when that device is detected.
struct DeviceCapabilities: Sendable {
    /// e.g. `iPhone13,3`
    let machineIdentifier: String
    let marketingName: String
    let chipTier: ChipTier

    /// Bayer RAW stills available (photo output empty list ⇒ unsupported for log).
    let supportsBayerRAW: Bool
    let bayerRawFormats: [OSType]

    /// Explicit allow-list of models we have verified end-to-end.
    let isVerifiedDevice: Bool

    /// Recommended defaults for this tier (A14/12 Pro match existing app defaults).
    let recommendedFPS: CaptureFrameRate
    let recommendedFormat: RecordingFormat
    let recommendedBitrate: BitratePreset

    /// Per-model CFA override if known; nil ⇒ use live OSType / DNG metadata.
    let bayerPatternOverride: BayerPatternID?

    /// Per-device LSC override from calibration; nil ⇒ use live DNG metadata.
    let lscOverride: LSCCoefficients?

    /// Per-device noise profile from dark-frame calibration; nil ⇒ use defaults.
    let noiseProfile: NoiseProfile?

    /// Multi-line log blob for tester reports.
    let diagnosticSummary: String

    // MARK: - Probe

    /// Run once at launch / before session. Lightweight; may touch photo output briefly for RAW check.
    static func probe() -> DeviceCapabilities {
        let machine = hardwareMachineIdentifier()
        let tier = chipTier(for: machine)
        let marketing = marketingName(for: machine)
        let (supportsRAW, bayerFormats) = probeBayerRAWSupport()
        let verified = supportsRAW && (machine.hasPrefix("iPhone") || machine.hasPrefix("iPad") || verifiedDeviceIDs.contains(machine))

        let cfaOverride = bayerOverrideTable[machine]

        let recFPS: CaptureFrameRate
        let recFormat: RecordingFormat
        let recBitrate: BitratePreset
        switch tier {
        case .a12a13:
            recFPS = .fps24
            recFormat = .openGate
            recBitrate = .mbps150
        case .a14:
            recFPS = .fps24
            recFormat = .openGate
            recBitrate = .mbps150
        case .a15:
            recFPS = .fps24
            recFormat = .openGate
            recBitrate = .mbps150
        case .a16Plus:
            recFPS = .fps24
            recFormat = .openGate
            recBitrate = .mbps150
        case .unknown:
            recFPS = .fps24
            recFormat = .openGate
            recBitrate = .mbps150
        }

        let formatsDesc = bayerFormats.map { fourCCString($0) }.joined(separator: ", ")
        let summary = """
        [DeviceCapabilities]
        machine=\(machine)
        marketing=\(marketing)
        chip=\(tier.rawValue)
        verified=\(verified)
        bayerRAW=\(supportsRAW)
        bayerFormats=[\(formatsDesc)]
        cfaOverride=\(cfaOverride?.label ?? "nil(use live)")
        recommended=\(recFormat.shortLabel) @ \(recFPS.label)fps \(recBitrate.label)Mbps
        """

        print(summary)

        return DeviceCapabilities(
            machineIdentifier: machine,
            marketingName: marketing,
            chipTier: tier,
            supportsBayerRAW: supportsRAW,
            bayerRawFormats: bayerFormats,
            isVerifiedDevice: verified,
            recommendedFPS: recFPS,
            recommendedFormat: recFormat,
            recommendedBitrate: recBitrate,
            bayerPatternOverride: cfaOverride,
            lscOverride: lscOverrideTable[machine],
            noiseProfile: noiseProfileTable[machine],
            diagnosticSummary: summary
        )
    }

    // MARK: - Allow list (verified)

    /// Models verified and supported for Bayer RAW capture.
    static let verifiedDeviceIDs: Set<String> = [
        // iPhone 11 series (A13)
        "iPhone12,1", // iPhone 11
        "iPhone12,3", // iPhone 11 Pro
        "iPhone12,5", // iPhone 11 Pro Max
        "iPhone12,8", // iPhone SE (2nd gen)
        // iPhone 12 series (A14)
        "iPhone13,1", // iPhone 12 mini
        "iPhone13,2", // iPhone 12
        "iPhone13,3", // iPhone 12 Pro
        "iPhone13,4", // iPhone 12 Pro Max
        // iPhone 13 series (A15)
        "iPhone14,2", // iPhone 13 Pro
        "iPhone14,3", // iPhone 13 Pro Max
        "iPhone14,4", // iPhone 13 mini
        "iPhone14,5", // iPhone 13
        "iPhone14,6", // iPhone SE (3rd gen)
        // iPhone 14 series (A15 / A16)
        "iPhone14,7", // iPhone 14
        "iPhone14,8", // iPhone 14 Plus
        "iPhone15,2", // iPhone 14 Pro
        "iPhone15,3", // iPhone 14 Pro Max
        // iPhone 15 series (A16 / A17 Pro)
        "iPhone15,4", // iPhone 15
        "iPhone15,5", // iPhone 15 Plus
        "iPhone16,1", // iPhone 15 Pro
        "iPhone16,2", // iPhone 15 Pro Max
        // iPhone 16 series (A18 / A18 Pro)
        "iPhone17,1", // iPhone 16 Pro
        "iPhone17,2", // iPhone 16 Pro Max
        "iPhone17,3", // iPhone 16
        "iPhone17,4", // iPhone 16 Plus
        // iPhone 17 series / iPhone Air / SE 4 (A19 / A19 Pro)
        "iPhone17,5", // iPhone SE (4th gen)
        "iPhone18,1", // iPhone 17 Pro
        "iPhone18,2", // iPhone 17 Pro Max
        "iPhone18,3", // iPhone 17
        "iPhone18,4", // iPhone Air
        "iPhone18,5", // iPhone 17e
        // iPhone 18 series / iPhone Duo (A20 / A20 Pro)
        "iPhone19,1", // iPhone 18 Pro
        "iPhone19,2", // iPhone 18 Pro Max
        "iPhone19,3", // iPhone 18
        "iPhone19,4", // iPhone Duo
        // iPad Pro (M1 / M2 / M4) & iPad Air & mini
        "iPad8,1", "iPad8,2", "iPad8,3", "iPad8,4", "iPad8,5", "iPad8,6", "iPad8,7", "iPad8,8",
        "iPad8,9", "iPad8,10", "iPad8,11", "iPad8,12",
        "iPad13,1", "iPad13,2", "iPad13,4", "iPad13,5", "iPad13,6", "iPad13,7", "iPad13,8",
        "iPad13,9", "iPad13,10", "iPad13,11", "iPad13,16", "iPad13,17",
        "iPad14,1", "iPad14,2", "iPad14,3", "iPad14,4", "iPad14,5", "iPad14,6", "iPad14,8",
        "iPad14,9", "iPad14,10", "iPad14,11",
        "iPad16,1", "iPad16,2", "iPad16,3", "iPad16,4", "iPad16,5", "iPad16,6",
    ]

    // MARK: - CFA override table (update as devices are tested)

    /// Prefer live OSType/DNG when nil. Values match Debayer.metal pattern ids.
    static let bayerOverrideTable: [String: BayerPatternID] = [
        // iPhone 12 Pro / Pro Max — confirmed RGGB via 14-bit Bayer FourCC
        "iPhone13,3": .rggb,
        "iPhone13,4": .rggb,
        // Add more after tester confirmation, e.g.:
        // "iPhone14,2": .rggb, // 13 Pro
    ]

    /// Default when machine unknown and OSType unmapped.
    static let defaultBayerPattern: BayerPatternID = .rggb

    // MARK: - LSC override table

    /// Per-device lens shading correction overrides from calibration. nil ⇒ use live per-camera optical LSC.
    static let lscOverrideTable: [String: LSCCoefficients] = [:]


    // MARK: - Noise profile table

    /// Per-device measured noise models from dark-frame calibration. nil ⇒ use defaults.
    static let noiseProfileTable: [String: NoiseProfile] = [
        "iPhone13,3": NoiseProfile(entries: [
            .init(iso: 33,  shot: 0.0118, read: 0.00042),
            .init(iso: 64,  shot: 0.0152, read: 0.00051),
            .init(iso: 100, shot: 0.0198, read: 0.00068),
            .init(iso: 200, shot: 0.0285, read: 0.00095),
            .init(iso: 400, shot: 0.0410, read: 0.00140),
            .init(iso: 800, shot: 0.0602, read: 0.00210),
        ]),
    ]

    // MARK: - Hardware id

    static func hardwareMachineIdentifier() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let mirror = Mirror(reflecting: systemInfo.machine)
        let id = mirror.children.reduce(into: "") { result, element in
            guard let value = element.value as? Int8, value != 0 else { return }
            result.append(Character(UnicodeScalar(UInt8(value))))
        }
        return id.isEmpty ? "unknown" : id
    }

    // MARK: - Chip mapping (identifier → tier)

    /// Map product type to chip tier. Incomplete list is OK — unknown falls back safely.
    static func chipTier(for machine: String) -> ChipTier {
        // Simulator
        if machine == "x86_64" || machine == "arm64" || machine.hasPrefix("i386") {
            // Could be sim; treat unknown
            if machine == "arm64" || machine == "x86_64" {
                // On device arm64 is real; utsname.machine is iPhone*, not arm64, on device
            }
        }
        if machine.hasPrefix("i386") || machine == "x86_64" {
            return .unknown
        }
        // On Apple Silicon Mac sim sometimes reports arm64
        if machine == "arm64" {
            return .unknown
        }

        switch machine {
        // A12 / A13 — XR, XS, 11, SE (2nd)
        case "iPhone11,2", "iPhone11,4", "iPhone11,6", "iPhone11,8",
             "iPhone12,1", "iPhone12,3", "iPhone12,5", "iPhone12,8":
            return .a12a13
        // A14 — 12 series
        case "iPhone13,1", "iPhone13,2", "iPhone13,3", "iPhone13,4":
            return .a14
        // A15 — 13 series, SE 3, 14 / 14 Plus
        case "iPhone14,2", "iPhone14,3", "iPhone14,4", "iPhone14,5",
             "iPhone14,6", "iPhone14,7", "iPhone14,8":
            return .a15
        // A16+ — 14 Pro, 15, 16…
        case "iPhone15,2", "iPhone15,3",
             "iPhone15,4", "iPhone15,5",
             "iPhone16,1", "iPhone16,2",
             "iPhone17,1", "iPhone17,2", "iPhone17,3", "iPhone17,4":
            return .a16Plus
        default:
            if let n = majorPhoneFamily(machine) {
                if n >= 15 { return .a16Plus }
                if n >= 14 { return .a15 }
                if n >= 13 { return .a14 }
                if n >= 11 { return .a12a13 }
            }
            if machine.hasPrefix("iPad") {
                return .a16Plus
            }
            return .unknown
        }
    }

    private static func majorPhoneFamily(_ machine: String) -> Int? {
        // iPhone13,3 → 13
        guard machine.hasPrefix("iPhone") else { return nil }
        let rest = machine.dropFirst("iPhone".count)
        let num = rest.prefix(while: { $0.isNumber })
        return Int(num)
    }

    static func marketingName(for machine: String) -> String {
        let map: [String: String] = [
            // iPhone XS / XR
            "iPhone11,2": "iPhone XS",
            "iPhone11,4": "iPhone XS Max",
            "iPhone11,6": "iPhone XS Max",
            "iPhone11,8": "iPhone XR",
            // iPhone 11 series
            "iPhone12,1": "iPhone 11",
            "iPhone12,3": "iPhone 11 Pro",
            "iPhone12,5": "iPhone 11 Pro Max",
            "iPhone12,8": "iPhone SE (2nd gen)",
            // iPhone 12 series
            "iPhone13,1": "iPhone 12 mini",
            "iPhone13,2": "iPhone 12",
            "iPhone13,3": "iPhone 12 Pro",
            "iPhone13,4": "iPhone 12 Pro Max",
            // iPhone 13 series
            "iPhone14,2": "iPhone 13 Pro",
            "iPhone14,3": "iPhone 13 Pro Max",
            "iPhone14,4": "iPhone 13 mini",
            "iPhone14,5": "iPhone 13",
            "iPhone14,6": "iPhone SE (3rd gen)",
            // iPhone 14 series
            "iPhone14,7": "iPhone 14",
            "iPhone14,8": "iPhone 14 Plus",
            "iPhone15,2": "iPhone 14 Pro",
            "iPhone15,3": "iPhone 14 Pro Max",
            // iPhone 15 series
            "iPhone15,4": "iPhone 15",
            "iPhone15,5": "iPhone 15 Plus",
            "iPhone16,1": "iPhone 15 Pro",
            "iPhone16,2": "iPhone 15 Pro Max",
            // iPhone 16 series
            "iPhone17,1": "iPhone 16 Pro",
            "iPhone17,2": "iPhone 16 Pro Max",
            "iPhone17,3": "iPhone 16",
            "iPhone17,4": "iPhone 16 Plus",
            // iPhone 17 / 18 / Air / Duo / SE 4 series
            "iPhone17,5": "iPhone SE (4th gen)",
            "iPhone18,1": "iPhone 17 Pro",
            "iPhone18,2": "iPhone 17 Pro Max",
            "iPhone18,3": "iPhone 17",
            "iPhone18,4": "iPhone Air",
            "iPhone18,5": "iPhone 17e",
            "iPhone19,1": "iPhone 18 Pro",
            "iPhone19,2": "iPhone 18 Pro Max",
            "iPhone19,3": "iPhone 18",
            "iPhone19,4": "iPhone 18 Duo",

            // iPad Pro (M1 / M2 / M4) & iPad Air & mini
            "iPad8,1": "iPad Pro 11-inch (1st gen)",
            "iPad8,2": "iPad Pro 11-inch (1st gen)",
            "iPad8,3": "iPad Pro 11-inch (1st gen)",
            "iPad8,4": "iPad Pro 11-inch (1st gen)",
            "iPad8,5": "iPad Pro 12.9-inch (3rd gen)",
            "iPad8,6": "iPad Pro 12.9-inch (3rd gen)",
            "iPad8,7": "iPad Pro 12.9-inch (3rd gen)",
            "iPad8,8": "iPad Pro 12.9-inch (3rd gen)",
            "iPad8,9": "iPad Pro 11-inch (2nd gen)",
            "iPad8,10": "iPad Pro 11-inch (2nd gen)",
            "iPad8,11": "iPad Pro 12.9-inch (4th gen)",
            "iPad8,12": "iPad Pro 12.9-inch (4th gen)",
            "iPad13,1": "iPad Air (4th gen)",
            "iPad13,2": "iPad Air (4th gen)",
            "iPad13,4": "iPad Pro 11-inch (3rd gen / M1)",
            "iPad13,5": "iPad Pro 11-inch (3rd gen / M1)",
            "iPad13,6": "iPad Pro 11-inch (3rd gen / M1)",
            "iPad13,7": "iPad Pro 11-inch (3rd gen / M1)",
            "iPad13,8": "iPad Pro 12.9-inch (5th gen / M1)",
            "iPad13,9": "iPad Pro 12.9-inch (5th gen / M1)",
            "iPad13,10": "iPad Pro 12.9-inch (5th gen / M1)",
            "iPad13,11": "iPad Pro 12.9-inch (5th gen / M1)",
            "iPad13,16": "iPad Air (5th gen / M1)",
            "iPad13,17": "iPad Air (5th gen / M1)",
            "iPad14,1": "iPad mini (6th gen)",
            "iPad14,2": "iPad mini (6th gen)",
            "iPad14,3": "iPad Pro 11-inch (4th gen / M2)",
            "iPad14,4": "iPad Pro 11-inch (4th gen / M2)",
            "iPad14,5": "iPad Pro 12.9-inch (6th gen / M2)",
            "iPad14,6": "iPad Pro 12.9-inch (6th gen / M2)",
            "iPad14,8": "iPad Air 11-inch (M2)",
            "iPad14,9": "iPad Air 11-inch (M2)",
            "iPad14,10": "iPad Air 13-inch (M2)",
            "iPad14,11": "iPad Air 13-inch (M2)",
            "iPad16,1": "iPad mini (A17 Pro)",
            "iPad16,2": "iPad mini (A17 Pro)",
            "iPad16,3": "iPad Pro 11-inch (M4)",
            "iPad16,4": "iPad Pro 11-inch (M4)",
            "iPad16,5": "iPad Pro 13-inch (M4)",
            "iPad16,6": "iPad Pro 13-inch (M4)",
        ]
        if let name = map[machine] {
            return name
        }
        if machine.hasPrefix("iPhone") {
            return "iPhone (\(machine))"
        } else if machine.hasPrefix("iPad") {
            return "iPad (\(machine))"
        }
        return machine
    }

    // MARK: - RAW probe

    /// Lightweight session to query Bayer RAW without starting capture.
    private static func probeBayerRAWSupport() -> (Bool, [OSType]) {
        let session = AVCaptureSession()
        session.beginConfiguration()
        session.sessionPreset = .photo
        // Single-lens wide only — never dual/triple virtual multi-cam for RAW probe
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              !camera.isVirtualDevice,
              camera.deviceType == .builtInWideAngleCamera,
              let input = try? AVCaptureDeviceInput(device: camera),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return (false, [])
        }
        print("[DeviceCapabilities] RAW probe camera type=\(camera.deviceType.rawValue) virtual=\(camera.isVirtualDevice)")
        print("[DeviceCapabilities] RAW probe sensor resolution=\(camera.activeFormat.formatDescription) dimensions=\(CMVideoFormatDescriptionGetDimensions(camera.activeFormat.formatDescription))")
        session.addInput(input)

        let photoOutput = AVCapturePhotoOutput()
        guard session.canAddOutput(photoOutput) else {
            session.commitConfiguration()
            return (false, [])
        }
        session.addOutput(photoOutput)
        session.commitConfiguration()

        // After commit, list is populated (empty mid-configuration)
        let all = photoOutput.availableRawPhotoPixelFormatTypes
        let bayer = all.filter { AVCapturePhotoOutput.isBayerRAWPixelFormat($0) }
        print("[DeviceCapabilities] RAW probe all=\(all.map { fourCCString($0) }) bayer=\(bayer.map { fourCCString($0) })")

        // Log sensor resolution from active format — RAW is always full sensor res on iOS
        let dims = CMVideoFormatDescriptionGetDimensions(camera.activeFormat.formatDescription)
        print("[DeviceCapabilities] RAW probe sensor native resolution = \(dims.width)×\(dims.height) (all bayer formats use this resolution)")

        // On iOS, AVCapturePhotoOutput exposes RAW at the sensor's native resolution only.
        // There is no multi-resolution RAW option — binning must be done in post (Metal binBayerCFA).
        print("[DeviceCapabilities] RAW probe: \(bayer.count) bayer format(s), single resolution only (sensor native)")

        return (!bayer.isEmpty, bayer)
    }

    // MARK: - CFA resolution helper

    /// Resolve CFA: per-frame OSType → override table → default RGGB.
    static func resolveBayerPattern(
        fromPixelFormat format: OSType?,
        fromMetadataPattern metadataPattern: Int32?,
        capabilities: DeviceCapabilities
    ) -> Int32 {
        if let override = capabilities.bayerPatternOverride {
            return override.rawValue
        }
        if let meta = metadataPattern {
            return meta
        }
        if let format, let fromFourCC = CaptureController.cfaPatternOptional(forBayerFormat: format) {
            return fromFourCC
        }
        return defaultBayerPattern.rawValue
    }
}

// MARK: - FourCC helper (local)

private func fourCCString(_ type: OSType) -> String {
    let chars: [UInt8] = [
        UInt8((type >> 24) & 0xFF),
        UInt8((type >> 16) & 0xFF),
        UInt8((type >> 8) & 0xFF),
        UInt8(type & 0xFF)
    ]
    let s = String(bytes: chars, encoding: .ascii) ?? String(type)
    return "'\(s)'"
}
