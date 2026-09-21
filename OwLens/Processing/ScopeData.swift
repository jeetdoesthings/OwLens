import Foundation

struct ScopeData: Equatable {
    let histogramRed: [Float]
    let histogramGreen: [Float]
    let histogramBlue: [Float]
    let waveform: [Float]
    let waveformColumns: Int
    let waveformRows: Int


    static let empty = ScopeData(
        histogramRed: Array(repeating: 0, count: 64),
        histogramGreen: Array(repeating: 0, count: 64),
        histogramBlue: Array(repeating: 0, count: 64),
        waveform: Array(repeating: 0, count: 64 * 48),
        waveformColumns: 64,
        waveformRows: 48
    )

    static func make(
        fromHalfRGBA pixels: [UInt16],
        width: Int,
        height: Int,
        histogramBins: Int = 64,
        waveformColumns: Int = 64,
        waveformRows: Int = 48
    ) -> ScopeData {
        guard width > 0, height > 0, pixels.count >= width * height * 4 else {
            return .empty
        }

        var histogramRed = [Float](repeating: 0, count: histogramBins)
        var histogramGreen = [Float](repeating: 0, count: histogramBins)
        var histogramBlue = [Float](repeating: 0, count: histogramBins)
        var waveform = [Float](repeating: 0, count: waveformColumns * waveformRows)

        let maxHistBin = histogramBins - 1
        let histScale = Float(maxHistBin)
        let maxWaveRow = waveformRows - 1
        let waveScale = Float(maxWaveRow)

        // Precompute column mapping: O(W) with zero heap allocations via stack/temporary buffer
        return withUnsafeTemporaryAllocation(of: Int.self, capacity: width) { colMap in
            for x in 0..<width {
                colMap[x] = min(waveformColumns - 1, x * waveformColumns / width)
            }

            pixels.withUnsafeBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                var idx = 0
                for _ in 0..<height {
                    for x in 0..<width {
                        let r16 = Float16(bitPattern: base[idx])
                        let g16 = Float16(bitPattern: base[idx + 1])
                        let b16 = Float16(bitPattern: base[idx + 2])
                        idx += 4

                        let r = min(1.0, max(0.0, Float(r16)))
                        let g = min(1.0, max(0.0, Float(g16)))
                        let b = min(1.0, max(0.0, Float(b16)))
                        let luma = min(1.0, max(0.0, 0.2627 * r + 0.6780 * g + 0.0593 * b))

                        let rBin = min(maxHistBin, Int(r * histScale))
                        let gBin = min(maxHistBin, Int(g * histScale))
                        let bBin = min(maxHistBin, Int(b * histScale))
                        histogramRed[rBin] += 1
                        histogramGreen[gBin] += 1
                        histogramBlue[bBin] += 1

                        let col = colMap[x]
                        let row = maxWaveRow - min(maxWaveRow, Int(luma * waveScale))
                        waveform[row * waveformColumns + col] += 1
                    }
                }
            }

            normalize(&histogramRed)
            normalize(&histogramGreen)
            normalize(&histogramBlue)
            normalize(&waveform)

            return ScopeData(
                histogramRed: histogramRed,
                histogramGreen: histogramGreen,
                histogramBlue: histogramBlue,
                waveform: waveform,
                waveformColumns: waveformColumns,
                waveformRows: waveformRows
            )
        }
    }

    private static func normalize(_ values: inout [Float]) {
        guard let maxValue = values.max(), maxValue > 0 else { return }
        let invMax = 1.0 / maxValue
        for i in values.indices {
            values[i] = min(1.0, values[i] * invMax)
        }
    }
}

/// Dedicated observable object holding live exposure scopes (histogram and waveform).
/// Isolates 10 Hz scope updates to `ScopesContainerView`, preventing camera HUD invalidations.
@MainActor
final class ScopeMonitor: ObservableObject {
    @Published private(set) var scopeData: ScopeData = .empty

    func update(_ data: ScopeData) {
        scopeData = data
    }

    func reset() {
        if scopeData != .empty {
            scopeData = .empty
        }
    }
}

