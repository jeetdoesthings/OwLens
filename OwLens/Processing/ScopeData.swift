import Foundation

struct ScopeData: Equatable {
    let histogramRed: [Float]
    let histogramGreen: [Float]
    let histogramBlue: [Float]
    let waveform: [Float]
    let waveformColumns: Int
    let waveformRows: Int

    var histogram: [Float] {
        zip(zip(histogramRed, histogramGreen), histogramBlue).map { channels in
            max(channels.0.0, channels.0.1, channels.1)
        }
    }

    static let empty = ScopeData(
        histogramRed: Array(repeating: 0, count: 64),
        histogramGreen: Array(repeating: 0, count: 64),
        histogramBlue: Array(repeating: 0, count: 64),
        waveform: Array(repeating: 0, count: 64 * 32),
        waveformColumns: 64,
        waveformRows: 32
    )

    static func make(
        fromHalfRGBA pixels: [UInt16],
        width: Int,
        height: Int,
        histogramBins: Int = 64,
        waveformColumns: Int = 64,
        waveformRows: Int = 32
    ) -> ScopeData {
        guard width > 0, height > 0, pixels.count >= width * height * 4 else {
            return .empty
        }

        var histogramRed = [Float](repeating: 0, count: histogramBins)
        var histogramGreen = [Float](repeating: 0, count: histogramBins)
        var histogramBlue = [Float](repeating: 0, count: histogramBins)
        var waveform = [Float](repeating: 0, count: waveformColumns * waveformRows)

        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                let r = min(1, max(0, Float(Float16(bitPattern: pixels[i]))))
                let g = min(1, max(0, Float(Float16(bitPattern: pixels[i + 1]))))
                let b = min(1, max(0, Float(Float16(bitPattern: pixels[i + 2]))))
                let luma = min(1, max(0, 0.2126 * r + 0.7152 * g + 0.0722 * b))

                histogramRed[min(histogramBins - 1, Int(r * Float(histogramBins - 1)))] += 1
                histogramGreen[min(histogramBins - 1, Int(g * Float(histogramBins - 1)))] += 1
                histogramBlue[min(histogramBins - 1, Int(b * Float(histogramBins - 1)))] += 1

                let col = min(waveformColumns - 1, x * waveformColumns / width)
                let row = waveformRows - 1 - min(waveformRows - 1, Int(luma * Float(waveformRows - 1)))
                waveform[row * waveformColumns + col] += 1
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

    private static func normalize(_ values: inout [Float]) {
        guard let maxValue = values.max(), maxValue > 0 else { return }
        for i in values.indices {
            values[i] = min(1, values[i] / maxValue)
        }
    }
}
