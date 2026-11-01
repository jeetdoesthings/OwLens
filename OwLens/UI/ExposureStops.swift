import Foundation

/// Discrete stops for ISO / WB plus snap targets for shutter angle.
enum ExposureStops {
    /// Kelvin white-balance stops (coarse is fine for WB).
    static let wbAll: [Float] = [
        2500, 2800, 3000, 3200, 3500, 4000, 4500, 5000,
        5600, 6000, 6500, 7000, 7500, 8000, 9000, 10000
    ]

    /// Standard cinematic 1/3-stop ISO stops
    static let cinemaISOStops: [Float] = [
        32, 40, 50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200
    ]

    /// Standard cinematic 1/3-stop ISO values within the device's supported range.
    static func isoStops(in range: ClosedRange<Float>) -> [Float] {
        let filtered = cinemaISOStops.filter { range.contains($0) }
        return filtered.isEmpty ? [range.lowerBound, range.upperBound] : filtered
    }

    /// Standard cinematic shutter angles
    static let shutterAngles: [Float] = [
        360.0, 270.0, 216.0, 180.0, 172.8, 144.0, 90.0, 45.0, 22.5, 11.25
    ]
    static func wbStops(in range: ClosedRange<Float> = 2000...10000) -> [Float] {
        let filtered = wbAll.filter { range.contains($0) }
        return filtered.isEmpty ? [range.lowerBound, range.upperBound] : filtered
    }

    /// Inclusive stepped list, rounded to step grid.
    static func stepped(in range: ClosedRange<Float>, step: Float) -> [Float] {
        guard step > 0 else { return [range.lowerBound] }
        let lo = (range.lowerBound / step).rounded(.up) * step
        let hi = range.upperBound
        var values: [Float] = []
        var v = lo
        // Cap count so sliders stay usable (very wide ISO range)
        let maxCount = 400
        while v <= hi + 0.001 && values.count < maxCount {
            values.append((v * 10).rounded() / 10) // tidy float noise
            v += step
        }
        if values.isEmpty {
            values = [range.lowerBound, range.upperBound]
        } else {
            // Ensure endpoints reachable if they fell off grid
            if let first = values.first, first > range.lowerBound + step * 0.5 {
                values.insert(range.lowerBound, at: 0)
            }
            if let last = values.last, last < range.upperBound - step * 0.5 {
                values.append(range.upperBound)
            }
        }
        return values
    }

    static func nearestIndex(in stops: [Float], to value: Float) -> Int {
        guard !stops.isEmpty else { return 0 }
        var best = 0
        var bestDist = abs(stops[0] - value)
        for i in 1..<stops.count {
            let d = abs(stops[i] - value)
            if d < bestDist {
                bestDist = d
                best = i
            }
        }
        return best
    }

    static func clampIndex(_ index: Int, count: Int) -> Int {
        guard count > 0 else { return 0 }
        return min(max(0, index), count - 1)
    }
}
