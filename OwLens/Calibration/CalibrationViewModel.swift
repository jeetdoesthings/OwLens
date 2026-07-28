import SwiftUI
import Combine

/// Source of RAW frames and hardware control for calibration.
@MainActor
protocol CalibrationFrameProvider: AnyObject {
    /// The machine identifier of the current device (e.g. `iPhone13,3`).
    var deviceMachine: String { get }

    /// Capture `count` RAW frames at the requested ISO. Returns nil if capture fails.
    func captureFrames(iso: Float, count: Int) async -> [RawFrameData]

    /// Current selected lens for calibration.
    var selectedLens: LensOption? { get }
}

@MainActor
final class CalibrationViewModel: ObservableObject {
    @Published var state: State = .idle

    enum State: Equatable {
        case idle
        case instructions
        case capturing(progress: Float)
        case processing
        case done
        case failed(String)

        static func == (lhs: State, rhs: State) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.instructions, .instructions), (.processing, .processing), (.done, .done):
                return true
            case let (.capturing(a), .capturing(b)): return a == b
            case let (.failed(a), .failed(b)): return a == b
            default: return false
            }
        }
    }

    weak var frameProvider: CalibrationFrameProvider?

    private let isoValues: [Float] = [33, 64, 100, 200, 400, 800]
    private let framesPerISO = 20
    private var cancellables = Set<AnyCancellable>()

    func configure(with provider: CalibrationFrameProvider) {
        self.frameProvider = provider
    }

    func showInstructions() {
        guard case .idle = state else { return }
        state = .instructions
    }

    func startCalibration() async {
        guard let provider = frameProvider,
              let lens = provider.selectedLens else {
            state = .failed("No lens selected for calibration")
            return
        }

        let machine = provider.deviceMachine
        let lensID = lens.uniqueID
        var entries: [NoiseCalibrationStore.NoiseEntry] = []

        state = .capturing(progress: 0)
        print("[Calibration] Starting calibration for \(lens.name) machine=\(machine) lensID=\(lensID)")

        for (index, iso) in isoValues.enumerated() {
            let progress = Float(index) / Float(isoValues.count)
            await MainActor.run { state = .capturing(progress: progress) }

            print("[Calibration] Capturing \(framesPerISO) frames at ISO \(iso)...")
            let frames = await provider.captureFrames(iso: iso, count: framesPerISO)
            print("[Calibration] ISO \(iso): got \(frames.count)/\(framesPerISO) frames")

            guard frames.count >= framesPerISO / 2 else {
                let msg = "Failed to capture frames at ISO \(Int(iso)) (got \(frames.count)/\(framesPerISO))"
                print("[Calibration] \(msg)")
                await MainActor.run { state = .failed(msg) }
                return
            }

            let stats = computeNoiseStats(frames: frames)
            entries.append(NoiseCalibrationStore.NoiseEntry(iso: iso, shot: stats.shot, read: stats.read))
            print("[Calibration] ISO \(iso): shot=\(String(format: "%.5f", stats.shot)) read=\(String(format: "%.6f", stats.read))")
        }

        await MainActor.run { state = .processing }

        // Flat-field frame for LSC and green balance.
        let flatFrames = await provider.captureFrames(iso: 100, count: 1)
        let (lsc, greenBalance) = computeFlatField(from: flatFrames.first)
        print("[Calibration] Flat field: lsc=\(lsc != nil ? "yes" : "no") greenBalance=\(String(format: "%.4f", greenBalance ?? 1.0))")

        let saved = NoiseCalibrationStore.shared.setCalibration(
            noiseEntries: entries,
            lensShading: lsc,
            greenBalance: greenBalance,
            deviceMachine: machine,
            lensID: lensID
        )
        print("[Calibration] Save result: \(saved ? "saved" : "FAILED")")

        await MainActor.run { state = .done }
    }

    func reset() {
        state = .idle
    }

    // MARK: - Noise model fitting

    private func computeNoiseStats(frames: [RawFrameData]) -> (shot: Float, read: Float) {
        // Collect (mean, variance) data points across all frames using a tiled grid.
        // This gives us many signal-vs-noise samples for a robust linear fit.
        var dataPoints: [(mean: Float, variance: Float)] = []

        for frame in frames {
            let pts = tiledStats(pixelBuffer: frame.pixelBuffer, gridSize: 4)
            dataPoints.append(contentsOf: pts)
        }

        guard !dataPoints.isEmpty else { return (0.012, 0.0004) }

        // Fit variance = shot * signal + read using least squares.
        // variance = shot * mean + read
        // We want shot and read such that sum((variance_i - shot*mean_i - read)^2) is minimized.
        // Using normal equations:
        // [sum(mean^2)  sum(mean)] [shot]   [sum(mean * variance)]
        // [sum(mean)    count    ] [read] = [sum(variance)       ]

        var sumM: Double = 0
        var sumV: Double = 0
        var sumMM: Double = 0
        var sumMV: Double = 0
        let count = Double(dataPoints.count)

        for pt in dataPoints {
            let m = Double(pt.mean)
            let v = Double(pt.variance)
            sumM += m
            sumV += v
            sumMM += m * m
            sumMV += m * v
        }

        let det = count * sumMM - sumM * sumM
        guard abs(det) > 1e-10 else { return (0.012, 0.0004) }

        let shot = Float((count * sumMV - sumM * sumV) / det)
        let read = Float((sumMM * sumV - sumM * sumMV) / det)

        return (
            shot: max(shot, 0.0),
            read: max(read, 1e-6)
        )
    }

    /// Sample a grid of tiles across the sensor, returning (mean, variance) per tile.
    private func tiledStats(pixelBuffer: CVPixelBuffer, gridSize: Int = 4) -> [(mean: Float, variance: Float)] {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return [] }

        let tileW = width / gridSize
        let tileH = height / gridSize
        var results: [(mean: Float, variance: Float)] = []

        for ty in 0..<gridSize {
            for tx in 0..<gridSize {
                let x0 = tx * tileW
                let y0 = ty * tileH
                var sum: Double = 0
                var sum2: Double = 0
                var count = 0

                for y in y0..<(y0 + tileH) {
                    let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt16.self)
                    for x in x0..<(x0 + tileW) {
                        let val = Float(row[x]) / 65535.0
                        sum += Double(val)
                        sum2 += Double(val) * Double(val)
                        count += 1
                    }
                }

                guard count > 0 else { continue }
                let mean = Float(sum / Double(count))
                let var_ = Float(max(sum2 / Double(count) - (sum / Double(count)) * (sum / Double(count)), 0))
                results.append((mean, var_))
            }
        }

        return results
    }

    // MARK: - Flat field

    private func computeFlatField(from frame: RawFrameData?) -> (lsc: LSCCoefficients?, greenBalance: Float?) {
        guard let frame = frame else { return (nil, nil) }
        return deriveLSCAndGreenBalance(pixelBuffer: frame.pixelBuffer, cfaPattern: frame.cfaPattern)
    }

    /// Derive radial LSC coefficients and Gr/Gb green balance from a flat-field frame.
    /// Splits Bayer pixels into R/Gr/Gb/B groups, measures radial falloff per channel,
    /// and computes the Gr/Gb ratio.
    private func deriveLSCAndGreenBalance(pixelBuffer: CVPixelBuffer, cfaPattern: Int32) -> (lsc: LSCCoefficients?, greenBalance: Float?) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return (nil, nil) }

        // Determine Bayer phase: (xEven, yEven) mapping per pattern
        // 0=RGGB: R(xEven,yEven), Gr(xOdd,yEven), Gb(xEven,yOdd), B(xOdd,yOdd)
        // 1=GRBG: G(xEven,yEven), R(xOdd,yEven), B(xEven,yOdd), G(xOdd,yOdd)
        // 2=GBRG: G(xEven,yEven), B(xOdd,yEven), R(xEven,yOdd), G(xOdd,yOdd)
        // 3=BGGR: B(xEven,yEven), G(xOdd,yEven), G(xEven,yOdd), R(xOdd,yOdd)
        let pattern = Int(cfaPattern)

        // Collect samples per Bayer quadrant divided into 4 radial bins (center, mid, outer, edge).
        var rSamples: [[Float]] = [[], [], [], []]
        var grSamples: [[Float]] = [[], [], [], []]
        var gbSamples: [[Float]] = [[], [], [], []]
        var bSamples: [[Float]] = [[], [], [], []]

        let cx = Float(width) / 2.0
        let cy = Float(height) / 2.0
        let maxR = sqrt(cx * cx + cy * cy)

        let step = 4 // sample every 2x2 block (4-pixel stride)
        for y in stride(from: 0, to: height - 1, by: step) {
            let row0 = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt16.self)
            let row1 = base.advanced(by: (y + 1) * bytesPerRow).assumingMemoryBound(to: UInt16.self)
            for x in stride(from: 0, to: width - 1, by: step) {
                // Read the 2x2 Bayer block
                let v00 = Float(row0[x]) / 65535.0     // (0,0)
                let v01 = Float(row0[x + 1]) / 65535.0 // (1,0)
                let v10 = Float(row1[x]) / 65535.0     // (0,1)
                let v11 = Float(row1[x + 1]) / 65535.0 // (1,1)

                // Map 2x2 block positions to colors based on CFA pattern
                // (0,0)=top-left, (1,0)=top-right, (0,1)=bottom-left, (1,1)=bottom-right
                let (c00, c01, c10, c11): (Int, Int, Int, Int) // 0=R,1=Gr,2=Gb,3=B
                switch pattern {
                case 0: // RGGB
                    c00 = 0; c01 = 1; c10 = 2; c11 = 3 // R Gr Gb B
                case 1: // GRBG
                    c00 = 1; c01 = 0; c10 = 3; c11 = 2 // Gr R B Gb
                case 2: // GBRG
                    c00 = 2; c01 = 3; c10 = 0; c11 = 1 // Gb B R Gr
                case 3: // BGGR
                    c00 = 3; c01 = 2; c10 = 1; c11 = 0 // B Gb Gr R
                default: continue
                }

                // Radial distance for the block center
                let cx_block = Float(x + 1)
                let cy_block = Float(y + 1)
                let dx = cx_block - cx
                let dy = cy_block - cy
                let r = sqrt(dx * dx + dy * dy) / maxR
                let bin = min(3, Int(r * 4.0))

                func addToChannel(_ color: Int, _ val: Float) {
                    switch color {
                    case 0: rSamples[bin].append(val)
                    case 1: grSamples[bin].append(val)
                    case 2: gbSamples[bin].append(val)
                    case 3: bSamples[bin].append(val)
                    default: break
                    }
                }
                addToChannel(c00, v00)
                addToChannel(c01, v01)
                addToChannel(c10, v10)
                addToChannel(c11, v11)
            }
        }

        // Compute mean per radial bin per channel
        func mean(_ arr: [Float]) -> Float {
            guard !arr.isEmpty else { return 0 }
            return arr.reduce(0, +) / Float(arr.count)
        }

        let rMeans = rSamples.map(mean)
        let grMeans = grSamples.map(mean)
        let gbMeans = gbSamples.map(mean)
        let bMeans = bSamples.map(mean)

        // Green balance: Gr/Gb ratio from center bin (bin 0)
        let greenBalance = (grMeans[0] > 0 && gbMeans[0] > 0) ? grMeans[0] / gbMeans[0] : 1.0

        // Fit radial coefficients: channelNormalized = 1 + radial * r^2
        // where r^2 is normalized distance squared, and radial is the correction.
        // For a radial falloff, the correction gain = 1 + coeff * r^2
        // We solve for coeff such that at r=1 (corner), the gain = centerBinMean / cornerBinMean
        // gain = 1 + coeff * 1 = cornerGain, so coeff = cornerGain - 1
        func fitRadialCoeff(centerVal: Float, edgeVal: Float) -> Float {
            guard centerVal > 0, edgeVal > 0 else { return 0 }
            // The edge (bin 3, r ~0.875) has gain = center/edge
            // coeff = (center/edge - 1) / r^2 where r^2 ≈ 0.875^2
            let invGain = centerVal / edgeVal // >1 if vignette
            let r2: Float = 0.765 // approximate r^2 for farthest bin
            let coeff = (invGain - 1.0) / r2
            return max(coeff, 0) // only positive correction for vignette
        }

        let useCenter = rMeans[0] > 0
        if useCenter {
            // Average bins 0 and 1 for center, bins 2 and 3 weighted for edge
            let rCenter = (rMeans[0] + rMeans[1]) * 0.5
            let rEdge = (rMeans[2] * 0.3 + rMeans[3] * 0.7)
            let gCenter = ((grMeans[0] + gbMeans[0]) + (grMeans[1] + gbMeans[1]) * 0.5) * 0.5
            let gEdge = ((grMeans[2] + gbMeans[2]) * 0.3 + (grMeans[3] + gbMeans[3]) * 0.7)
            let bCenter = (bMeans[0] + bMeans[1]) * 0.5
            let bEdge = (bMeans[2] * 0.3 + bMeans[3] * 0.7)

            let lsc = LSCCoefficients(
                radialR: fitRadialCoeff(centerVal: rCenter, edgeVal: rEdge),
                radialG: fitRadialCoeff(centerVal: gCenter, edgeVal: gEdge),
                radialB: fitRadialCoeff(centerVal: bCenter, edgeVal: bEdge)
            )
            return (lsc, greenBalance)
        }

        return (nil, greenBalance)
    }
}
