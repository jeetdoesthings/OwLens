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
        var means: [Float] = []
        var variances: [Float] = []

        for frame in frames {
            let (mean, variance) = centerRegionStats(pixelBuffer: frame.pixelBuffer)
            means.append(mean)
            variances.append(variance)
        }

        let meanSignal = means.reduce(0, +) / Float(means.count)
        let meanVariance = variances.reduce(0, +) / Float(variances.count)

        // Fit variance = shot * signal + read using the average of all captured frames.
        // Clamp read to a small positive value to avoid numerical issues.
        let signalNorm = max(meanSignal, 1e-4)
        let shot = max(meanVariance / signalNorm, 0.0)
        let read = max(meanVariance - shot * signalNorm, 1e-6)
        return (shot: shot, read: read)
    }

    private func centerRegionStats(pixelBuffer: CVPixelBuffer) -> (mean: Float, variance: Float) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return (0, 0) }

        // Center 50% region.
        let x0 = width / 4
        let y0 = height / 4
        let w = width / 2
        let h = height / 2

        var sum: Double = 0
        var sum2: Double = 0
        var count = 0

        // Bayer RAW is 16-bit single channel.
        for y in y0..<(y0 + h) {
            let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt16.self)
            for x in x0..<(x0 + w) {
                let value = Float(row[x]) / 65535.0
                sum += Double(value)
                sum2 += Double(value) * Double(value)
                count += 1
            }
        }

        let mean = Float(sum / Double(count))
        let variance = Float(max(sum2 / Double(count) - (sum / Double(count)) * (sum / Double(count)), 0))
        return (mean, variance)
    }

    // MARK: - Flat field

    private func computeFlatField(from frame: RawFrameData?) -> (lsc: LSCCoefficients?, greenBalance: Float?) {
        guard let frame = frame else { return (nil, nil) }
        // TODO: derive true flat-field LSC and Gr/Gb ratio from per-quadrant averages.
        // For now, default to neutral so calibration does not degrade the image.
        return (nil, 1.0)
    }
}
