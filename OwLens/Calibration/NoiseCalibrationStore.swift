import Foundation

/// Persisted per-lens calibration data: noise model, LSC coefficients, and Gr/Gb balance.
final class NoiseCalibrationStore {
    static let shared = NoiseCalibrationStore()

    struct NoiseEntry: Codable {
        let iso: Float
        let shot: Float
        let read: Float
    }

    struct LensCalibration: Codable {
        var noiseEntries: [NoiseEntry]
        var lensShading: LSCCoefficients?
        var greenBalance: Float?
        var calibratedAt: Date
    }

    private var store: [String: LensCalibration] = [:]
    private let queue = DispatchQueue(label: "owlens.calibration.persistence")
    private let fileURL: URL

    private init() {
        let fm = FileManager.default
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = support.appendingPathComponent("OwLens", isDirectory: true)
            .appendingPathComponent("Calibration", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        fileURL = dir.appendingPathComponent("calibration.json")
        load()
    }

    func isCalibrated(deviceMachine: String, lensID: String) -> Bool {
        queue.sync { store[key(machine: deviceMachine, lensID: lensID)] != nil }
    }

    func noiseCoefficients(forISO iso: Float, deviceMachine: String, lensID: String) -> (shot: Float, read: Float)? {
        queue.sync {
            guard let cal = store[key(machine: deviceMachine, lensID: lensID)] else { return nil }
            return interpolate(entries: cal.noiseEntries, iso: iso)
        }
    }

    func lensShadingCoefficients(deviceMachine: String, lensID: String) -> LSCCoefficients? {
        queue.sync {
            store[key(machine: deviceMachine, lensID: lensID)]?.lensShading
        }
    }

    func greenBalance(deviceMachine: String, lensID: String) -> Float? {
        queue.sync {
            store[key(machine: deviceMachine, lensID: lensID)]?.greenBalance
        }
    }

    func setCalibration(
        noiseEntries: [NoiseEntry],
        lensShading: LSCCoefficients?,
        greenBalance: Float?,
        deviceMachine: String,
        lensID: String
    ) {
        queue.async {
            self.store[self.key(machine: deviceMachine, lensID: lensID)] = LensCalibration(
                noiseEntries: noiseEntries,
                lensShading: lensShading,
                greenBalance: greenBalance,
                calibratedAt: Date()
            )
            self.save()
        }
    }

    private func key(machine: String, lensID: String) -> String { "\(machine)_\(lensID)" }

    private func interpolate(entries: [NoiseEntry], iso: Float) -> (shot: Float, read: Float)? {
        guard !entries.isEmpty else { return nil }
        let sorted = entries.sorted { $0.iso < $1.iso }
        if iso <= sorted.first!.iso { return (sorted.first!.shot, sorted.first!.read) }
        if iso >= sorted.last!.iso  { return (sorted.last!.shot, sorted.last!.read) }
        for i in 0..<sorted.count-1 {
            let a = sorted[i], b = sorted[i+1]
            if iso >= a.iso && iso <= b.iso {
                let t = (iso - a.iso) / (b.iso - a.iso)
                return (
                    a.shot + t * (b.shot - a.shot),
                    a.read + t * (b.read - a.read)
                )
            }
        }
        return nil
    }

    private func load() {
        guard let data = try? Data(contentsOf: fileURL),
              let decoded = try? JSONDecoder().decode([String: LensCalibration].self, from: data) else { return }
        store = decoded
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(store) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }
}
