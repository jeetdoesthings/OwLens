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
        let schemaVersion: Int
        var noiseEntries: [NoiseEntry]
        var lensShading: LSCCoefficients?
        var greenBalance: Float?
        var calibratedAt: Date
        /// ISO ladder used during capture, for reference.
        let isoValues: [Float]

        init(noiseEntries: [NoiseEntry], lensShading: LSCCoefficients?, greenBalance: Float?, isoValues: [Float]) {
            self.schemaVersion = 1
            self.noiseEntries = noiseEntries
            self.lensShading = lensShading
            self.greenBalance = greenBalance
            self.isoValues = isoValues
            self.calibratedAt = Date()
        }
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

    /// Load the full calibration for a device+lens combo.
    func loadCalibration(deviceMachine: String, lensID: String) -> LensCalibration? {
        queue.sync { store[key(machine: deviceMachine, lensID: lensID)] }
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

    /// Save calibration data. Returns true on success, false on serialization failure.
    @discardableResult
    func setCalibration(
        noiseEntries: [NoiseEntry],
        lensShading: LSCCoefficients?,
        greenBalance: Float?,
        deviceMachine: String,
        lensID: String,
        isoValues: [Float] = [33, 64, 100, 200, 400, 800]
    ) -> Bool {
        queue.sync {
            self.store[self.key(machine: deviceMachine, lensID: lensID)] = LensCalibration(
                noiseEntries: noiseEntries,
                lensShading: lensShading,
                greenBalance: greenBalance,
                isoValues: isoValues
            )
            return self.save()
        }
    }

    /// Remove calibration for a specific device+lens combo.
    func removeCalibration(deviceMachine: String, lensID: String) {
        queue.sync {
            store.removeValue(forKey: key(machine: deviceMachine, lensID: lensID))
            save()
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

    @discardableResult
    private func save() -> Bool {
        guard let data = try? JSONEncoder().encode(store) else { return false }
        do {
            try data.write(to: fileURL, options: .atomic)
            return true
        } catch {
            print("[NoiseCalibrationStore] Save failed: \(error)")
            return false
        }
    }
}
