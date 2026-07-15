import Foundation
import AVFoundation
import Combine
import Metal
import Photos
import simd
import UIKit

/// Central view model — CaptureController → RawFrameBuffer → MetalPipeline → preview + VideoWriter.
@MainActor
final class CameraViewModel: ObservableObject {
    // MARK: - Published State

    @Published var currentTexture: MTLTexture?
    @Published var isRecording = false
    @Published var controlsLocked = false
    @Published var thermalState: ProcessInfo.ThermalState = .nominal
    @Published var selectedCurve: LogCurveType = .sLog3Approx {
        didSet {
            guard !controlsLocked else { return }
            metalPipeline?.curveType = selectedCurve
        }
    }
    @Published var selectedFormat: RecordingFormat = .openGate {
        didSet {
            guard !controlsLocked, !isRecording else { return }
            activeEncodeWidth = selectedFormat.width
            activeEncodeHeight = selectedFormat.height
            // Suggest bitrate for format if user hasn't customized — keep current if already set
            refreshStatusLine()
        }
    }
    @Published var selectedFPS: CaptureFrameRate = .fps24 {
        didSet {
            guard !controlsLocked, !isRecording else { return }
            captureController.setCaptureFPS(selectedFPS.rawValue)
            activeFPS = selectedFPS.rawValue
            // 180° shutter rule → snap to nearest stop
            let target = Float(selectedFPS.rawValue * 2)
            shutterStopIndex = ExposureStops.nearestIndex(in: shutterStops, to: target)
            refreshStatusLine()
        }
    }
    @Published var selectedBitrate: BitratePreset = .mbps100 {
        didSet {
            guard !controlsLocked else { return }
            refreshStatusLine()
        }
    }
    @Published var audioSources: [AudioSourceOption] = [.none]
    @Published var selectedAudioSource: AudioSourceOption = .none {
        didSet {
            guard !controlsLocked, !isRecording else { return }
            // Skip no-op reassign (same port) — avoids hang loops
            guard oldValue.portUID != selectedAudioSource.portUID else { return }
            applyAudioSource()
        }
    }
    @Published var isSwitchingMic = false
    @Published var isSwitchingLens = false

    /// Dynamic back lenses for this iPhone.
    @Published var availableLenses: [LensOption] = []
    @Published var selectedLens: LensOption? {
        didSet {
            guard !controlsLocked, !isRecording else { return }
            guard let lens = selectedLens else { return }
            guard oldValue?.uniqueID != lens.uniqueID else { return }
            applyLens()
        }
    }

    @Published var showGrid = false
    @Published var showClipping = false
    @Published var showLevel = false {
        didSet {
            if showLevel {
                levelMonitor.start()
            } else {
                levelMonitor.stop()
            }
        }
    }

    let levelMonitor = LevelMonitor()

    @Published var frameCount: Int = 0
    @Published var recordingDuration: String = "00:00"
    @Published var statusText: String = "Starting…"
    @Published var errorMessage: String?
    @Published var droppedFrames: Int = 0
    @Published var cfaLabel: String = "—"

    /// Runtime device probe (set once at setup).
    @Published private(set) var capabilities: DeviceCapabilities?
    /// No Bayer RAW — hard gate, record disabled.
    @Published private(set) var isDeviceUnsupportedForLog = false
    /// Allow-list miss — soft warning, record still allowed.
    @Published private(set) var showUnverifiedDeviceWarning = false
    /// True when session started (or hard-failed setup) so splash can dismiss.
    @Published private(set) var isCameraReady = false

    @Published private(set) var isoValue: Float = 100
    @Published private(set) var shutterValue: Float = 48
    @Published private(set) var wbKelvin: Float = 5600

    // Focus properties
    @Published var isFocusLocked: Bool = false
    @Published var isAutoFocus: Bool = true {
        didSet {
            if isAutoFocus {
                isFocusLocked = false
                captureController.setContinuousAutoFocus()
            } else {
                captureController.setManualFocus(lensPosition: focusLensPosition)
            }
        }
    }
    @Published var focusLensPosition: Float = 0.5 {
        didSet {
            guard !isAutoFocus else { return }
            captureController.setManualFocus(lensPosition: focusLensPosition)
        }
    }

    /// Discrete stop lists (snap slider).
    @Published private(set) var isoStops: [Float] = ExposureStops.isoStops(in: 50...2000)
    @Published private(set) var shutterStops: [Float] = ExposureStops.shutterStops(in: 24...8000)
    @Published private(set) var wbStops: [Float] = ExposureStops.wbStops()

    @Published var isoStopIndex: Int = 0 {
        didSet {
            guard !isoStops.isEmpty else { return }
            let i = ExposureStops.clampIndex(isoStopIndex, count: isoStops.count)
            if i != isoStopIndex { isoStopIndex = i; return }
            let v = isoStops[i]
            guard v != isoValue else { return }
            isoValue = v
            guard !controlsLocked else { return }
            applyManualExposureAndWB()
        }
    }
    @Published var shutterStopIndex: Int = 0 {
        didSet {
            guard !shutterStops.isEmpty else { return }
            let i = ExposureStops.clampIndex(shutterStopIndex, count: shutterStops.count)
            if i != shutterStopIndex { shutterStopIndex = i; return }
            let v = shutterStops[i]
            guard v != shutterValue else { return }
            shutterValue = v
            guard !controlsLocked else { return }
            applyManualExposureAndWB()
        }
    }
    @Published var wbStopIndex: Int = 0 {
        didSet {
            guard !wbStops.isEmpty else { return }
            let i = ExposureStops.clampIndex(wbStopIndex, count: wbStops.count)
            if i != wbStopIndex { wbStopIndex = i; return }
            let v = wbStops[i]
            guard v != wbKelvin else { return }
            wbKelvin = v
            guard !controlsLocked else { return }
            applyManualExposureAndWB()
        }
    }

    @Published var wbMode: WBMode = .auto
    @Published var activePanel: ControlPanel? = nil

    var isoRange: ClosedRange<Float> = 50...2000
    var shutterRange: ClosedRange<Float> = 24...8000

    // MARK: - Pipeline

    let captureController = CaptureController()
    nonisolated let metalPipeline: MetalPipeline?
    private let videoWriter = VideoWriter()
    /// Capacity 5: absorb burst stalls while keeping latency low via dequeueLatest.
    nonisolated(unsafe) private let frameBuffer = RawFrameBuffer(capacity: 5)

    private var cancellables = Set<AnyCancellable>()
    private var recordingStartTime: Date?
    private var recordingTimer: Timer?
    nonisolated(unsafe) private var frameIndex: Int64 = 0
    private var outputURL: URL?
    private var hasTakenManualControl = false

    private let processQueue = DispatchQueue(label: "raw.process.queue", qos: .userInitiated)
    nonisolated private let processLock = NSLock()
    nonisolated(unsafe) private var isProcessing = false

    nonisolated(unsafe) private var activeEncodeWidth = 1920
    nonisolated(unsafe) private var activeEncodeHeight = 1440
    nonisolated(unsafe) private var activeFPS: Double = 24
    nonisolated(unsafe) private var isRecordingUnsafe = false
    /// Metal may not submit GPU work when app is backgrounded.
    nonisolated(unsafe) var isAppActive = true

    enum WBMode: String, CaseIterable, Identifiable {
        case auto = "Auto"
        case daylight = "Daylight"
        case tungsten = "Tungsten"
        case manual = "Manual"
        var id: String { rawValue }
    }

    enum ControlPanel: String, Identifiable {
        case iso, shutter, wb, focus, fps, format, log, bitrate, mic, lens
        var id: String { rawValue }
    }

    // MARK: - Init

    init() {
        metalPipeline = MetalPipeline()
        metalPipeline?.curveType = selectedCurve
        activeEncodeWidth = selectedFormat.width
        activeEncodeHeight = selectedFormat.height
        activeFPS = selectedFPS.rawValue
        selectedBitrate = selectedFormat.suggestedBitratePreset

        NotificationCenter.default.publisher(for: ProcessInfo.thermalStateDidChangeNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.thermalState = ProcessInfo.processInfo.thermalState
            }
            .store(in: &cancellables)

        // Refresh mic list when route changes (external mic plug/unplug)
        NotificationCenter.default.publisher(for: AVAudioSession.routeChangeNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.refreshAudioSources()
            }
            .store(in: &cancellables)

        // Also catch AVCapture audio device connect/disconnect
        NotificationCenter.default.publisher(for: AVCaptureDevice.wasConnectedNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] note in
                guard let device = note.object as? AVCaptureDevice, device.hasMediaType(.audio) else { return }
                self?.refreshAudioSources()
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: AVCaptureDevice.wasDisconnectedNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] note in
                guard let device = note.object as? AVCaptureDevice, device.hasMediaType(.audio) else { return }
                self?.refreshAudioSources()
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.handleAppInactive()
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.handleAppActive()
            }
            .store(in: &cancellables)

        captureController.onRawFrameData = { [weak self] frameData in
            self?.handleIncomingFrame(frameData)
        }
        captureController.onAudioSample = { [weak self] sample in
            guard let self else { return }
            // isRecordingUnsafe set from MainActor when record starts/stops
            if self.isRecordingUnsafe {
                _ = self.videoWriter.appendAudio(sampleBuffer: sample)
            }
        }
    }

    private func handleAppInactive() {
        isAppActive = false
        // Stop stills + drain queue so no Metal submits after background
        captureController.stopSession()
        frameBuffer.flush()
        if isRecording {
            stopRecording()
        }
        print("[CameraViewModel] App inactive — capture/GPU paused")
    }

    private func handleAppActive() {
        isAppActive = true
        // Restart session if we already configured once
        if captureController.activeDevice != nil {
            captureController.setCaptureFPS(selectedFPS.rawValue)
            captureController.startSession()
            if !controlsLocked {
                applyManualExposureAndWB()
            }
        }
        print("[CameraViewModel] App active — capture resumed")
    }

    // MARK: - Session

    func setupCamera() {
        errorMessage = nil
        statusText = "Configuring camera…"
        requestMicThenConfigure()
    }

    private func requestMicThenConfigure() {
        let micStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        switch micStatus {
        case .authorized:
            finishCameraSetup()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] _ in
                Task { @MainActor in
                    self?.finishCameraSetup()
                }
            }
        default:
            // Camera can still work without mic
            finishCameraSetup()
        }
    }

    private func finishCameraSetup() {
        // ── Capability probe (before session) ──
        let caps = DeviceCapabilities.probe()
        capabilities = caps
        showUnverifiedDeviceWarning = !caps.isVerifiedDevice
        isDeviceUnsupportedForLog = !caps.supportsBayerRAW

        // Apply tier defaults only when they match current defaults for A14/12 Pro
        // (same openGate + 24 + 100) — other tiers get safer recommendations.
        selectedFPS = caps.recommendedFPS
        selectedFormat = caps.recommendedFormat
        selectedBitrate = caps.recommendedBitrate
        activeEncodeWidth = selectedFormat.width
        activeEncodeHeight = selectedFormat.height
        activeFPS = selectedFPS.rawValue

        print(caps.diagnosticSummary)

        if isDeviceUnsupportedForLog {
            errorMessage = "This device does not support Bayer RAW stills. Log recording is not available."
            statusText = "Unsupported device"
            isCameraReady = false
            print("[CameraViewModel] HARD GATE: no Bayer RAW — record disabled")
            return
        }

        // CFA override table (nil on unknown → live OSType/DNG; 12 Pro → RGGB explicit)
        captureController.bayerPatternOverride = caps.bayerPatternOverride?.rawValue

        do {
            try AVAudioSession.sharedInstance().setCategory(
                .playAndRecord,
                mode: .videoRecording,
                options: [.defaultToSpeaker, .allowBluetooth, .allowBluetoothA2DP, .mixWithOthers]
            )
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            print("[CameraViewModel] Audio session: \(error)")
        }

        do {
            try captureController.configureSession()
            availableLenses = CaptureController.discoverBackLenses()
            if let currentID = captureController.currentLensUniqueID,
               let match = availableLenses.first(where: { $0.uniqueID == currentID }) {
                selectedLens = match
            } else {
                selectedLens = availableLenses.first
            }
            seedControlRanges()
            captureController.setCaptureFPS(selectedFPS.rawValue)
            refreshAudioSources()
            captureController.startSession()
            hasTakenManualControl = true
            applyManualExposureAndWB()
            refreshStatusLine()
            isCameraReady = true
            print("[CameraViewModel] Camera session started · \(caps.marketingName) · lenses=\(availableLenses.map(\.shortLabel))")
        } catch {
            errorMessage = error.localizedDescription
            statusText = "Camera failed"
            isCameraReady = false
            // If session fails due to no Bayer after all, treat as unsupported
            if (error as NSError).code == 4 {
                isDeviceUnsupportedForLog = true
            }
            print("[CameraViewModel] Failed to configure camera: \(error.localizedDescription)")
        }
    }

    func teardownCamera() {
        isRecordingUnsafe = false
        levelMonitor.stop()
        captureController.stopSession()
        frameBuffer.flush()
        try? AVAudioSession.sharedInstance().setActive(false)
    }

    func togglePanel(_ panel: ControlPanel) {
        if controlsLocked {
            switch panel {
            case .iso, .shutter, .wb, .focus, .fps, .format, .bitrate, .mic, .lens:
                return
            case .log:
                break
            }
        }
        if activePanel == panel {
            activePanel = nil
        } else {
            activePanel = panel
            // Fresh mic list when opening MIC (hot-plug refresh)
            if panel == .mic {
                refreshAudioSources()
            }
            if panel == .lens {
                availableLenses = CaptureController.discoverBackLenses()
            }
        }
    }

    func toggleGrid() {
        showGrid.toggle()
    }

    func toggleLevel() {
        showLevel.toggle()
    }

    func toggleClipping() {
        showClipping.toggle()
    }

    func refreshAudioSources() {
        let sources = captureController.availableAudioSources()
        audioSources = sources

        // Keep current port if still present (name-only update won't re-apply)
        if let match = sources.first(where: { $0.portUID == selectedAudioSource.portUID
                                              && $0.portUID != nil })
            ?? sources.first(where: { $0.id == selectedAudioSource.id }) {
            if match.id != selectedAudioSource.id || match.name != selectedAudioSource.name {
                selectedAudioSource = match
            }
            return
        }

        // Port gone (unplugged) → prefer iPhone built-in, else first available
        if let iphone = sources.first(where: { $0.name == "iPhone" }) {
            selectedAudioSource = iphone
        } else if let firstMic = sources.first {
            selectedAudioSource = firstMic
        }
    }

    private func applyAudioSource() {
        guard !isSwitchingMic else { return }
        isSwitchingMic = true
        let portUID = selectedAudioSource.portUID
        captureController.selectAudioSource(portUID: portUID) { [weak self] error in
            Task { @MainActor in
                self?.isSwitchingMic = false
                // Re-enumerate after switch — new ports often appear only after activate/route
                self?.refreshAudioSources()
                if let error {
                    self?.errorMessage = "Mic: \(error.localizedDescription)"
                } else {
                    self?.refreshStatusLine()
                }
            }
        }
    }

    private func applyLens() {
        guard !isSwitchingLens, let lens = selectedLens else { return }
        isSwitchingLens = true
        captureController.selectLens(uniqueID: lens.uniqueID) { [weak self] error in
            Task { @MainActor in
                guard let self else { return }
                self.isSwitchingLens = false
                if let error {
                    self.errorMessage = "Lens: \(error.localizedDescription)"
                    // Resync selection to actual device
                    if let id = self.captureController.currentLensUniqueID,
                       let match = self.availableLenses.first(where: { $0.uniqueID == id }) {
                        self.selectedLens = match
                    }
                } else {
                    // ISO/shutter ranges can change per lens
                    self.seedControlRanges()
                    if !self.controlsLocked {
                        self.applyManualExposureAndWB()
                    }
                    self.refreshStatusLine()
                }
            }
        }
    }

    private func refreshStatusLine() {
        if isRecording { return }
        let mic = selectedAudioSource.portUID == nil ? "mute" : "mic"
        if controlsLocked {
            statusText = "Locked · \(selectedFormat.shortLabel) · \(selectedFPS.label)fps · \(selectedBitrate.label)M"
        } else {
            statusText = "Edit · \(selectedFormat.shortLabel) · \(selectedFPS.label)fps · \(mic)"
        }
    }

    private func seedControlRanges() {
        guard let device = captureController.activeDevice ??
                AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        isoRange = device.activeFormat.minISO...device.activeFormat.maxISO
        isoStops = ExposureStops.isoStops(in: isoRange)
        isoStopIndex = ExposureStops.nearestIndex(in: isoStops, to: isoValue)

        let minDur = CMTimeGetSeconds(device.activeFormat.minExposureDuration)
        let maxDur = CMTimeGetSeconds(device.activeFormat.maxExposureDuration)
        if minDur > 0, maxDur > 0 {
            let maxShutter: Float = Float(1.0 / minDur)
            let minShutter: Float = Float(1.0 / maxDur)
            shutterRange = min(minShutter, maxShutter)...max(minShutter, maxShutter)
            shutterStops = ExposureStops.shutterStops(in: shutterRange)
            let target = Float(selectedFPS.rawValue * 2)
            shutterStopIndex = ExposureStops.nearestIndex(in: shutterStops, to: target)
        }

        wbStops = ExposureStops.wbStops()
        wbStopIndex = ExposureStops.nearestIndex(in: wbStops, to: wbKelvin)
    }

    func nudgeISO(_ delta: Int) {
        guard !controlsLocked else { return }
        isoStopIndex = ExposureStops.clampIndex(isoStopIndex + delta, count: isoStops.count)
    }

    func nudgeShutter(_ delta: Int) {
        guard !controlsLocked else { return }
        shutterStopIndex = ExposureStops.clampIndex(shutterStopIndex + delta, count: shutterStops.count)
    }

    func nudgeWB(_ delta: Int) {
        guard !controlsLocked else { return }
        wbStopIndex = ExposureStops.clampIndex(wbStopIndex + delta, count: wbStops.count)
    }

    // MARK: - Manual Controls (live when unlocked)

    /// Push ISO / shutter / WB to hardware. Call only when unlocked (or once on lock).
    func applyManualExposureAndWB() {
        guard let device = captureController.activeDevice ??
                AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }

        do {
            try device.lockForConfiguration()

            let clampedISO = max(device.activeFormat.minISO, min(device.activeFormat.maxISO, isoValue))
            var shutterDuration = CMTimeMakeWithSeconds(1.0 / Double(shutterValue), preferredTimescale: 1_000_000)
            let minD = device.activeFormat.minExposureDuration
            let maxD = device.activeFormat.maxExposureDuration
            if CMTimeCompare(shutterDuration, minD) < 0 { shutterDuration = minD }
            if CMTimeCompare(shutterDuration, maxD) > 0 { shutterDuration = maxD }
            device.setExposureModeCustom(duration: shutterDuration, iso: clampedISO)

            let temperatureAndTint = AVCaptureDevice.WhiteBalanceTemperatureAndTintValues(temperature: wbKelvin, tint: 0)
            let wbGains = device.deviceWhiteBalanceGains(for: temperatureAndTint)
            let clampedGains = clampWhiteBalanceGains(wbGains, for: device)
            device.setWhiteBalanceModeLocked(with: clampedGains)

            device.unlockForConfiguration()
            updateWBParams(from: device)
            hasTakenManualControl = true
        } catch {
            print("[CameraViewModel] applyManualExposureAndWB: \(error)")
        }
    }

    func lockControls() {
        // Freeze current values on device
        applyManualExposureAndWB()

        guard let device = captureController.activeDevice ??
                AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }

        do {
            try device.lockForConfiguration()
            // We don't lock focus here anymore, focus is managed independently
            device.unlockForConfiguration()
        } catch {}

        if let device = captureController.activeDevice {
            isoRange = device.activeFormat.minISO...device.activeFormat.maxISO
        }

        controlsLocked = true
        activePanel = nil
        refreshStatusLine()
        print("[CameraViewModel] Controls locked")
    }

    func unlockControls() {
        // Stay on custom exposure — do NOT restore continuous AE
        controlsLocked = false
        refreshStatusLine()
        print("[CameraViewModel] Controls unlocked (still manual)")
    }

    func setFocusPoint(_ point: CGPoint, lock: Bool = false) {
        isAutoFocus = true
        isFocusLocked = lock
        captureController.setFocusPointOfInterest(point, lock: lock)
    }

    private func clampWhiteBalanceGains(_ gains: AVCaptureDevice.WhiteBalanceGains, for device: AVCaptureDevice) -> AVCaptureDevice.WhiteBalanceGains {
        let maxGain = device.maxWhiteBalanceGain
        return AVCaptureDevice.WhiteBalanceGains(
            redGain: max(1.0, min(maxGain, gains.redGain)),
            greenGain: max(1.0, min(maxGain, gains.greenGain)),
            blueGain: max(1.0, min(maxGain, gains.blueGain))
        )
    }

    private func updateWBParams(from device: AVCaptureDevice) {
        switch wbMode {
        case .daylight:
            metalPipeline?.wbParams = .daylight
        case .tungsten:
            metalPipeline?.wbParams = .tungsten
        case .manual, .auto:
            let gains = device.deviceWhiteBalanceGains
            let g = max(gains.greenGain, 0.001)
            metalPipeline?.wbParams = WhiteBalanceParams(
                gains: SIMD3<Float>(
                    max(gains.redGain / g, 0.01),
                    1.0,
                    max(gains.blueGain / g, 0.01)
                ),
                colorMatrix: matrix_identity_float3x3
            )
        }
    }

    // MARK: - Recording

    func startRecording() {
        guard !isDeviceUnsupportedForLog else {
            errorMessage = "Recording disabled — device has no Bayer RAW."
            return
        }
        guard controlsLocked, !isRecording else { return }

        metalPipeline?.curveType = selectedCurve
        activeEncodeWidth = selectedFormat.width
        activeEncodeHeight = selectedFormat.height
        activeFPS = selectedFPS.rawValue

        let fileName = "OwLens_\(selectedFormat.shortLabel)_\(selectedFPS.label)fps_HEVC_\(Int(Date().timeIntervalSince1970)).mov"
        outputURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)

        let includeAudio = selectedAudioSource.portUID != nil

        do {
            try videoWriter.start(
                outputURL: outputURL!,
                width: selectedFormat.width,
                height: selectedFormat.height,
                bitrate: selectedBitrate.bitsPerSecond,
                targetFPS: selectedFPS.rawValue,
                includeAudio: includeAudio
            )
            isRecording = true
            isRecordingUnsafe = true
            frameIndex = 0
            frameCount = 0
            recordingStartTime = Date()
            recordingDuration = "00:00"
            activePanel = nil

            recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    self?.updateRecordingDuration()
                }
            }
            statusText = "REC · \(selectedFormat.shortLabel) · \(selectedFPS.label)fps · HEVC"
            let capsLine = capabilities?.diagnosticSummary ?? ""
            print("[CameraViewModel] Recording start \(selectedFormat.width)x\(selectedFormat.height) CFR \(selectedFPS.label) HEVC\n\(capsLine)")
        } catch {
            errorMessage = "Record failed: \(error.localizedDescription)"
            print("[CameraViewModel] Failed to start recording: \(error)")
        }
    }

    func stopRecording() {
        guard isRecording else { return }

        isRecording = false
        isRecordingUnsafe = false
        recordingTimer?.invalidate()
        recordingTimer = nil
        statusText = "Saving…"

        videoWriter.finish { [weak self] url in
            guard let url else {
                Task { @MainActor in
                    self?.statusText = "Save failed"
                    self?.errorMessage = "No output file"
                }
                return
            }
            PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
                guard status == .authorized || status == .limited else {
                    Task { @MainActor in
                        self?.errorMessage = "Photo library access denied"
                        self?.refreshStatusLine()
                    }
                    return
                }
                PHPhotoLibrary.shared().performChanges {
                    PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
                } completionHandler: { success, error in
                    Task { @MainActor in
                        if success {
                            self?.statusText = "Saved to Photos"
                        } else {
                            self?.errorMessage = error?.localizedDescription ?? "Save failed"
                            self?.statusText = "Save failed"
                        }
                        self?.refreshStatusLine()
                    }
                    try? FileManager.default.removeItem(at: url)
                }
            }
        }

        let realNote = "frames=\(frameCount) drops=\(droppedFrames) fps=\(selectedFPS.label) fmt=\(selectedFormat.shortLabel)"
        print("[CameraViewModel] Recording stopped \(realNote)")
        if let caps = capabilities {
            print("[CameraViewModel] Tester diagnostics:\n\(caps.diagnosticSummary)\n\(realNote)")
        }
    }

    func exportLUT() {
        let lut = LUTGenerator.generate(for: selectedCurve, size: 33)
        let fileName = "RawLogCam_\(selectedCurve.displayName.replacingOccurrences(of: " ", with: "_")).cube"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)

        do {
            try lut.write(toFile: url.path, atomically: true, encoding: String.Encoding.utf8)
            Task { @MainActor in
                guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                      let rootVC = scene.windows.first?.rootViewController else { return }
                let activityVC = UIActivityViewController(activityItems: [url], applicationActivities: nil)
                if let pop = activityVC.popoverPresentationController {
                    pop.sourceView = rootVC.view
                    pop.sourceRect = CGRect(x: rootVC.view.bounds.midX, y: rootVC.view.bounds.midY, width: 0, height: 0)
                }
                rootVC.present(activityVC, animated: true)
            }
        } catch {
            errorMessage = "LUT export failed"
        }
    }

    private func updateRecordingDuration() {
        guard let start = recordingStartTime else { return }
        let elapsed = Date().timeIntervalSince(start)
        let minutes = Int(elapsed) / 60
        let seconds = Int(elapsed) % 60
        recordingDuration = String(format: "%02d:%02d", minutes, seconds)
    }

    // MARK: - Frame Pipeline

    nonisolated private func handleIncomingFrame(_ frameData: RawFrameData) {
        // Never enqueue GPU work while backgrounded (IOGPUMetalError 00000006)
        guard isAppActive else { return }
        frameBuffer.enqueue(frameData)
        scheduleProcess()
    }

    nonisolated private func scheduleProcess() {
        processLock.lock()
        if isProcessing {
            processLock.unlock()
            return
        }
        isProcessing = true
        processLock.unlock()

        processQueue.async { [weak self] in
            self?.drainBuffer()
        }
    }

    nonisolated private func drainBuffer() {
        // Only the newest frame — never process a backlog (that made preview laggy/choppy)
        if let frame = frameBuffer.dequeueLatest() {
            processFrame(frame)
        }
        processLock.lock()
        isProcessing = false
        let remaining = frameBuffer.currentCount
        processLock.unlock()
        if remaining > 0 {
            scheduleProcess()
        }
    }

    nonisolated private func processFrame(_ frameData: RawFrameData) {
        guard isAppActive else { return }
        guard let pipeline = metalPipeline else { return }

        pipeline.bayerPattern = frameData.cfaPattern
        pipeline.blackLevel = frameData.blackLevel
        pipeline.whiteLevel = frameData.whiteLevel

        if let gains = frameData.whiteBalanceGains {
            let g = max(gains.greenGain, 0.001)
            pipeline.wbParams = WhiteBalanceParams(
                gains: SIMD3<Float>(
                    max(gains.redGain / g, 0.01),
                    1.0,
                    max(gains.blueGain / g, 0.01)
                ),
                colorMatrix: matrix_identity_float3x3
            )
        }

        let w = activeEncodeWidth
        let h = activeEncodeHeight
        // CFA-safe 2× bin (when huge) + demosaic + WB + log + crop/scale to encode size
        guard let framed = pipeline.process(frameData.pixelBuffer, encodeWidth: w, encodeHeight: h) else { return }

        let cfaName: String
        switch frameData.cfaPattern {
        case 0: cfaName = "RGGB"
        case 1: cfaName = "GRBG"
        case 2: cfaName = "GBRG"
        case 3: cfaName = "BGGR"
        default: cfaName = "?\(frameData.cfaPattern)"
        }
        let drops = frameBuffer.droppedCount

        // BGRA conversion + video encoding stays on processQueue (off MainActor)
        if isRecordingUnsafe {
            if let pb = pipeline.textureToPixelBufferBGRA(framed) {
                if videoWriter.appendFrame(pixelBuffer: pb) {
                    // frameIndex is nonisolated(unsafe), safe from processQueue
                    frameIndex += 1
                }
            }
        }

        // Only preview texture + UI counters go to MainActor
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.currentTexture = framed
            self.currentTexture = framed
            self.cfaLabel = cfaName
            self.droppedFrames = drops
            if self.isRecording {
                self.frameCount = Int(self.frameIndex)
            }
        }
    }
}
