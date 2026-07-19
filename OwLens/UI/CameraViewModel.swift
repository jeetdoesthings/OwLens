import Foundation
import AVFoundation
import Combine
import Metal
import Photos
import QuartzCore
import simd
import UIKit
import UniformTypeIdentifiers

/// Central view model — CaptureController → RawFrameBuffer → MetalPipeline → preview + VideoWriter.
@MainActor
final class CameraViewModel: NSObject, ObservableObject, UIDocumentPickerDelegate {
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
            // 180° shutter rule is now natively maintained by the angle system.
            // We just need to refresh limits and push the new duration to hardware.
            seedControlRanges()
            if isCameraReady { applyManualExposureAndWB() }
            refreshStatusLine()
        }
    }
    @Published var selectedBitrate: BitratePreset = .mbps100 {
        didSet {
            guard !controlsLocked else { return }
            refreshStatusLine()
        }
    }
    @Published private(set) var selectedSaveDestination: VideoSaveDestination = .photos
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
    @Published var showFocusPeaking = false
    @Published var showScopes = false
    @Published var scopeData: ScopeData = .empty
    @Published var previewDisplayMode: PreviewDisplayMode = .log
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
    @Published private(set) var shutterValue: Float = 180
    @Published private(set) var wbKelvin: Float = 5600
    @Published var isAutoExposureEnabled: Bool = false {
        didSet {
            guard oldValue != isAutoExposureEnabled else { return }
            if isCameraReady { applyManualExposureAndWB() }
        }
    }
    @Published var isAutoWhiteBalanceEnabled: Bool = false {
        didSet {
            guard oldValue != isAutoWhiteBalanceEnabled else { return }
            if isCameraReady { applyManualExposureAndWB() }
        }
    }
    @Published private(set) var isAutoExposureAdjusting: Bool = false
    @Published private(set) var isAutoWhiteBalanceAdjusting: Bool = false

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
            guard !controlsLocked, !isAutoFocus else { return }
            captureController.setManualFocus(lensPosition: focusLensPosition)
        }
    }
    @Published var focusPointLocation: CGPoint? = nil

    /// Discrete stop lists (snap slider).
    @Published private(set) var isoStops: [Float] = ExposureStops.isoStops(in: 50...2000)
    @Published private(set) var wbStops: [Float] = ExposureStops.wbStops()

    @Published var isoStopIndex: Int = 0 {
        didSet {
            guard !isoStops.isEmpty else { return }
            let i = ExposureStops.clampIndex(isoStopIndex, count: isoStops.count)
            if i != isoStopIndex { isoStopIndex = i; return }
            let v = isoStops[i]
            guard v != isoValue else { return }
            isoValue = v
            guard !controlsLocked, !isAutoExposureEnabled else { return }
            applyManualExposureAndWB()
        }
    }
    func setShutterAngleWithSnapping(_ rawValue: Float) {
        guard !isAutoExposureEnabled else { return }
        let snapTargets = ExposureStops.shutterAngles
        var finalValue = rawValue
        
        for target in snapTargets {
            // Magnetic snap radius of 15 degrees
            if abs(rawValue - target) < 15.0 {
                finalValue = target
                break
            }
        }
        
        finalValue = max(shutterRange.lowerBound, min(shutterRange.upperBound, finalValue))
        
        if finalValue != shutterValue {
            shutterValue = finalValue
            if isCameraReady { applyManualExposureAndWB() }
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
            guard !controlsLocked, !isAutoWhiteBalanceEnabled else { return }
            applyManualExposureAndWB()
        }
    }

    @Published var activePanel: ControlPanel? = nil

    var isoRange: ClosedRange<Float> = 50...2000
    var shutterRange: ClosedRange<Float> = 11.25...360.0

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
    private var filesFolderBookmark: Data?
    private let filesFolderBookmarkKey = "OwLens.FilesFolderBookmark"

    private let processQueue = DispatchQueue(label: "raw.process.queue", qos: .userInitiated)
    nonisolated private let processLock = NSLock()
    nonisolated(unsafe) private var isProcessing = false

    nonisolated(unsafe) private var activeEncodeWidth = 1920
    nonisolated(unsafe) private var activeEncodeHeight = 1440
    nonisolated(unsafe) private var activeFPS: Double = 24
    nonisolated(unsafe) private var isRecordingUnsafe = false
    nonisolated(unsafe) private var showScopesUnsafe = false
    nonisolated(unsafe) private var lastScopeUpdateTime: CFTimeInterval = 0
    /// Metal may not submit GPU work when app is backgrounded.
    nonisolated(unsafe) var isAppActive = true

    enum ControlPanel: String, Identifiable {
        case exposure, iso, shutter, wb, focus, fps, format, log, bitrate, mic, lens, save
        var id: String { rawValue }
    }

    // MARK: - Init

    override init() {
        metalPipeline = MetalPipeline()
        super.init()
        loadFilesFolderBookmark()
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
                options: [.defaultToSpeaker, .allowBluetoothHFP, .allowBluetoothA2DP, .mixWithOthers]
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
            case .exposure, .iso, .shutter, .wb, .focus, .fps, .format, .bitrate, .mic, .lens:
                return
            case .log, .save:
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

    func chooseSaveDestination(_ destination: VideoSaveDestination) {
        guard !isRecording else { return }
        activePanel = nil
        switch destination {
        case .photos:
            selectedSaveDestination = .photos
        case .files:
            presentFilesFolderPicker()
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
    
    func toggleFocusPeaking() {
        showFocusPeaking.toggle()
    }

    func toggleScopes() {
        showScopes.toggle()
        showScopesUnsafe = showScopes
        if !showScopes {
            scopeData = .empty
        }
    }

    func togglePreviewDisplayMode() {
        previewDisplayMode = previewDisplayMode == .log ? .normalVideo : .log
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
        let mode = controlsLocked ? "Locked" : "Edit"
        statusText = "\(mode) · \(selectedFormat.shortLabel) · \(selectedFPS.label)fps · \(statusMicName)"
    }

    private var statusMicName: String {
        guard selectedAudioSource.portUID != nil else { return "mute" }
        let name = selectedAudioSource.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return "mic" }
        if name.count <= 12 { return name }
        return String(name.prefix(11)) + "…"
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
            let maxDur = Float(device.activeFormat.maxExposureDuration.seconds)
            let minDur = Float(device.activeFormat.minExposureDuration.seconds)
            
            let maxAngle = min(360.0, maxDur * 360.0 * Float(activeFPS))
            let minAngle = minDur * 360.0 * Float(activeFPS)
            
            let target: Float = 180.0 // Default 180° shutter rule
            
            // Just clamp the current shutter value to the new range, or snap to 180 if out of bounds
            if shutterValue < minAngle || shutterValue > maxAngle {
                shutterValue = max(minAngle, min(maxAngle, target))
            }
        }

        wbStops = ExposureStops.wbStops()
        wbStopIndex = ExposureStops.nearestIndex(in: wbStops, to: wbKelvin)
    }

    func nudgeISO(_ delta: Int) {
        guard !controlsLocked, !isAutoExposureEnabled else { return }
        isoStopIndex = ExposureStops.clampIndex(isoStopIndex + delta, count: isoStops.count)
    }

    func nudgeWB(_ delta: Int) {
        guard !controlsLocked, !isAutoWhiteBalanceEnabled else { return }
        wbStopIndex = ExposureStops.clampIndex(wbStopIndex + delta, count: wbStops.count)
    }

    // MARK: - Manual Controls (live when unlocked)

    /// Push ISO / shutter / WB to hardware. Call only when unlocked (or once on lock).
    func applyManualExposureAndWB() {
        guard let device = captureController.activeDevice ??
                AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }

        do {
            try device.lockForConfiguration()

            if isAutoExposureEnabled {
                if device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposureMode = .continuousAutoExposure
                }
            } else {
                let clampedISO = max(device.activeFormat.minISO, min(device.activeFormat.maxISO, isoValue))
                var shutterDuration = CMTimeMakeWithSeconds((Double(shutterValue) / 360.0) / activeFPS, preferredTimescale: 1_000_000)
                let minD = device.activeFormat.minExposureDuration
                let maxD = device.activeFormat.maxExposureDuration
                if CMTimeCompare(shutterDuration, minD) < 0 { shutterDuration = minD }
                if CMTimeCompare(shutterDuration, maxD) > 0 { shutterDuration = maxD }
                device.setExposureModeCustom(duration: shutterDuration, iso: clampedISO)
            }

            if isAutoWhiteBalanceEnabled {
                if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                    device.whiteBalanceMode = .continuousAutoWhiteBalance
                }
            } else {
                let temperatureAndTint = AVCaptureDevice.WhiteBalanceTemperatureAndTintValues(temperature: wbKelvin, tint: 0)
                let wbGains = device.deviceWhiteBalanceGains(for: temperatureAndTint)
                let clampedGains = clampWhiteBalanceGains(wbGains, for: device)
                device.setWhiteBalanceModeLocked(with: clampedGains)
            }

            device.unlockForConfiguration()
            updateWBParams(from: device)
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
        controlsLocked = false
        refreshStatusLine()
        print("[CameraViewModel] Controls unlocked")
    }

    func setFocusPoint(_ point: CGPoint, lock: Bool = false) {
        isAutoFocus = true
        isFocusLocked = lock
        captureController.setFocusPointOfInterest(point, lock: lock)
    }
    
    func triggerTapToFocus(at screenPoint: CGPoint, normalized: CGPoint) {
        setFocusPoint(normalized, lock: false)
        focusPointLocation = screenPoint
        
        // Hide indicator after 1.5 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            if self.focusPointLocation == screenPoint {
                self.focusPointLocation = nil
            }
        }
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
        metalPipeline?.isAutoWBEnabled = isAutoWhiteBalanceEnabled
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

    // MARK: - Recording

    func startRecording() {
        guard !isDeviceUnsupportedForLog else {
            errorMessage = "Recording disabled — device has no Bayer RAW."
            return
        }
        guard controlsLocked, !isRecording else { return }

        lockAutoModesForRecording()
        metalPipeline?.curveType = selectedCurve
        activeEncodeWidth = selectedFormat.width
        activeEncodeHeight = selectedFormat.height
        activeFPS = selectedFPS.rawValue

        let fileName = "OwLens_\(selectedFormat.shortLabel)_\(selectedFPS.label)fps_HEVC_\(Int(Date().timeIntervalSince1970)).mov"
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)

        let includeAudio = selectedAudioSource.portUID != nil

        do {
            try videoWriter.start(
                outputURL: outputURL,
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

    private func lockAutoModesForRecording() {
        guard isAutoExposureEnabled || isAutoWhiteBalanceEnabled else { return }
        guard let device = captureController.activeDevice else { return }

        isoValue = device.iso
        let angle = Float(device.exposureDuration.seconds * activeFPS * 360.0)
        if angle.isFinite && angle > 0 {
            shutterValue = max(shutterRange.lowerBound, min(shutterRange.upperBound, angle))
        }
        let gains = device.deviceWhiteBalanceGains
        let temperatureAndTint = device.temperatureAndTintValues(for: gains)
        wbKelvin = max(2000, min(10000, temperatureAndTint.temperature))

        isAutoExposureEnabled = false
        isAutoWhiteBalanceEnabled = false
        isAutoExposureAdjusting = false
        isAutoWhiteBalanceAdjusting = false
        applyManualExposureAndWB()
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
            Task { @MainActor [weak self] in
                self?.saveFinishedRecording(at: url)
            }
        }

        let realNote = "frames=\(frameCount) drops=\(droppedFrames) fps=\(selectedFPS.label) fmt=\(selectedFormat.shortLabel)"
        print("[CameraViewModel] Recording stopped \(realNote)")
        if let caps = capabilities {
            print("[CameraViewModel] Tester diagnostics:\n\(caps.diagnosticSummary)\n\(realNote)")
        }
    }

    private func saveFinishedRecording(at url: URL) {
        switch selectedSaveDestination {
        case .photos:
            PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
                guard status == .authorized || status == .limited else {
                    Task { @MainActor in
                        self.errorMessage = "Photo library access denied"
                        self.refreshStatusLine()
                    }
                    return
                }
                PHPhotoLibrary.shared().performChanges {
                    PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
                } completionHandler: { success, error in
                    Task { @MainActor in
                        if success {
                            self.statusText = "Saved to Photos"
                        } else {
                            self.errorMessage = error?.localizedDescription ?? "Save failed"
                            self.statusText = "Save failed"
                        }
                        self.refreshStatusLine()
                    }
                    try? FileManager.default.removeItem(at: url)
                }
            }
        case .files:
            saveRecordingToChosenFilesFolder(url)
        }
    }

    private func presentFilesFolderPicker() {
        guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let rootVC = scene.windows.first(where: { $0.isKeyWindow })?.rootViewController else {
            errorMessage = "Files picker unavailable"
            refreshStatusLine()
            return
        }

        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.folder], asCopy: false)
        picker.delegate = self
        picker.allowsMultipleSelection = false
        if let pop = picker.popoverPresentationController {
            pop.sourceView = rootVC.view
            pop.sourceRect = CGRect(x: rootVC.view.bounds.midX, y: rootVC.view.bounds.midY, width: 0, height: 0)
            pop.permittedArrowDirections = []
        }
        rootVC.present(picker, animated: true)
    }

    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        guard let folderURL = urls.first else {
            selectedSaveDestination = .photos
            refreshStatusLine()
            return
        }
        do {
            let shouldStop = folderURL.startAccessingSecurityScopedResource()
            defer {
                if shouldStop {
                    folderURL.stopAccessingSecurityScopedResource()
                }
            }
            filesFolderBookmark = try folderURL.bookmarkData(
                options: [.minimalBookmark],
                includingResourceValuesForKeys: nil,
                relativeTo: nil
            )
            UserDefaults.standard.set(filesFolderBookmark, forKey: filesFolderBookmarkKey)
            selectedSaveDestination = .files
            statusText = "Files selected"
        } catch {
            selectedSaveDestination = .photos
            errorMessage = "Files folder failed: \(error.localizedDescription)"
        }
        refreshStatusLine()
    }

    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
        selectedSaveDestination = .photos
        refreshStatusLine()
    }

    private func loadFilesFolderBookmark() {
        guard let bookmark = UserDefaults.standard.data(forKey: filesFolderBookmarkKey) else { return }
        filesFolderBookmark = bookmark
        if resolveFilesFolderURL() != nil {
            selectedSaveDestination = .files
        }
    }

    private func resolveFilesFolderURL() -> URL? {
        guard let filesFolderBookmark else { return nil }
        do {
            var isStale = false
            let url = try URL(
                resolvingBookmarkData: filesFolderBookmark,
                options: [],
                relativeTo: nil,
                bookmarkDataIsStale: &isStale
            )
            if isStale {
                UserDefaults.standard.removeObject(forKey: filesFolderBookmarkKey)
                self.filesFolderBookmark = nil
                selectedSaveDestination = .photos
                return nil
            }
            return url
        } catch {
            UserDefaults.standard.removeObject(forKey: filesFolderBookmarkKey)
            self.filesFolderBookmark = nil
            selectedSaveDestination = .photos
            return nil
        }
    }

    private func saveRecordingToChosenFilesFolder(_ url: URL) {
        guard let folderURL = resolveFilesFolderURL() else {
            errorMessage = "Choose a Files folder before recording"
            statusText = "Save failed"
            return
        }

        let shouldStop = folderURL.startAccessingSecurityScopedResource()
        defer {
            if shouldStop {
                folderURL.stopAccessingSecurityScopedResource()
            }
        }

        do {
            let destination = uniqueDestinationURL(in: folderURL, preferredName: url.lastPathComponent)
            try FileManager.default.copyItem(at: url, to: destination)
            try? FileManager.default.removeItem(at: url)
            statusText = "Saved to Files"
            refreshStatusLine()
            print("[CameraViewModel] Saved recording to Files: \(destination.path)")
        } catch {
            errorMessage = "Files save failed: \(error.localizedDescription)"
            statusText = "Save failed"
        }
    }

    private func uniqueDestinationURL(in folderURL: URL, preferredName: String) -> URL {
        let baseURL = folderURL.appendingPathComponent(preferredName)
        guard FileManager.default.fileExists(atPath: baseURL.path) else { return baseURL }

        let ext = baseURL.pathExtension
        let stem = baseURL.deletingPathExtension().lastPathComponent
        for index in 1..<1000 {
            let candidateName = ext.isEmpty ? "\(stem)-\(index)" : "\(stem)-\(index).\(ext)"
            let candidate = folderURL.appendingPathComponent(candidateName)
            if !FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }
        return folderURL.appendingPathComponent(UUID().uuidString + (ext.isEmpty ? "" : ".\(ext)"))
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

    struct SendablePixelBuffer: @unchecked Sendable {
        let buffer: CVPixelBuffer
    }

    nonisolated private func processFrame(_ frameData: RawFrameData) {
        guard isAppActive else { return }
        guard let pipeline = metalPipeline else { return }

        pipeline.bayerPattern = frameData.cfaPattern
        pipeline.blackLevel = frameData.blackLevel
        pipeline.whiteLevel = frameData.whiteLevel
        pipeline.lscCoefficients = frameData.lscCoefficients
        pipeline.iso = frameData.iso

        if pipeline.isAutoWBEnabled, let gains = frameData.whiteBalanceGains {
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
        let newScopeData: ScopeData?
        let scopeNow = CACurrentMediaTime()
        if showScopesUnsafe && scopeNow - lastScopeUpdateTime >= 0.1 {
            lastScopeUpdateTime = scopeNow
            newScopeData = pipeline.makeScopeData(from: framed)
        } else {
            newScopeData = nil
        }

        // BGRA conversion + video encoding is fully asynchronous (zero CPU blocking)
        if isRecordingUnsafe {
            pipeline.textureToPixelBufferBGRA(framed) { [weak self] pb in
                guard let self = self, let pb = pb else { return }
                let sendablePB = SendablePixelBuffer(buffer: pb)
                self.processQueue.async {
                    Task { @MainActor [weak self] in
                        guard let self = self else { return }
                        if self.videoWriter.appendFrame(pixelBuffer: sendablePB.buffer) {
                            self.frameIndex += 1
                        }
                    }
                }
            }
        }

        // Only preview texture + UI counters go to MainActor
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.syncLiveAutoValues(from: frameData)
            self.currentTexture = framed
            self.cfaLabel = cfaName
            self.droppedFrames = drops
            if let newScopeData {
                self.scopeData = newScopeData
            }
            if self.isRecording {
                self.frameCount = Int(self.frameIndex)
            }
        }
    }

    private func syncLiveAutoValues(from frameData: RawFrameData) {
        guard let device = captureController.activeDevice else { return }
        if isAutoExposureEnabled {
            if frameData.iso > 0 {
                isoValue = frameData.iso
            }
            if frameData.exposureDurationSeconds > 0 {
                let angle = Float(frameData.exposureDurationSeconds * activeFPS * 360.0)
                shutterValue = max(shutterRange.lowerBound, min(shutterRange.upperBound, angle))
            }
            isAutoExposureAdjusting = device.isAdjustingExposure
        } else {
            isAutoExposureAdjusting = false
        }

        if isAutoWhiteBalanceEnabled {
            if let gains = frameData.whiteBalanceGains {
                let temperatureAndTint = device.temperatureAndTintValues(for: gains)
                wbKelvin = max(2000, min(10000, temperatureAndTint.temperature))
            }
            isAutoWhiteBalanceAdjusting = device.isAdjustingWhiteBalance
        } else {
            isAutoWhiteBalanceAdjusting = false
        }
    }
}
