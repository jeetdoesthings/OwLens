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
    /// Incremented with every new texture frame. UInt64 is Equatable so SwiftUI can
    /// reliably detect the change and call updateUIView on CameraPreviewView, even
    /// though MTLTexture itself is not Equatable.
    @Published var textureChangeCount: UInt64 = 0
    @Published var isRecording = false
    @Published var controlsLocked = false
    @Published var thermalState: ProcessInfo.ThermalState = .nominal
    @Published var selectedCurve: LogCurveType = .sLog3Approx {
        didSet {
            guard !isRecording else { return }
            metalPipeline?.curveType = selectedCurve
            if let dev = captureController.activeDevice {
                updateWBParams(from: dev)
            }
            refreshStatusLine()
            print("[CameraViewModel] Log curve switched to: \(selectedCurve.displayName)")
        }
    }
    @Published var selectedFormat: RecordingFormat = .openGate {
        didSet {
            guard !isRecording else { return }
            activeEncodeWidth = selectedFormat.width
            activeEncodeHeight = selectedFormat.height
            if selectedBitrate.rawValue > selectedFormat.maxBitratePreset.rawValue {
                selectedBitrate = selectedFormat.suggestedBitratePreset
            }
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

    /// Dynamic back lenses for this device.
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
    @Published var showScopes = true
    @Published var scopeData: ScopeData = .empty
    @Published var previewDisplayMode: PreviewDisplayMode = .log
    /// Optional viewfinder display transform; defaults to false so log preview remains untouched.
    @Published var showDisplayLUT: Bool = false
    @Published var showLevel = false {
        didSet {
            if showLevel {
                levelMonitor.start()
            } else {
                levelMonitor.stop()
            }
        }
    }

    @Published var wbTint: Float = 0.0
    @Published var meteringMode: MeteringMode = .matrix {
        didSet {
            captureController.setMeteringMode(meteringMode)
        }
    }

    let levelMonitor = LevelMonitor()

    /// Frame count for recording telemetry (not published to avoid thrashing SwiftUI on every frame).
    private(set) var frameCount: Int = 0
    @Published var recordingDuration: String = "00:00"
    @Published var statusText: String = "Starting…"
    private var errorDismissWork: DispatchWorkItem?
    @Published var errorMessage: String? {
        didSet {
            errorDismissWork?.cancel()
            errorDismissWork = nil
            if errorMessage != nil {
                let work = DispatchWorkItem { [weak self] in
                    self?.errorMessage = nil
                }
                errorDismissWork = work
                DispatchQueue.main.asyncAfter(deadline: .now() + 4.0, execute: work)
            }
        }
    }
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

    /// User-adjustable denoise strength (0.0–1.0). Written to MetalPipeline on change.
    @Published var denoiseStrength: Float = 1.0 {
        didSet {
            metalPipeline?.denoiseStrength = denoiseStrength
        }
    }

    // Focus properties
    /// Tracks work item to cancel if focus is re-locked before 2s auto-dismiss fires.
    private var focusLockDismissWork: DispatchWorkItem?
    @Published var isFocusLocked: Bool = false {
        didSet {
            focusLockDismissWork?.cancel()
            if isFocusLocked {
                let work = DispatchWorkItem { [weak self] in
                    self?.isFocusLocked = false
                }
                focusLockDismissWork = work
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0, execute: work)
            }
        }
    }
    @Published var isAutoFocus: Bool = true {
        didSet {
            guard !controlsLocked else { return }
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

    // Exposure / WB control values

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
            scheduleExposureUpdate()
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
            if isCameraReady { scheduleExposureUpdate() }
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
            scheduleExposureUpdate()
        }
    }

    @Published var activePanel: ControlPanel? = nil

    var isoRange: ClosedRange<Float> = 50...2000
    var shutterRange: ClosedRange<Float> = 11.25...360.0

    // MARK: - Pipeline

    let captureController = CaptureController()
    nonisolated let metalPipeline: MetalPipeline?
    nonisolated private let videoWriter = VideoWriter()
    /// Capacity 8: absorb burst stalls while keeping latency low.
    nonisolated(unsafe) private let frameBuffer = RawFrameBuffer(capacity: 8)

    private var cancellables = Set<AnyCancellable>()
    /// Debounce rapid slider changes to avoid blocking the main thread on lockForConfiguration().
    private var exposureDebounceWork: DispatchWorkItem?
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
    nonisolated(unsafe) private var showScopesUnsafe = true
    nonisolated(unsafe) private var lastScopeUpdateTime: CFTimeInterval = 0
    nonisolated(unsafe) var isAppActive = true
    /// Measured LSC override from device calibration (set once at setup).
    nonisolated(unsafe) private var lscOverride: LSCCoefficients?

    /// Measured noise coefficients from device calibration (set once at setup).
    nonisolated(unsafe) private var noiseCoeffs: (shot: Float, read: Float) = (0.012, 0.0004)
    /// Noise profile for per-ISO coefficient lookup (nil on unknown device).
    nonisolated(unsafe) private var noiseProfileForISO: NoiseProfile?
    /// Last ISO seen by processFrame; used to detect scene-cut ISO jumps.
    nonisolated(unsafe) private var lastProcessedISO: Float = 0
    /// Latest calibrated color matrices extracted from DNG metadata.
    nonisolated(unsafe) private var latestColorMatrix: simd_float3x3?
    nonisolated(unsafe) private var latestSGamutMatrix: simd_float3x3?



    enum ControlPanel: String, Identifiable {
        case exposure, iso, shutter, wb, focus, fps, format, log, bitrate, denoise, mic, lens, save
        var id: String { rawValue }
    }

    // MARK: - Init

    override init() {
        metalPipeline = MetalPipeline()
        super.init()
#if DEBUG
        if let pipeline = metalPipeline {
            Task { @MainActor in
                _ = pipeline.runSyntheticHotPixelTest()
                _ = MetalPipeline.runAppleLog2AccuracyTest()
                _ = MetalPipeline.runColorMatrixValidationTest()
                _ = MetalPipeline.runHighlightShoulderTest()
                _ = pipeline.runPipelineThroughputBenchmark()
                _ = CameraViewModel.runFileNameGenerationTest()
            }
        }
#endif
        loadFilesFolderBookmark()
        metalPipeline?.curveType = selectedCurve
        metalPipeline?.denoiseStrength = denoiseStrength
        metalPipeline?.thermalState = ProcessInfo.processInfo.thermalState
        activeEncodeWidth = selectedFormat.width
        activeEncodeHeight = selectedFormat.height
        activeFPS = selectedFPS.rawValue
        selectedBitrate = selectedFormat.suggestedBitratePreset

        NotificationCenter.default.publisher(for: ProcessInfo.thermalStateDidChangeNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                guard let self else { return }
                let state = ProcessInfo.processInfo.thermalState
                self.thermalState = state
                self.metalPipeline?.thermalState = state
                if state.rawValue >= ProcessInfo.ThermalState.serious.rawValue {
                    print("[CameraViewModel] Thermal state elevated (\(state.rawValue)) - stepping down processing load")
                }
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

        NotificationCenter.default.publisher(for: NSNotification.Name.AVCaptureSessionWasInterrupted)
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
        metalPipeline?.clearTemporalHistory()
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
            metalPipeline?.clearTemporalHistory()
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

// LSC and noise profile overrides from device calibration
        lscOverride = caps.lscOverride
        noiseProfileForISO = caps.noiseProfile
        if let prof = caps.noiseProfile {
            let c = prof.coefficients(at: 33.0)  // base ISO; updated per-frame in processFrame
            noiseCoeffs = c
        }

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
            try AVAudioSession.sharedInstance().setPreferredSampleRate(48_000)
            try AVAudioSession.sharedInstance().setPreferredIOBufferDuration(0.02)
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
        if controlsLocked || isRecording {
            activePanel = nil
            return
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

    func toggleDisplayLUT() {
        showDisplayLUT.toggle()
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

        // Port gone (unplugged) → prefer built-in mic, else first available
        if let builtIn = sources.first(where: { $0.id == "Built-In Microphone" || $0.name == "Built-in Mic" }) {
            selectedAudioSource = builtIn
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
                defer { self.isSwitchingLens = false }
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

                    self.metalPipeline?.clearTemporalHistory()
                    self.applyManualExposureAndWB()
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

    /// Push the user's denoiseStrength to the Metal pipeline.
    /// Called when recording starts or controls lock state changes.
    private func updateDenoiseStrength() {
        guard let pipeline = metalPipeline else { return }
        pipeline.denoiseStrength = denoiseStrength
        print("[CameraViewModel] denoiseStrength=\(pipeline.denoiseStrength) mode=\(controlsLocked ? "record" : "preview")")
    }

    /// Debounced exposure/WB push — coalesces rapid slider changes into one hardware call.
    /// Cancels any pending update and schedules a new one 100ms out.
    /// This prevents blocking the main thread on AVCaptureDevice.lockForConfiguration()
    /// during slider drag, which was causing the app to freeze.
    private func scheduleExposureUpdate() {
        exposureDebounceWork?.cancel()
        let work = DispatchWorkItem { [weak self] in
            self?.applyManualExposureAndWB()
        }
        exposureDebounceWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1, execute: work)
    }

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
                // setExposureModeCustom raises NSInvalidArgumentException ("Unsupported
                // exposure mode") when .custom isn't supported by the active format/device
                // (e.g. some ultrawides) — and Swift `try/catch` does NOT catch ObjC
                // exceptions. Guard it, mirroring the setFocusModeLocked fix, and fall back
                // to a supported mode so toggling exposure doesn't crash the app.
                if device.isExposureModeSupported(.custom) {
                    device.setExposureModeCustom(duration: shutterDuration, iso: clampedISO)
                } else if device.isExposureModeSupported(.locked) {
                    device.exposureMode = .locked
                } else if device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposureMode = .continuousAutoExposure
                } else {
                    print("[CameraViewModel] No supported manual exposure mode on \(device.localizedName)")
                }
            }

            if isAutoWhiteBalanceEnabled {
                if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                    device.whiteBalanceMode = .continuousAutoWhiteBalance
                }
            } else {
                let temperatureAndTint = AVCaptureDevice.WhiteBalanceTemperatureAndTintValues(temperature: wbKelvin, tint: wbTint)
                let wbGains = device.deviceWhiteBalanceGains(for: temperatureAndTint)
                let clampedGains = clampWhiteBalanceGains(wbGains, for: device)
                // setWhiteBalanceModeLocked raises NSInvalidArgumentException when .locked
                // isn't supported by the active format/device; guard it (mirroring the
                // setFocusModeLocked fix) and fall back to auto white balance.
                if device.isWhiteBalanceModeSupported(.locked) {
                    device.setWhiteBalanceModeLocked(with: clampedGains)
                } else if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                    device.whiteBalanceMode = .continuousAutoWhiteBalance
                } else if device.isWhiteBalanceModeSupported(.autoWhiteBalance) {
                    device.whiteBalanceMode = .autoWhiteBalance
                } else {
                    print("[CameraViewModel] No supported manual white balance mode on \(device.localizedName)")
                }
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
        metalPipeline?.curveType = selectedCurve
        updateDenoiseStrength()
        activePanel = nil
        refreshStatusLine()
        print("[CameraViewModel] Controls locked (curve=\(selectedCurve.displayName))")
    }

    func unlockControls() {
        controlsLocked = false
        isFocusLocked = false
        metalPipeline?.curveType = selectedCurve
        updateDenoiseStrength()
        refreshStatusLine()
        print("[CameraViewModel] Controls unlocked (curve=\(selectedCurve.displayName))")
    }

    func setFocusPoint(_ point: CGPoint, lock: Bool = false) {
        if meteringMode == .spot {
            captureController.setMeteringMode(.spot, at: point)
        }
        if isRecording || controlsLocked {
            // During recording: tap always locks focus (no continuous AF).
            // Set the focus point and lock at that position.
            isAutoFocus = false
            isFocusLocked = true
            captureController.setFocusPointOfInterest(point, lock: true)
        } else {
            isAutoFocus = true
            isFocusLocked = lock
            captureController.setFocusPointOfInterest(point, lock: lock)
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
        let cMatrix: simd_float3x3
        switch selectedCurve {
        case .linear:
            cMatrix = matrix_identity_float3x3
        case .appleLog2:
            cMatrix = latestColorMatrix ?? WhiteBalanceParams.defaultSensorToBT2020
        case .sLog3Approx:
            cMatrix = latestSGamutMatrix ?? WhiteBalanceParams.defaultSensorToSGamut3Cine
        }
        metalPipeline?.wbParams = WhiteBalanceParams(
            gains: SIMD3<Float>(
                max(gains.redGain / g, 0.01),
                1.0,
                max(gains.blueGain / g, 0.01)
            ),
            colorMatrix: cMatrix
        )
    }

    // MARK: - File Naming

    private static let fileNameDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter
    }()

    static func generateRecordingFileName(
        date: Date = Date(),
        format: RecordingFormat,
        fps: CaptureFrameRate,
        curve: LogCurveType,
        bitrateMbps: Int
    ) -> String {
        let dateString = fileNameDateFormatter.string(from: date)
        return "OWL_\(dateString)_\(format.shortLabel)_\(fps.label)fps_\(curve.fileLabel)_\(bitrateMbps)M.mov"
    }

#if DEBUG
    @discardableResult
    static func runFileNameGenerationTest() -> Bool {
        var components = DateComponents()
        components.year = 2026
        components.month = 9
        components.day = 11
        components.hour = 14
        components.minute = 32
        components.second = 7
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone.current
        guard let testDate = cal.date(from: components) else {
            print("[FileNameTest] FAIL: couldn't construct test date")
            return false
        }

        let name = generateRecordingFileName(
            date: testDate,
            format: .uhd4k,
            fps: .fps24,
            curve: .sLog3Approx,
            bitrateMbps: 150
        )
        let expected = "OWL_20260911_143207_4K_24fps_SLog3_150M.mov"
        guard name == expected else {
            print("[FileNameTest] FAIL: expected \(expected), got \(name)")
            return false
        }
        print("[FileNameTest] PASS: \(name)")
        return true
    }
#endif

    // MARK: - Recording

    func startRecording() {
        guard !isDeviceUnsupportedForLog else {
            errorMessage = "Recording disabled — device has no Bayer RAW."
            return
        }
        guard controlsLocked, !isRecording else { return }

        activeEncodeWidth = selectedFormat.width
        activeEncodeHeight = selectedFormat.height
        activeFPS = selectedFPS.rawValue
        lockAutoModesForRecording()
        updateDenoiseStrength()
        metalPipeline?.curveType = selectedCurve
        metalPipeline?.clearTemporalHistory()
        frameBuffer.flush()
        captureController.setRecordingMode(true)
        let effectiveBitrate = min(selectedBitrate.bitsPerSecond, selectedFormat.maxBitratePreset.bitsPerSecond)
        if effectiveBitrate < selectedBitrate.bitsPerSecond {
            print("[CameraViewModel] Bitrate clamped: \(selectedBitrate.label)Mbps → \(selectedFormat.maxBitratePreset.label)Mbps (max for \(selectedFormat.shortLabel))")
        }

        let recordingDate = Date()
        let fileName = Self.generateRecordingFileName(
            date: recordingDate,
            format: selectedFormat,
            fps: selectedFPS,
            curve: selectedCurve,
            bitrateMbps: effectiveBitrate / 1_000_000
        )
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)

        let includeAudio = selectedAudioSource.portUID != nil

        do {
            try videoWriter.start(
                outputURL: outputURL,
                width: selectedFormat.width,
                height: selectedFormat.height,
                bitrate: effectiveBitrate,
                targetFPS: selectedFPS.rawValue,
                includeAudio: includeAudio,
                curveType: selectedCurve
            )
            isRecording = true
            isRecordingUnsafe = true
            frameIndex = 0
            frameCount = 0
            recordingStartTime = recordingDate
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
        guard let device = captureController.activeDevice else { return }

        do {
            try device.lockForConfiguration()

            // Lock exposure if auto: freeze hardware directly at current duration and ISO
            if isAutoExposureEnabled {
                let curDuration = device.exposureDuration
                let curISO = device.iso
                isoValue = curISO
                let angle = Float(curDuration.seconds * activeFPS * 360.0)
                if angle.isFinite && angle > 0 {
                    shutterValue = max(shutterRange.lowerBound, min(shutterRange.upperBound, angle))
                }
                isAutoExposureEnabled = false
                isAutoExposureAdjusting = false
                let minISO = device.activeFormat.minISO
                let maxISO = device.activeFormat.maxISO
                let clampedISO = max(minISO, min(maxISO, curISO))
                let minDuration = device.activeFormat.minExposureDuration
                let maxDuration = device.activeFormat.maxExposureDuration
                var clampedDuration = curDuration
                if CMTimeCompare(clampedDuration, minDuration) < 0 { clampedDuration = minDuration }
                if CMTimeCompare(clampedDuration, maxDuration) > 0 { clampedDuration = maxDuration }
                if device.isExposureModeSupported(.custom) {
                    device.setExposureModeCustom(duration: clampedDuration, iso: clampedISO)
                } else if device.isExposureModeSupported(.locked) {
                    device.exposureMode = .locked
                }
            }

            // Lock white balance if auto: preserve exact device gains and tint without lossy round-tripping
            if isAutoWhiteBalanceEnabled {
                let currentGains = device.deviceWhiteBalanceGains
                let tempTint = device.temperatureAndTintValues(for: currentGains)
                wbKelvin = max(2000, min(10000, tempTint.temperature))
                wbTint = tempTint.tint
                isAutoWhiteBalanceEnabled = false
                isAutoWhiteBalanceAdjusting = false
                let clamped = clampWhiteBalanceGains(currentGains, for: device)
                if device.isWhiteBalanceModeSupported(.locked) {
                    device.setWhiteBalanceModeLocked(with: clamped)
                }
            }

            // Lock focus at current position without calling setFocusModeLocked with invalid lensPosition.
            if isAutoFocus {
                let pos = device.lensPosition
                if pos >= 0.0 && pos <= 1.0 {
                    focusLensPosition = pos
                }
                isAutoFocus = false
                isFocusLocked = true
                if device.isFocusModeSupported(.locked) {
                    device.focusMode = .locked
                }
            }

            device.unlockForConfiguration()
            updateWBParams(from: device)
        } catch {
            print("[CameraViewModel] lockAutoModesForRecording failed: \(error)")
        }
    }

    func stopRecording() {
        guard isRecording else { return }

        isRecording = false
        isRecordingUnsafe = false
        recordingTimer?.invalidate()
        recordingTimer = nil
        statusText = "Saving…"
        captureController.setRecordingMode(false)

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
        metalPipeline?.trimMemory()

        let realNote = "frames=\(frameCount) drops=\(droppedFrames) fps=\(selectedFPS.label) fmt=\(selectedFormat.shortLabel)"
        print("[CameraViewModel] Recording stopped \(realNote)")
        if let caps = capabilities {
            print("[CameraViewModel] Tester diagnostics:\n\(caps.diagnosticSummary)\n\(realNote)")
        }
    }

    private func saveFinishedRecording(at url: URL) {
        // Validate the file before attempting any save
        guard validateVideoFile(at: url) else {
            statusText = "Save failed"
            errorMessage = "Video file is corrupt — try a lower bitrate for 4K recordings"
            print("[CameraViewModel] File validation failed for \(url.lastPathComponent)")
            try? FileManager.default.removeItem(at: url)
            return
        }

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

    /// Verify the output file exists and has reasonable size before handing to Photos.
    private func validateVideoFile(at url: URL) -> Bool {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? Int64 else {
            print("[CameraViewModel] Cannot stat output file")
            return false
        }
        // Must have at least 100KB to be a valid video
        return size > 100_000
    }

    private func presentFilesFolderPicker() {
        // Build and present the picker asynchronously — the UIDocumentPickerViewController
        // creation involves security-scoped resource coordination that can stall the
        // main thread if done synchronously during an active capture loop.
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                  let rootVC = scene.windows.first(where: { $0.isKeyWindow })?.rootViewController else {
                self.errorMessage = "Files picker unavailable"
                self.refreshStatusLine()
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
        _ = resolveFilesFolderURL()
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
        let frame: RawFrameData?
        if isRecordingUnsafe {
            // FIFO during recording: process every single captured frame in order without skipping
            frame = frameBuffer.dequeue()
        } else {
            // Preview only: drop older backlog frames to keep viewfinder latency minimal
            frame = frameBuffer.dequeueLatest()
        }
        guard let frame else {
            processLock.lock()
            isProcessing = false
            processLock.unlock()
            return
        }
        processFrame(frame) { [weak self] in
            guard let self else { return }
            processLock.lock()
            isProcessing = false
            let remaining = frameBuffer.currentCount
            processLock.unlock()
            if remaining > 0 {
                scheduleProcess()
            }
        }
    }

    struct SendablePixelBuffer: @unchecked Sendable {
        let buffer: CVPixelBuffer
    }

    nonisolated private func processFrame(_ frameData: RawFrameData, completion: @escaping () -> Void) {
        guard isAppActive else { completion(); return }
        guard let pipeline = metalPipeline else { completion(); return }

        pipeline.bayerPattern = frameData.cfaPattern
        pipeline.blackLevel = frameData.blackLevel
        pipeline.whiteLevel = frameData.whiteLevel

        // LSC: calibration > device table > frame default (live DNG).
        pipeline.lscParams = Self.simd4ToLSCParams(frameData.lscCoefficients)
        pipeline.greenBalance = 1.0
        pipeline.iso = frameData.iso

        // Recompute noise coefficients from ISO-dependent profile
        if let prof = noiseProfileForISO {
            noiseCoeffs = prof.coefficients(at: frameData.iso)
        }
        pipeline.noiseShotCoeff = noiseCoeffs.shot
        pipeline.noiseReadCoeff = noiseCoeffs.read

        // Scene-cut detection: >2 stop ISO jump is a secondary trigger; primary motion
        // detection now happens inside the temporal kernel via a global frame metric.
        if lastProcessedISO > 0, abs(frameData.iso - lastProcessedISO) > 2.0 * lastProcessedISO {
            pipeline.clearTemporalHistory()
        }
        lastProcessedISO = frameData.iso

        if let cm = frameData.colorMatrix { latestColorMatrix = cm }
        if let sm = frameData.sgamutMatrix { latestSGamutMatrix = sm }

        let cMatrix: simd_float3x3
        switch pipeline.curveType {
        case .linear:
            cMatrix = matrix_identity_float3x3
            pipeline.headroomScale = 1.0
        case .appleLog2:
            cMatrix = latestColorMatrix ?? WhiteBalanceParams.defaultSensorToBT2020
            pipeline.headroomScale = 10.0
        case .sLog3Approx:
            cMatrix = latestSGamutMatrix ?? WhiteBalanceParams.defaultSensorToSGamut3Cine
            pipeline.headroomScale = 10.0
        }

        if pipeline.isAutoWBEnabled, let gains = frameData.whiteBalanceGains {
            let g = max(gains.greenGain, 0.001)
            pipeline.wbParams = WhiteBalanceParams(
                gains: SIMD3<Float>(
                    max(gains.redGain / g, 0.01),
                    1.0,
                    max(gains.blueGain / g, 0.01)
                ),
                colorMatrix: cMatrix
            )
        } else {
            // When WB is locked (e.g. during recording), ensure the active colorMatrix stays in sync with the selected curve!
            pipeline.wbParams.colorMatrix = cMatrix
        }

        let w = activeEncodeWidth
        let h = activeEncodeHeight

        if isRecordingUnsafe {
            // ── Recording ──
            // 4K or elevated thermal state: use previewFast (radius=2, no local-sigma stats, no chroma history store)
            // to keep GPU cool and responsive.
            // Lower res (OpenGate, 1080p) under normal thermals: full denoise pipeline fits within budget.
            let is4K = w >= 3840
            let isThermalElevated = (metalPipeline?.thermalState.rawValue ?? 0) >= ProcessInfo.ThermalState.serious.rawValue
            pipeline.processingQuality = (is4K || isThermalElevated) ? .previewFast : .recordQuality
            pipeline.process(frameData.pixelBuffer, encodeWidth: w, encodeHeight: h, encodeAsBGRA: true) { [weak self] framed, bgraPB in
                guard let self else { completion(); return }
                handleRecordedFrame(framed, bgraPB: bgraPB, frameData: frameData, completion: completion)
            }
        } else {
            // ── Preview (non-recording): lightweight path, no denoise ──
            pipeline.processingQuality = .previewFast
            // Cap preview resolution to max 1920 (preserving exact aspect ratio) to avoid
            // upscaling to 4K just for on-screen viewfinder rendering.
            let maxPreviewDim = 1920
            let prevW: Int
            let prevH: Int
            if w > maxPreviewDim || h > maxPreviewDim {
                let scale = Double(maxPreviewDim) / Double(max(w, h))
                prevW = (Int(Double(w) * scale) & ~1)
                prevH = (Int(Double(h) * scale) & ~1)
            } else {
                prevW = w
                prevH = h
            }
            pipeline.processPreviewOnly(frameData.pixelBuffer, encodeWidth: prevW, encodeHeight: prevH, encodeAsBGRA: false) { [weak self] framed, _ in
                defer { completion() }
                guard let self, let framed else { return }

                let cfaName: String
                switch frameData.cfaPattern {
                case 0: cfaName = "RGGB"
                case 1: cfaName = "GRBG"
                case 2: cfaName = "GBRG"
                case 3: cfaName = "BGGR"
                default: cfaName = "?\(frameData.cfaPattern)"
                }
                let drops = frameBuffer.droppedCount
                let scopeNow = CACurrentMediaTime()
                if showScopesUnsafe && !isRecordingUnsafe && scopeNow - lastScopeUpdateTime >= 0.1 {
                    lastScopeUpdateTime = scopeNow
                    pipeline.makeScopeData(from: framed) { [weak self] scope in
                        guard let self, let scope else { return }
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            self.scopeData = scope
                        }
                    }
                }

                let frameDataBox = SendableBox(value: frameData)
                let framedBox = SendableBox(value: framed)
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.syncLiveAutoValues(from: frameDataBox.value)
                    self.currentTexture = framedBox.value
                    self.textureChangeCount &+= 1
                    if self.cfaLabel != cfaName { self.cfaLabel = cfaName }
                    if self.droppedFrames != drops { self.droppedFrames = drops }
                }
            }
        }
    }

    /// Shared completion for both preview-only and full-quality recording paths.
    nonisolated private func handleRecordedFrame(
        _ framed: MTLTexture?,
        bgraPB: CVPixelBuffer?,
        frameData: RawFrameData,
        completion: @escaping () -> Void
    ) {
        defer { completion() }
        guard let framed else { return }

        let cfaName: String
        switch frameData.cfaPattern {
        case 0: cfaName = "RGGB"
        case 1: cfaName = "GRBG"
        case 2: cfaName = "GBRG"
        case 3: cfaName = "BGGR"
        default: cfaName = "?\(frameData.cfaPattern)"
        }
        let drops = frameBuffer.droppedCount

        if let bgraPB, isRecordingUnsafe {
            if self.videoWriter.appendFrame(pixelBuffer: bgraPB) {
                self.frameIndex += 1
            }
        }

        let frameDataBox = SendableBox(value: frameData)
        let framedBox = SendableBox(value: framed)
        let recordedIndex = self.frameIndex
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.syncLiveAutoValues(from: frameDataBox.value)
            self.currentTexture = framedBox.value
            self.textureChangeCount &+= 1
            if self.cfaLabel != cfaName { self.cfaLabel = cfaName }
            if self.droppedFrames != drops { self.droppedFrames = drops }
            self.frameCount = Int(recordedIndex)
        }
    }

    /// Convert SIMD4 LSC coefficients (radial k1, radial k2, azimuth, unused) to LSCParams.
    nonisolated private static func simd4ToLSCParams(_ coeffs: SIMD4<Float>) -> LSCParams {
        let r2 = coeffs[0]
        let r4 = coeffs[1]
        return LSCParams(
            radialR: r2,
            radialG: r2,
            radialB: r2,
            radial4R: r4,
            radial4G: r4,
            radial4B: r4,
            azimuthR: 0, azimuthG: 0, azimuthB: 0
        )
    }

    private func syncLiveAutoValues(from frameData: RawFrameData) {
        guard let device = captureController.activeDevice else { return }
        if isAutoExposureEnabled {
            if frameData.iso > 0 && abs(isoValue - frameData.iso) >= 1.0 {
                isoValue = frameData.iso
            }
            if frameData.exposureDurationSeconds > 0 {
                let angle = Float(frameData.exposureDurationSeconds * activeFPS * 360.0)
                let clampedAngle = max(shutterRange.lowerBound, min(shutterRange.upperBound, angle))
                if abs(shutterValue - clampedAngle) >= 0.5 {
                    shutterValue = clampedAngle
                }
            }
            let isAdj = device.isAdjustingExposure
            if isAutoExposureAdjusting != isAdj {
                isAutoExposureAdjusting = isAdj
            }
        } else if isAutoExposureAdjusting {
            isAutoExposureAdjusting = false
        }

        if isAutoWhiteBalanceEnabled {
            if let gains = frameData.whiteBalanceGains {
                let clamped = clampWhiteBalanceGains(gains, for: device)
                let temperatureAndTint = device.temperatureAndTintValues(for: clamped)
                let temp = max(2000, min(10000, temperatureAndTint.temperature))
                if abs(wbKelvin - temp) >= 25 {
                    wbKelvin = temp
                }
                wbTint = temperatureAndTint.tint
            }
            let isAdj = device.isAdjustingWhiteBalance
            if isAutoWhiteBalanceAdjusting != isAdj {
                isAutoWhiteBalanceAdjusting = isAdj
            }
        } else if isAutoWhiteBalanceAdjusting {
            isAutoWhiteBalanceAdjusting = false
        }
    }
}

