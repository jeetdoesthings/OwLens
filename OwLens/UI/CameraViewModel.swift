import Foundation
import AVFoundation
import Combine
import Metal
import Photos
import QuartzCore
import simd
import UIKit
import UniformTypeIdentifiers
import SwiftUI
import os

/// Dedicated observable object holding live audio peak levels.
/// Throttled to ~15 Hz with a deadband filter, isolating VU meter animations
/// to `MicButton` without invalidating the parent camera HUD.
@MainActor
final class AudioMonitor: ObservableObject {
    @Published private(set) var level: Float = 0.0
    private var lastUpdateTime: CFTimeInterval = 0

    func update(peak: Float) {
        let now = CACurrentMediaTime()
        // Throttle UI update frequency to ~15 Hz (66 ms)
        guard now - lastUpdateTime >= 0.066 else { return }

        let newLevel: Float
        if peak >= level {
            newLevel = peak
        } else {
            newLevel = level * 0.80 + peak * 0.20
        }

        // 0.015 deadband to prevent microscopic floating-point jitter
        if abs(newLevel - level) >= 0.015 || (newLevel == 0 && level != 0) {
            level = newLevel
            lastUpdateTime = now
        }
    }

    func reset() {
        if level != 0 {
            level = 0
            lastUpdateTime = CACurrentMediaTime()
        }
    }
}

/// Central view model — CaptureController → RawFrameBuffer → MetalPipeline → preview + VideoWriter.
@MainActor
final class CameraViewModel: NSObject, ObservableObject, UIDocumentPickerDelegate {
    // MARK: - Published State

    /// High-speed direct preview feed to MTKView. Eliminates per-frame SwiftUI re-renders.
    let previewFeed = PreviewFeed()
    var currentTexture: MTLTexture? { previewFeed.currentTexture }
    @Published var isRecording = false
    @Published var isSaving = false
    @Published var controlsLocked = false
    private var wasControlsLockedBeforeRecording = false
    private var wasAutoFocusBeforeRecording = true
    private var wasAutoWBBeforeRecording = false
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
            updateStorageEstimate()
            refreshStatusLine()
        }
    }
    @Published var selectedFPS: CaptureFrameRate = .fps30 {
        didSet {
            guard !controlsLocked, !isRecording else { return }
            captureController.setCaptureFPS(selectedFPS.rawValue)
            activeFPS = selectedFPS.rawValue
            // 180° shutter rule is now natively maintained by the angle system.
            // We just need to refresh limits and push the new duration to hardware.
            seedControlRanges()
            if isCameraReady { applyManualExposureAndWB() }
            updateStorageEstimate()
            refreshStatusLine()
        }
    }
    @Published var selectedBitrate: BitratePreset = .mbps150 {
        didSet {
            guard !controlsLocked else { return }
            updateStorageEstimate()
            refreshStatusLine()
        }
    }
    @Published var selectedCodec: VideoCodecOption = .hevc {
        didSet {
            guard !isRecording else { return }
            updateStorageEstimate()
            refreshStatusLine()
        }
    }
    @Published private(set) var selectedSaveDestination: VideoSaveDestination = .photos
    @Published var audioSources: [AudioSourceOption] = [.none]
    @Published var selectedAudioSource: AudioSourceOption = .none {
        didSet {
            isAudioMutedUnsafe = (selectedAudioSource.portUID == nil)
            guard !controlsLocked, !isRecording else { return }
            // Skip no-op reassign (same port) — avoids hang loops
            guard oldValue.portUID != selectedAudioSource.portUID else { return }
            applyAudioSource()
            updateStorageEstimate()
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
    let scopeMonitor = ScopeMonitor()
    var scopeData: ScopeData { scopeMonitor.scopeData }
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

    @Published var wbTint: Float = 0.0 {
        didSet {
            guard !isAutoWhiteBalanceEnabled, !controlsLocked else { return }
            scheduleExposureUpdate()
        }
    }

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
    @Published var averageFrameTimeMs: Double = 0.0
    @Published var cfaLabel: String = "—"

    /// Runtime device probe (set once at setup).
    @Published private(set) var capabilities: DeviceCapabilities?
    /// No Bayer RAW — hard gate, record disabled.
    @Published private(set) var isDeviceUnsupportedForLog = false
    /// Allow-list miss — soft warning, record still allowed.
    @Published private(set) var showUnverifiedDeviceWarning = false
    /// True when session started (or hard-failed setup) so splash can dismiss.
    @Published private(set) var isCameraReady = false {
        didSet {
            if isCameraReady {
                updateStorageEstimate()
                startStorageMonitor()
            }
        }
    }

    @Published private(set) var isoValue: Float = 100
    @Published private(set) var shutterValue: Float = 180
    @Published private(set) var wbKelvin: Float = 5600

    @Published var isAutoWhiteBalanceEnabled: Bool = false {
        didSet {
            guard oldValue != isAutoWhiteBalanceEnabled else { return }
            if !isAutoWhiteBalanceEnabled {
                wbStopIndex = ExposureStops.nearestIndex(in: wbStops, to: wbKelvin)
            }
            if isCameraReady { applyManualExposureAndWB() }
        }
    }
    /// Auto white balance locks by default when recording starts to prevent color drift.
    @Published var isAutoWBLockEnabled: Bool = true

    @Published private(set) var isAutoWhiteBalanceAdjusting: Bool = false

    /// Formatted remaining record time (e.g. "45:12" or "1h 20m").
    @Published var estimatedRecordTimeText: String = "—"
    /// Estimated remaining recording time in seconds.
    @Published var remainingRecordSeconds: Int = 0
    /// Available storage capacity in bytes.
    @Published var availableStorageBytes: Int64 = 0

    private var storageRefreshTimer: Timer?


    // Focus properties
    @Published var isFocusLocked: Bool = false
    @Published var isAutoFocus: Bool = true {
        didSet {
            guard !controlsLocked else { return }
            if isAutoFocus {
                isFocusLocked = false
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
            guard !controlsLocked else { return }
            scheduleExposureUpdate()
        }
    }

    func setISOWithSnapping(_ rawValue: Float) {
        guard !controlsLocked else { return }
        let snapTargets: [Float] = [50, 100, 200, 400, 800, 1600, 3200].filter { isoRange.contains($0) }
        var finalValue = rawValue
        var didSnap = false

        for target in snapTargets {
            let logDiff = abs(log2(max(1.0, rawValue)) - log2(max(1.0, target)))
            // Magnetic snap radius of ~0.10 stops (~7% difference)
            if logDiff < 0.10 {
                finalValue = target
                didSnap = true
                break
            }
        }

        if !didSnap {
            // Round to nearest 5 for clean intervals
            finalValue = (finalValue / 5.0).rounded() * 5.0
        }

        finalValue = max(isoRange.lowerBound, min(isoRange.upperBound, finalValue))

        if finalValue != isoValue {
            let wasDifferent = abs(isoValue - finalValue) >= 1.0
            isoValue = finalValue
            if let idx = isoStops.firstIndex(where: { abs($0 - finalValue) < 5 }) {
                isoStopIndex = idx
            }
            if didSnap && wasDifferent {
                Haptics.selection()
            }
            if isCameraReady { scheduleExposureUpdate() }
        }
    }

    func setShutterAngleWithSnapping(_ rawValue: Float) {
        guard !controlsLocked else { return }
        let snapTargets: [Float] = [45.0, 90.0, 144.0, 172.8, 180.0, 360.0].filter { shutterRange.contains($0) }
        var finalValue = rawValue
        var didSnap = false
        
        for target in snapTargets {
            // Magnetic snap radius of 6 degrees
            if abs(rawValue - target) < 6.0 {
                finalValue = target
                didSnap = true
                break
            }
        }
        
        finalValue = max(shutterRange.lowerBound, min(shutterRange.upperBound, finalValue))
        
        if finalValue != shutterValue {
            let wasDifferent = abs(shutterValue - finalValue) > 0.05
            shutterValue = finalValue
            if didSnap && wasDifferent {
                Haptics.selection()
            }
            if isCameraReady { scheduleExposureUpdate() }
        }
    }

    var shutterSpeedText: String {
        let angle = Double(shutterValue)
        let fps = activeFPS
        guard angle > 0, fps > 0 else { return "" }
        let duration = angle / (360.0 * fps)
        guard duration > 0 else { return "" }
        let denom = Int((1.0 / duration).rounded())
        return "1/\(denom)s"
    }
    @Published var wbStopIndex: Int = 0 {
        didSet {
            guard !wbStops.isEmpty else { return }
            let i = ExposureStops.clampIndex(wbStopIndex, count: wbStops.count)
            if i != wbStopIndex { wbStopIndex = i; return }
            guard !isAutoWhiteBalanceEnabled else { return }
            let v = wbStops[i]
            guard v != wbKelvin else { return }
            wbKelvin = v
            guard !controlsLocked else { return }
            scheduleExposureUpdate()
        }
    }

    @Published var activePanel: ControlPanel? = nil

    // MARK: - Real-Time Audio & Battery Telemetry
    let audioMonitor = AudioMonitor()
    var audioLevel: Float { audioMonitor.level }
    nonisolated(unsafe) private var lastAudioLevelUpdateTime: CFTimeInterval = 0

    @Published var currentTimeString: String = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm"
        f.locale = Locale(identifier: "en_US_POSIX")
        return f.string(from: Date())
    }()
    @Published var batteryLevel: Float = {
        UIDevice.current.isBatteryMonitoringEnabled = true
        let raw = UIDevice.current.batteryLevel
        return raw >= 0 ? raw : -1.0
    }()
    @Published var isBatteryCharging: Bool = false

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter
    }()

    // MARK: - HUD State & Toast Notifications
    @Published var isHUDHidden: Bool = false
    @Published var activeToast: String? = nil
    private var toastWorkItem: DispatchWorkItem?

    var isoRange: ClosedRange<Float> = 50...2000
    var shutterRange: ClosedRange<Float> = 11.25...360.0

    // MARK: - Pipeline

    let captureController = CaptureController()
    nonisolated let metalPipeline: MetalPipeline?
    nonisolated private let videoWriter = VideoWriter()
    /// Capacity 16: absorb burst stalls while keeping latency low.
    nonisolated(unsafe) private let frameBuffer = RawFrameBuffer(capacity: 16)

    private var cancellables = Set<AnyCancellable>()
    /// Debounce rapid slider changes to avoid blocking the main thread on lockForConfiguration().
    private var exposureDebounceWork: DispatchWorkItem?
    private var recordingStartTime: Date?
    private var recordingTimer: Timer?
    private var saveBackgroundTask: UIBackgroundTaskIdentifier = .invalid
    nonisolated(unsafe) private var frameIndex: Int64 = 0
    private var filesFolderBookmark: Data?
    private let filesFolderBookmarkKey = "OwLens.FilesFolderBookmark"

    private let processQueue = DispatchQueue(label: "raw.process.queue", qos: .userInteractive)
    private let recordingQueue = DispatchQueue(label: "com.owlens.recording", qos: .userInteractive)
    nonisolated private let processLock = OSAllocatedUnfairLock()
    nonisolated(unsafe) private var freeSlots: Set<Int> = [0, 1, 2]

    // Sequencer for guaranteeing strictly monotonic frame ordering during recording
    nonisolated(unsafe) private var nextDispatchSequenceID: UInt64 = 0
    nonisolated(unsafe) private var nextOutputSequenceID: UInt64 = 0
    nonisolated(unsafe) private var reorderBuffer: [UInt64: (MTLTexture?, CVPixelBuffer?, RawFrameData)] = [:]

    nonisolated(unsafe) private var activeEncodeWidth = 2016
    nonisolated(unsafe) private var activeEncodeHeight = 1512
    nonisolated(unsafe) private var activeFPS: Double = 30
    nonisolated(unsafe) private var isRecordingUnsafe = false
    nonisolated(unsafe) private var isAudioMutedUnsafe = false
    nonisolated(unsafe) private var showScopesUnsafe = true
    nonisolated(unsafe) private var lastScopeUpdateTime: CFTimeInterval = 0
    nonisolated(unsafe) private var lastMainActorSyncTime: CFTimeInterval = 0
    nonisolated(unsafe) private var lastReportedDrops: Int = 0
    nonisolated(unsafe) var isAppActive = true
    /// Measured LSC override from device calibration (set once at setup).
    nonisolated(unsafe) private var lscOverride: LSCCoefficients?

    /// Measured noise coefficients from device calibration (set once at setup).
    nonisolated(unsafe) private var noiseCoeffs: (shot: Float, read: Float) = (0.012, 0.0004)
    /// Noise profile for per-ISO coefficient lookup (nil on unknown device).
    nonisolated(unsafe) private var noiseProfileForISO: NoiseProfile?
    /// Latest calibrated color matrices extracted from DNG metadata.
    nonisolated(unsafe) private var latestColorMatrix: simd_float3x3?
    nonisolated(unsafe) private var latestSGamutMatrix: simd_float3x3?

    enum ControlPanel: String, Identifiable {
        case exposure, iso, shutter, wb, focus, fps, format, bitrate, mic, lens, save, logCurve
        var id: String { rawValue }
    }

    // MARK: - Init

    override init() {
        metalPipeline = MetalPipeline()
        super.init()
        startClockAndBatteryMonitoring()
#if DEBUG
        // Automated unit tests and benchmarks are invoked via test suites rather than on camera startup.
#endif
        loadFilesFolderBookmark()
        metalPipeline?.curveType = selectedCurve
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

        // Refresh mic list and handle disconnects when route changes
        NotificationCenter.default.publisher(for: AVAudioSession.routeChangeNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] note in
                guard let self else { return }
                self.refreshAudioSources()
                if let reasonValue = note.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt,
                   let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue) {
                    if reason == .oldDeviceUnavailable {
                        print("[CameraViewModel] Audio route oldDeviceUnavailable, falling back to default mic")
                    }
                }
            }
            .store(in: &cancellables)

        // Handle audio interruptions (incoming phone call, alarm, Siri)
        NotificationCenter.default.publisher(for: AVAudioSession.interruptionNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] note in
                guard let self else { return }
                guard let userInfo = note.userInfo,
                      let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
                      let type = AVAudioSession.InterruptionType(rawValue: typeValue) else { return }
                switch type {
                case .began:
                    if self.isRecording {
                        self.stopRecording()
                        self.errorMessage = "Audio interrupted (phone call/alarm) — Recording stopped"
                        self.refreshStatusLine()
                    }
                case .ended:
                    if let optionsValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt {
                        let options = AVAudioSession.InterruptionOptions(rawValue: optionsValue)
                        if options.contains(.shouldResume) {
                            DispatchQueue.global(qos: .userInitiated).async {
                                try? AVAudioSession.sharedInstance().setActive(true)
                            }
                        }
                    }
                    self.refreshAudioSources()
                @unknown default:
                    break
                }
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
            self.processAudioSample(sample)
            // isRecordingUnsafe set from MainActor when record starts/stops
            if self.isRecordingUnsafe {
                self.recordingQueue.async { [weak self] in
                    guard let self, self.isRecordingUnsafe else { return }
                    _ = self.videoWriter.appendAudio(sampleBuffer: sample)
                }
            }
        }

        updateStorageEstimate()
        startStorageMonitor()
    }

    private func handleAppInactive() {
        isAppActive = false
        UIApplication.shared.isIdleTimerDisabled = false
        storageRefreshTimer?.invalidate()
        storageRefreshTimer = nil
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
        UIApplication.shared.isIdleTimerDisabled = true
        UIDevice.current.isBatteryMonitoringEnabled = true
        updateBatteryStatus()
        updateCurrentTime()
        updateStorageEstimate()
        startStorageMonitor()
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
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(
                .playAndRecord,
                mode: .videoRecording,
                options: [.defaultToSpeaker, .allowBluetoothHFP, .allowBluetoothA2DP, .mixWithOthers]
            )
            try audioSession.setPreferredSampleRate(48_000)
            try audioSession.setPreferredIOBufferDuration(0.02)
            DispatchQueue.global(qos: .userInitiated).async {
                try? audioSession.setActive(true)
            }
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
            UIApplication.shared.isIdleTimerDisabled = true
            Self.cleanStaleTemporaryRecordings()
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
        UIApplication.shared.isIdleTimerDisabled = false
        levelMonitor.stop()
        captureController.stopSession()
        frameBuffer.flush()
        DispatchQueue.global(qos: .userInitiated).async {
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        }
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

    func toggleHUDVisibility() {
        Haptics.selection()
        withAnimation(.easeInOut(duration: 0.22)) {
            isHUDHidden.toggle()
        }
    }

    func showToast(_ message: String) {
        toastWorkItem?.cancel()
        withAnimation(.spring(response: 0.24, dampingFraction: 0.8)) {
            activeToast = message
        }
        let work = DispatchWorkItem { [weak self] in
            withAnimation(.easeOut(duration: 0.25)) {
                self?.activeToast = nil
            }
        }
        toastWorkItem = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.6, execute: work)
    }

    private func startClockAndBatteryMonitoring() {
        UIDevice.current.isBatteryMonitoringEnabled = true
        updateBatteryStatus()
        updateCurrentTime()

        // Hardware PMU needs a cycle after enabling battery monitoring to populate true level
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            self?.updateBatteryStatus()
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.updateBatteryStatus()
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
            self?.updateBatteryStatus()
        }

        NotificationCenter.default.publisher(for: UIDevice.batteryLevelDidChangeNotification)
            .sink { [weak self] _ in self?.updateBatteryStatus() }
            .store(in: &cancellables)
        NotificationCenter.default.publisher(for: UIDevice.batteryStateDidChangeNotification)
            .sink { [weak self] _ in self?.updateBatteryStatus() }
            .store(in: &cancellables)
        NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)
            .sink { [weak self] _ in
                UIDevice.current.isBatteryMonitoringEnabled = true
                self?.updateBatteryStatus()
                self?.updateCurrentTime()
            }
            .store(in: &cancellables)

        // 1 Hz timer to keep 24-hour clock and battery level accurate
        Timer.publish(every: 1.0, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                guard let self else { return }
                self.updateCurrentTime()
                self.updateBatteryStatus()
            }
            .store(in: &cancellables)
    }

    private func updateCurrentTime() {
        let formatted = Self.timeFormatter.string(from: Date())
        if currentTimeString != formatted {
            currentTimeString = formatted
        }
    }

    private func updateBatteryStatus() {
        if !UIDevice.current.isBatteryMonitoringEnabled {
            UIDevice.current.isBatteryMonitoringEnabled = true
        }
        let raw = UIDevice.current.batteryLevel
        if raw >= 0 {
            if abs(batteryLevel - raw) > 0.001 {
                batteryLevel = raw
            }
        }
        let state = UIDevice.current.batteryState
        let charging = (state == .charging || state == .full)
        if isBatteryCharging != charging {
            isBatteryCharging = charging
        }
    }

    nonisolated private func processAudioSample(_ sample: CMSampleBuffer) {
        let now = CACurrentMediaTime()
        if isAudioMutedUnsafe {
            // Throttled reset to prevent queuing up to 100 async tasks/sec on the main runloop when muted
            guard now - lastAudioLevelUpdateTime >= 0.25 else { return }
            lastAudioLevelUpdateTime = now
            DispatchQueue.main.async { [weak self] in
                self?.audioMonitor.reset()
            }
            return
        }
        // Align background sample processing directly with AudioMonitor's ~15 Hz (66 ms) UI cadence
        guard now - lastAudioLevelUpdateTime >= 0.066 else { return }
        lastAudioLevelUpdateTime = now

        guard let blockBuffer = CMSampleBufferGetDataBuffer(sample) else { return }
        var totalLength: Int = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        guard CMBlockBufferGetDataPointer(blockBuffer, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &totalLength, dataPointerOut: &dataPointer) == noErr,
              let dataPtr = dataPointer, totalLength > 0 else { return }

        let sampleCount = totalLength / MemoryLayout<Int16>.size
        guard sampleCount > 0 else { return }

        var maxAmp: Float = 0
        let step = max(1, sampleCount / 48)
        dataPtr.withMemoryRebound(to: Int16.self, capacity: sampleCount) { ptr in
            var i = 0
            while i < sampleCount {
                let val = abs(Float(ptr[i])) / 32768.0
                if val > maxAmp { maxAmp = val }
                i += step
            }
        }

        let peak = min(1.0, maxAmp)
        DispatchQueue.main.async { [weak self] in
            self?.audioMonitor.update(peak: peak)
        }
    }

    func toggleGrid() {
        showGrid.toggle()
        showToast(showGrid ? "Framing Grid: ON" : "Framing Grid: OFF")
    }

    func toggleLevel() {
        showLevel.toggle()
        showToast(showLevel ? "Horizon Level: ON" : "Horizon Level: OFF")
    }

    func toggleClipping() {
        showClipping.toggle()
        showToast(showClipping ? "Zebra Clipping: ON" : "Zebra Clipping: OFF")
    }
    
    func toggleFocusPeaking() {
        showFocusPeaking.toggle()
        showToast(showFocusPeaking ? "Focus Peaking: ON" : "Focus Peaking: OFF")
    }

    func toggleScopes() {
        showScopes.toggle()
        showScopesUnsafe = showScopes
        if !showScopes {
            scopeMonitor.reset()
        }
        showToast(showScopes ? "Waveform & Histogram: ON" : "Scopes: OFF")
    }

    func togglePreviewDisplayMode() {
        previewDisplayMode = previewDisplayMode == .log ? .normalVideo : .log
    }

    func cycleFormat() {
        guard !isRecording else { return }
        let cases = RecordingFormat.allCases
        if let idx = cases.firstIndex(of: selectedFormat) {
            let next = cases[(idx + 1) % cases.count]
            selectedFormat = next
            showToast("Format: \(next.displayName)")
        }
    }

    func cycleFPS() {
        guard !controlsLocked, !isRecording else { return }
        let cases = CaptureFrameRate.allCases
        if let idx = cases.firstIndex(of: selectedFPS) {
            let next = cases[(idx + 1) % cases.count]
            selectedFPS = next
            showToast("\(next.label) FPS")
        }
    }

    func toggleLogCurve() {
        guard !isRecording else { return }
        let cases = LogCurveType.uiCases
        if let idx = cases.firstIndex(of: selectedCurve) {
            let nextIdx = (idx + 1) % cases.count
            selectedCurve = cases[nextIdx]
        } else {
            selectedCurve = cases.first ?? .sLog3Approx
        }
        showToast("\(selectedCurve.displayName) · 10-Bit")
    }

    func toggleDisplayLUT() {
        showDisplayLUT.toggle()
        showToast(showDisplayLUT ? "Rec.709 Preview: ON" : "Log (Flat) Preview: ON")
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
            let minAngle = max(1.0, minDur * 360.0 * Float(activeFPS))
            if minAngle <= maxAngle {
                shutterRange = minAngle...maxAngle
            }

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
        guard !controlsLocked else { return }
        isoStopIndex = ExposureStops.clampIndex(isoStopIndex + delta, count: isoStops.count)
    }

    func nudgeShutterAngle(_ direction: Int) {
        guard !controlsLocked else { return }
        let snapTargets: [Float] = [11.25, 22.5, 45.0, 90.0, 144.0, 172.8, 180.0, 270.0, 360.0]
            .filter { shutterRange.contains($0) }
        if direction > 0 {
            if let next = snapTargets.first(where: { $0 > shutterValue + 0.5 }) {
                setShutterAngleWithSnapping(next)
            } else {
                setShutterAngleWithSnapping(min(shutterRange.upperBound, shutterValue + 5.0))
            }
        } else {
            if let prev = snapTargets.last(where: { $0 < shutterValue - 0.5 }) {
                setShutterAngleWithSnapping(prev)
            } else {
                setShutterAngleWithSnapping(max(shutterRange.lowerBound, shutterValue - 5.0))
            }
        }
    }

    func nudgeWB(_ delta: Int) {
        guard !controlsLocked, !isAutoWhiteBalanceEnabled else { return }
        wbStopIndex = ExposureStops.clampIndex(wbStopIndex + delta, count: wbStops.count)
    }

    func nudgeTint(_ delta: Float) {
        guard !controlsLocked, !isAutoWhiteBalanceEnabled else { return }
        wbTint = max(-50.0, min(50.0, (wbTint + delta).rounded()))
    }

    func nudgeFocus(_ delta: Float) {
        guard !controlsLocked else { return }
        isAutoFocus = false
        focusLensPosition = max(0.0, min(1.0, focusLensPosition + delta * 0.05))
    }

    // MARK: - Manual Controls (live when unlocked)

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

    /// Push ISO / shutter / WB to hardware asynchronously off the main thread.
    func applyManualExposureAndWB() {
        guard let device = captureController.activeDevice ??
                AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }

        let iso = isoValue
        let shutter = shutterValue
        let fps = activeFPS
        let autoWB = isAutoWhiteBalanceEnabled
        let kelvin = wbKelvin
        let tint = wbTint

        // Dispatch hardware configuration off the main thread to completely prevent UI hitching
        DispatchQueue.global(qos: .userInitiated).async { [weak self, weak device] in
            guard let device else { return }
            var explicitGains: AVCaptureDevice.WhiteBalanceGains? = nil
            do {
                try device.lockForConfiguration()

                let clampedISO = max(device.activeFormat.minISO, min(device.activeFormat.maxISO, iso))
                var shutterDuration = CMTimeMakeWithSeconds((Double(shutter) / 360.0) / fps, preferredTimescale: 1_000_000)
                let minD = device.activeFormat.minExposureDuration
                let maxD = device.activeFormat.maxExposureDuration
                if CMTimeCompare(shutterDuration, minD) < 0 { shutterDuration = minD }
                if CMTimeCompare(shutterDuration, maxD) > 0 { shutterDuration = maxD }

                if device.isExposureModeSupported(.custom) {
                    device.setExposureModeCustom(duration: shutterDuration, iso: clampedISO, completionHandler: nil)
                } else if device.isExposureModeSupported(.locked) {
                    device.exposureMode = .locked
                } else if device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposureMode = .continuousAutoExposure
                } else {
                    print("[CameraViewModel] No supported manual exposure mode on \(device.localizedName)")
                }

                if autoWB {
                    if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                        device.whiteBalanceMode = .continuousAutoWhiteBalance
                    }
                } else {
                    let temperatureAndTint = AVCaptureDevice.WhiteBalanceTemperatureAndTintValues(temperature: kelvin, tint: tint)
                    let wbGains = device.deviceWhiteBalanceGains(for: temperatureAndTint)
                    let maxGain = device.maxWhiteBalanceGain
                    let clamped = AVCaptureDevice.WhiteBalanceGains(
                        redGain: max(1.0, min(maxGain, wbGains.redGain)),
                        greenGain: max(1.0, min(maxGain, wbGains.greenGain)),
                        blueGain: max(1.0, min(maxGain, wbGains.blueGain))
                    )
                    if device.isWhiteBalanceModeSupported(.locked) {
                        device.setWhiteBalanceModeLocked(with: clamped, completionHandler: nil)
                    } else if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                        device.whiteBalanceMode = .continuousAutoWhiteBalance
                    } else if device.isWhiteBalanceModeSupported(.autoWhiteBalance) {
                        device.whiteBalanceMode = .autoWhiteBalance
                    } else {
                        print("[CameraViewModel] No supported manual white balance mode on \(device.localizedName)")
                    }
                    explicitGains = clamped
                }
                device.unlockForConfiguration()
            } catch {
                print("[CameraViewModel] applyManualExposureAndWB: \(error)")
            }

            let finalGains = explicitGains
            DispatchQueue.main.async { [weak self] in
                guard let self, let activeDevice = self.captureController.activeDevice else { return }
                self.updateWBParams(from: activeDevice, explicitGains: finalGains)
            }
        }
    }

    /// Freeze active auto white balance directly on hardware at its current values,
    /// ensuring that locking controls or starting recording prevents color drift.
    private func freezeAutoWhiteBalance(on device: AVCaptureDevice) {
        guard isAutoWhiteBalanceEnabled else { return }
        let currentGains = device.deviceWhiteBalanceGains
        let tempTint = device.temperatureAndTintValues(for: currentGains)
        wbKelvin = max(2000, min(10000, tempTint.temperature))
        wbStopIndex = ExposureStops.nearestIndex(in: wbStops, to: wbKelvin)
        wbTint = tempTint.tint
        isAutoWhiteBalanceEnabled = false
        isAutoWhiteBalanceAdjusting = false
        let clamped = clampWhiteBalanceGains(currentGains, for: device)
        if device.isWhiteBalanceModeSupported(.locked) {
            device.setWhiteBalanceModeLocked(with: clamped)
        }
        updateWBParams(from: device, explicitGains: clamped)
    }

    func lockControls() {
        guard let device = captureController.activeDevice ??
                AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }

        do {
            try device.lockForConfiguration()
            freezeAutoWhiteBalance(on: device)
            device.unlockForConfiguration()
        } catch {
            print("[CameraViewModel] lockControls: \(error)")
        }

        // Push manual exposure and WB settings
        applyManualExposureAndWB()

        if let device = captureController.activeDevice {
            isoRange = device.activeFormat.minISO...device.activeFormat.maxISO
        }

        controlsLocked = true
        metalPipeline?.curveType = selectedCurve
        activePanel = nil
        refreshStatusLine()
        print("[CameraViewModel] Controls locked (curve=\(selectedCurve.displayName))")
    }

    func unlockControls() {
        controlsLocked = false
        isFocusLocked = false
        metalPipeline?.curveType = selectedCurve
        refreshStatusLine()
        print("[CameraViewModel] Controls unlocked (curve=\(selectedCurve.displayName))")
    }

    private func restoreAutoModesAfterRecording() {
        if wasAutoWBBeforeRecording {
            isAutoWhiteBalanceEnabled = true
        }
    }

    func setFocusPoint(_ point: CGPoint, lock: Bool = true) {
        if meteringMode == .spot {
            captureController.setMeteringMode(.spot, at: point)
        }
        isAutoFocus = true
        isFocusLocked = lock
        captureController.setFocusPointOfInterest(point)
        showToast("AF Locked")
    }

    private func clampWhiteBalanceGains(_ gains: AVCaptureDevice.WhiteBalanceGains, for device: AVCaptureDevice) -> AVCaptureDevice.WhiteBalanceGains {
        let maxGain = device.maxWhiteBalanceGain
        return AVCaptureDevice.WhiteBalanceGains(
            redGain: max(1.0, min(maxGain, gains.redGain)),
            greenGain: max(1.0, min(maxGain, gains.greenGain)),
            blueGain: max(1.0, min(maxGain, gains.blueGain))
        )
    }

    private func updateWBParams(from device: AVCaptureDevice, explicitGains: AVCaptureDevice.WhiteBalanceGains? = nil) {
        metalPipeline?.isAutoWBEnabled = isAutoWhiteBalanceEnabled
        let gains = explicitGains ?? device.deviceWhiteBalanceGains
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
        metalPipeline?.headroomScale = 1.0
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
            format: .openGate,
            fps: .fps24,
            curve: .sLog3Approx,
            bitrateMbps: 150
        )
        let expected = "OWL_20260911_143207_OG_24fps_SLog3_150M.mov"
        guard name == expected else {
            print("[FileNameTest] FAIL: expected \(expected), got \(name)")
            return false
        }
        print("[FileNameTest] PASS: \(name)")
        return true
    }
#endif

    nonisolated static func cleanStaleTemporaryRecordings() {
        let fileManager = FileManager.default
        let tempDir = fileManager.temporaryDirectory
        guard let contents = try? fileManager.contentsOfDirectory(at: tempDir, includingPropertiesForKeys: [.creationDateKey]) else { return }
        let oneHourAgo = Date().addingTimeInterval(-3600)
        for url in contents where url.pathExtension.lowercased() == "mov" && url.lastPathComponent.hasPrefix("OWL_") {
            if let attrs = try? fileManager.attributesOfItem(atPath: url.path),
               let creationDate = attrs[.creationDate] as? Date,
               creationDate < oneHourAgo {
                try? fileManager.removeItem(at: url)
                print("[CameraViewModel] Cleaned stale temp recording: \(url.lastPathComponent)")
            }
        }
    }

    // MARK: - Recording

    private let initializationTime: Date = Date()

    func toggleRecording() {
        guard isCameraReady, !isDeviceUnsupportedForLog, !isSaving else { return }
        // Guard against any spurious triggers during initial view hierarchy stabilization
        guard Date().timeIntervalSince(initializationTime) > 1.0 else { return }
        Haptics.notification(.success)
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    func startRecording() {
        guard !isDeviceUnsupportedForLog else {
            errorMessage = "Recording disabled — device has no Bayer RAW."
            return
        }
        guard !isRecording, !isSaving else { return }

        // Clean stale temporary recordings from previous interrupted sessions in background
        Task.detached(priority: .utility) {
            Self.cleanStaleTemporaryRecordings()
        }

        // Verify storage capacity before starting recording
        updateStorageEstimate()
        guard remainingRecordSeconds > 5 else {
            errorMessage = "Storage Full (<500MB left) — Cannot record."
            return
        }

        wasControlsLockedBeforeRecording = controlsLocked
        wasAutoWBBeforeRecording = isAutoWhiteBalanceEnabled

        if !controlsLocked {
            lockControls()
        }

        activeEncodeWidth = selectedFormat.width
        activeEncodeHeight = selectedFormat.height
        activeFPS = selectedFPS.rawValue
        lockAutoModesForRecording()
        metalPipeline?.curveType = selectedCurve
        metalPipeline?.prewarm(width: selectedFormat.width, height: selectedFormat.height, curveType: selectedCurve)
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

        let interfaceOrientation: UIInterfaceOrientation
        if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene {
            interfaceOrientation = scene.interfaceOrientation
        } else {
            interfaceOrientation = .landscapeRight
        }

        videoWriter.onLowDiskSpace = { [weak self] in
            Task { @MainActor [weak self] in
                guard let self, self.isRecording else { return }
                self.stopRecording()
                self.errorMessage = "Storage Full (<500MB left) — Recording Stopped"
                self.refreshStatusLine()
            }
        }

        do {
            try videoWriter.start(
                outputURL: outputURL,
                width: selectedFormat.width,
                height: selectedFormat.height,
                bitrate: effectiveBitrate,
                targetFPS: selectedFPS.rawValue,
                includeAudio: includeAudio,
                curveType: selectedCurve,
                codec: selectedCodec,
                orientation: interfaceOrientation
            )
            isRecording = true
            isRecordingUnsafe = true
            processLock.withLock {
                nextDispatchSequenceID = 0
                nextOutputSequenceID = 0
                reorderBuffer.removeAll()
            }
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
            statusText = "REC · \(selectedFormat.shortLabel) · \(selectedFPS.label)fps · \(selectedCodec.displayName)"
            let capsLine = capabilities?.diagnosticSummary ?? ""
            print("[CameraViewModel] Recording start \(selectedFormat.width)x\(selectedFormat.height) CFR \(selectedFPS.label) \(selectedCodec.displayName) orientation=\(interfaceOrientation.rawValue)\n\(capsLine)")
        } catch {
            errorMessage = "Record failed: \(error.localizedDescription)"
            print("[CameraViewModel] Failed to start recording: \(error)")
            if !wasControlsLockedBeforeRecording {
                unlockControls()
                restoreAutoModesAfterRecording()
            }
        }
    }

    private func lockAutoModesForRecording() {
        guard let device = captureController.activeDevice else { return }

        // If white balance and focus are already locked (e.g. via lockControls()),
        // skip locking hardware configuration to avoid stalling the capture pipeline.
        let needsWB = isAutoWhiteBalanceEnabled
        let needsFocus = isAutoFocus
        guard needsWB || needsFocus else { return }

        do {
            try device.lockForConfiguration()

            if needsWB {
                freezeAutoWhiteBalance(on: device)
            }

            // Lock focus at current position without calling setFocusModeLocked with invalid lensPosition.
            if needsFocus {
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
            if needsWB {
                updateWBParams(from: device)
            }
        } catch {
            print("[CameraViewModel] lockAutoModesForRecording failed: \(error)")
        }
    }

    func stopRecording() {
        guard isRecording, !isSaving else { return }

        isRecording = false
        isSaving = true
        videoWriter.onLowDiskSpace = nil
        recordingTimer?.invalidate()
        recordingTimer = nil
        statusText = "Saving…"
        captureController.setRecordingMode(false)

        if !wasControlsLockedBeforeRecording {
            unlockControls()
            restoreAutoModesAfterRecording()
        }

        saveBackgroundTask = UIApplication.shared.beginBackgroundTask(withName: "OwLens-FinalizeRecording") { [weak self] in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.endSaveTask()
            }
        }

        let remaining: [(MTLTexture, CVPixelBuffer?, RawFrameData)] = processLock.withLock {
            let bufferedKeys = reorderBuffer.keys.sorted()
            var rem: [(MTLTexture, CVPixelBuffer?, RawFrameData)] = []
            for k in bufferedKeys {
                if let item = reorderBuffer.removeValue(forKey: k), let tex = item.0 {
                    rem.append((tex, item.1, item.2))
                }
            }
            return rem
        }

        for item in remaining {
            dispatchOrderedRecordedFrame(item.0, bgraPB: item.1, frameData: item.2, isTailFlush: true)
        }

        recordingQueue.async { [weak self] in
            guard let self else { return }
            self.isRecordingUnsafe = false
            self.videoWriter.finish { [weak self] url, error in
                self?.processQueue.async {
                    self?.metalPipeline?.trimMemory()
                }
                guard let url else {
                    Task { @MainActor [weak self] in
                        self?.statusText = "Save failed"
                        self?.errorMessage = error?.localizedDescription ?? "Recording ended with no frames"
                        self?.endSaveTask()
                    }
                    return
                }
                Task { @MainActor [weak self] in
                    await self?.saveFinishedRecording(at: url)
                }
            }
        }

        frameCount = Int(frameIndex)
        let realNote = "frames=\(frameCount) drops=\(droppedFrames) fps=\(selectedFPS.label) fmt=\(selectedFormat.shortLabel)"
        print("[CameraViewModel] Recording stopped \(realNote)")
        if let caps = capabilities {
            print("[CameraViewModel] Tester diagnostics:\n\(caps.diagnosticSummary)\n\(realNote)")
        }
    }

    @MainActor
    private func endSaveTask() {
        isSaving = false
        if saveBackgroundTask != .invalid {
            let task = saveBackgroundTask
            saveBackgroundTask = .invalid
            UIApplication.shared.endBackgroundTask(task)
        }
        if !wasControlsLockedBeforeRecording {
            unlockControls()
            restoreAutoModesAfterRecording()
        }
    }

    private func saveFinishedRecording(at url: URL) async {
        let tempGcsvURL = url.deletingPathExtension().appendingPathExtension("gcsv")

        // Validate the file before attempting any save
        guard await validateVideoFile(at: url) else {
            statusText = "Save failed"
            errorMessage = "Video file is corrupt or empty"
            print("[CameraViewModel] File validation failed for \(url.lastPathComponent)")
            try? FileManager.default.removeItem(at: url)
            try? FileManager.default.removeItem(at: tempGcsvURL)
            endSaveTask()
            return
        }

        switch selectedSaveDestination {
        case .photos:
            PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
                guard status == .authorized || status == .limited else {
                    Task { @MainActor in
                        self.errorMessage = "Photo library access denied"
                        self.refreshStatusLine()
                        try? FileManager.default.removeItem(at: url)
                        try? FileManager.default.removeItem(at: tempGcsvURL)
                        self.endSaveTask()
                    }
                    return
                }
                PHPhotoLibrary.shared().performChanges {
                    PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
                } completionHandler: { success, error in
                    Task { @MainActor in
                        if success {
                            self.statusText = "Saved to Photos"
                            self.showToast("Saved to Photos")
                        } else {
                            self.errorMessage = error?.localizedDescription ?? "Save failed"
                            self.statusText = "Save failed"
                        }
                        self.refreshStatusLine()
                        self.endSaveTask()
                    }
                    try? FileManager.default.removeItem(at: url)
                    try? FileManager.default.removeItem(at: tempGcsvURL)
                }
            }
        case .files:
            saveRecordingToChosenFilesFolder(url)
        }
    }

    /// Verify the output file exists, is non-empty, and contains a playable video track.
    private func validateVideoFile(at url: URL) async -> Bool {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? Int64,
              size > 1024 else {
            print("[CameraViewModel] Output file is missing or empty")
            return false
        }
        let asset = AVURLAsset(url: url)
        do {
            let tracks = try await asset.loadTracks(withMediaType: .video)
            guard let track = tracks.first else {
                print("[CameraViewModel] Output file has no video track")
                return false
            }
            let timeRange = try await track.load(.timeRange)
            return timeRange.duration.seconds > 0
        } catch {
            print("[CameraViewModel] Failed to validate video track: \(error)")
            return false
        }
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
        defer { endSaveTask() }
        let tempGcsvURL = url.deletingPathExtension().appendingPathExtension("gcsv")
        guard let folderURL = resolveFilesFolderURL() else {
            errorMessage = "Choose a Files folder before recording"
            statusText = "Save failed"
            try? FileManager.default.removeItem(at: url)
            try? FileManager.default.removeItem(at: tempGcsvURL)
            refreshStatusLine()
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

            // Also copy companion .gcsv file alongside the video into the chosen folder
            if FileManager.default.fileExists(atPath: tempGcsvURL.path) {
                let gcsvDestination = destination.deletingPathExtension().appendingPathExtension("gcsv")
                try? FileManager.default.removeItem(at: gcsvDestination)
                try? FileManager.default.copyItem(at: tempGcsvURL, to: gcsvDestination)
                try? FileManager.default.removeItem(at: tempGcsvURL)
                print("[CameraViewModel] Copied companion .gcsv to Files folder: \(gcsvDestination.path)")
            }

            statusText = "Saved to Files"
            showToast("Saved to Files")
            refreshStatusLine()
            print("[CameraViewModel] Saved recording to Files: \(destination.path)")
        } catch {
            try? FileManager.default.removeItem(at: url)
            try? FileManager.default.removeItem(at: tempGcsvURL)
            errorMessage = "Files save failed: \(error.localizedDescription)"
            statusText = "Save failed"
            refreshStatusLine()
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

        // Storage countdown & auto-stop check
        updateStorageEstimate()
        if remainingRecordSeconds <= 0 && isRecording {
            print("[CameraViewModel] Auto-stopping recording: storage limit reached.")
            stopRecording()
            errorMessage = "Storage limit reached — recording stopped and saved."
            refreshStatusLine()
        }
    }

    // MARK: - Storage Estimation & Monitoring

    func updateStorageEstimate() {
        let available = StorageEstimator.availableDiskSpaceBytes()
        availableStorageBytes = available
        let bytesPerSec = StorageEstimator.estimatedBytesPerSecond(
            format: selectedFormat,
            fps: selectedFPS,
            codec: selectedCodec,
            bitratePreset: selectedBitrate,
            includeAudio: selectedAudioSource.portUID != nil
        )
        let remainingSecs = StorageEstimator.estimatedRemainingSeconds(
            availableBytes: available,
            bytesPerSecond: bytesPerSec
        )
        remainingRecordSeconds = remainingSecs
        estimatedRecordTimeText = StorageEstimator.formatRemainingTime(seconds: remainingSecs)
    }

    private func startStorageMonitor() {
        storageRefreshTimer?.invalidate()
        storageRefreshTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, !self.isRecording else { return }
                self.updateStorageEstimate()
            }
        }
    }

    // MARK: - Frame Pipeline

    nonisolated private func handleIncomingFrame(_ frameData: RawFrameData) {
        // Never enqueue GPU work while backgrounded (IOGPUMetalError 00000006)
        guard isAppActive else { return }

        frameBuffer.enqueue(frameData)
        scheduleProcess()
    }

    nonisolated private func scheduleProcess() {
        let slotToUse: Int? = processLock.withLock {
            guard !freeSlots.isEmpty, frameBuffer.currentCount > 0 else { return nil }
            return freeSlots.popFirst()
        }
        guard let slot = slotToUse else { return }

        processQueue.async { [weak self] in
            self?.drainBuffer(slot: slot)
        }
    }

    nonisolated private func drainBuffer(slot: Int) {
        var frame: RawFrameData?
        if isRecordingUnsafe {
            // FIFO during recording: process every single captured frame in order without skipping
            frame = frameBuffer.dequeue()
            if var f = frame {
                processLock.withLock {
                    f.sequenceID = nextDispatchSequenceID
                    nextDispatchSequenceID &+= 1
                }
                frame = f
            }
        } else {
            // Preview only: drop older backlog frames to keep viewfinder latency minimal
            frame = frameBuffer.dequeueLatest()
        }
        guard let frame else {
            processLock.withLock {
                freeSlots.insert(slot)
            }
            return
        }

        // Check if another frame can begin encoding concurrently in another available slot
        scheduleProcess()

        processFrame(frame, slot: slot) { [weak self] in
            guard let self else { return }
            let remaining: Int = self.processLock.withLock {
                self.freeSlots.insert(slot)
                return self.frameBuffer.currentCount
            }
            if remaining > 0 {
                self.scheduleProcess()
            }
        }
    }

    nonisolated private func processFrame(_ frameData: RawFrameData, slot: Int = 0, completion: @escaping () -> Void) {
        guard isAppActive else { completion(); return }
        guard let pipeline = metalPipeline else { completion(); return }

        pipeline.bayerPattern = frameData.cfaPattern
        pipeline.blackLevel = frameData.blackLevel
        pipeline.whiteLevel = frameData.whiteLevel

        // LSC: calibrated override > frame default > neutral
        if let override = lscOverride {
            pipeline.lscParams = override.asLSCParams
        } else if frameData.lscCoefficients != .zero {
            pipeline.lscParams = Self.simd4ToLSCParams(frameData.lscCoefficients)
        } else {
            pipeline.lscParams = LSCParams(
                radialR: 0, radialG: 0, radialB: 0,
                radial4R: 0, radial4G: 0, radial4B: 0,
                azimuthR: 0, azimuthG: 0, azimuthB: 0
            )
        }
        pipeline.greenBalance = 1.0
        pipeline.iso = frameData.iso

        // Recompute noise coefficients from ISO-dependent profile
        if let prof = noiseProfileForISO {
            noiseCoeffs = prof.coefficients(at: frameData.iso)
        }
        pipeline.noiseShotCoeff = noiseCoeffs.shot
        pipeline.noiseReadCoeff = noiseCoeffs.read

        if let cm = frameData.colorMatrix { latestColorMatrix = cm }
        if let sm = frameData.sgamutMatrix { latestSGamutMatrix = sm }

        let cMatrix: simd_float3x3
        switch pipeline.curveType {
        case .linear:
            cMatrix = matrix_identity_float3x3
            pipeline.headroomScale = 1.0
        case .appleLog2:
            cMatrix = latestColorMatrix ?? WhiteBalanceParams.defaultSensorToBT2020
            pipeline.headroomScale = 1.0
        case .sLog3Approx:
            cMatrix = latestSGamutMatrix ?? WhiteBalanceParams.defaultSensorToSGamut3Cine
            pipeline.headroomScale = 1.0
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
            let isThermalElevated = (metalPipeline?.thermalState.rawValue ?? 0) >= ProcessInfo.ThermalState.serious.rawValue
            pipeline.processingQuality = isThermalElevated ? .previewFast : .recordQuality
            pipeline.process(frameData.pixelBuffer, encodeWidth: w, encodeHeight: h, encodeAsBGRA: true, slot: slot) { [weak self] framed, bgraPB in
                guard let self else { completion(); return }
                handleRecordedFrame(framed, bgraPB: bgraPB, frameData: frameData, completion: completion)
            }
        } else {
            // ── Preview (non-recording): lightweight path ──
            pipeline.processingQuality = .previewFast
            // Cap preview resolution to max 1920 (preserving exact aspect ratio) to avoid
            // rendering excessive resolution just for on-screen viewfinder rendering.
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
            pipeline.processPreviewOnly(frameData.pixelBuffer, encodeWidth: prevW, encodeHeight: prevH, slot: slot) { [weak self] framed in
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
                updateScopesIfNeeded(from: framed, pipeline: pipeline)

                // Submit texture directly to PreviewFeed for zero-allocation rendering on MTKView
                previewFeed.submit(texture: framed)

                let now = CACurrentMediaTime()
                let dropsChanged = (drops != lastReportedDrops)
                if now - lastMainActorSyncTime >= 0.10 || dropsChanged {
                    lastMainActorSyncTime = now
                    lastReportedDrops = drops
                    let frameDataBox = SendableBox(value: frameData)
                    let avgMs = pipeline.averageFrameTimeMs
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        self.syncLiveAutoValues(from: frameDataBox.value)
                        if self.cfaLabel != cfaName { self.cfaLabel = cfaName }
                        if self.droppedFrames != drops { self.droppedFrames = drops }
                        if abs(self.averageFrameTimeMs - avgMs) >= 0.1 { self.averageFrameTimeMs = avgMs }
                    }
                }
            }
        }
    }

    /// Updates live exposure scopes (histogram + waveform) throttled to ~10 Hz (4 Hz when recording) without blocking capture/render.
    nonisolated private func updateScopesIfNeeded(from framed: MTLTexture, pipeline: MetalPipeline?) {
        guard showScopesUnsafe else { return }
        let now = CACurrentMediaTime()
        // During active recording, throttle from 10 Hz (100ms) to 4 Hz (250ms) to reduce GPU downsampling passes
        // and CPU texture memory readbacks (getBytes) while VideoToolbox encodes real-time frames.
        let interval: Double = isRecordingUnsafe ? 0.25 : 0.10
        guard now - lastScopeUpdateTime >= interval else { return }
        lastScopeUpdateTime = now
        pipeline?.makeScopeData(from: framed) { [weak self] scope in
            guard let self, let scope else { return }
            Task { @MainActor [weak self] in
                self?.scopeMonitor.update(scope)
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
        // Immediately release the Metal pipeline slot so the next frame can begin GPU work concurrently!
        completion()

        guard isRecordingUnsafe else {
            if let framed {
                dispatchOrderedRecordedFrame(framed, bgraPB: bgraPB, frameData: frameData)
            }
            return
        }

        let readyFrames: [(MTLTexture, CVPixelBuffer?, RawFrameData)] = processLock.withLock {
            if let framed {
                reorderBuffer[frameData.sequenceID] = (framed, bgraPB, frameData)
            } else {
                // If this frame failed, insert a placeholder or advance
                if frameData.sequenceID == nextOutputSequenceID {
                    nextOutputSequenceID &+= 1
                } else if frameData.sequenceID > nextOutputSequenceID {
                    // Mark as failed placeholder so sequencer doesn't stall when reaching it
                    reorderBuffer[frameData.sequenceID] = (nil, nil, frameData)
                }
            }

            // Gap recovery: if a sequence ID was lost before reaching handleRecordedFrame (e.g. dropped in ring buffer),
            // prevent reorderBuffer from stalling. Since only 3 slots exist, waiting for 6 frames caused 200ms freezes.
            if let minKey = reorderBuffer.keys.min() {
                if minKey > nextOutputSequenceID && (reorderBuffer.count >= 2 || freeSlots.count == 3) {
                    nextOutputSequenceID = minKey
                }
            }

            var ready: [(MTLTexture, CVPixelBuffer?, RawFrameData)] = []
            while let next = reorderBuffer.removeValue(forKey: nextOutputSequenceID) {
                if let tex = next.0 {
                    ready.append((tex, next.1, next.2))
                }
                nextOutputSequenceID &+= 1
            }
            return ready
        }

        for item in readyFrames {
            dispatchOrderedRecordedFrame(item.0, bgraPB: item.1, frameData: item.2)
        }
    }

    nonisolated private func dispatchOrderedRecordedFrame(
        _ framed: MTLTexture?,
        bgraPB: CVPixelBuffer?,
        frameData: RawFrameData,
        isTailFlush: Bool = false
    ) {
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

        updateScopesIfNeeded(from: framed, pipeline: metalPipeline)

        if let bgraPB, (isRecordingUnsafe || isTailFlush) {
            let timestamp = frameData.timestamp
            recordingQueue.async { [weak self] in
                guard let self else { return }
                if isTailFlush || self.isRecordingUnsafe {
                    if self.videoWriter.appendFrame(pixelBuffer: bgraPB, captureTime: timestamp) {
                        self.frameIndex += 1
                    }
                }
            }
        }

        // Submit texture directly to PreviewFeed for zero-allocation rendering on MTKView
        previewFeed.submit(texture: framed)

        let now = CACurrentMediaTime()
        let dropsChanged = (drops != lastReportedDrops)
        if now - lastMainActorSyncTime >= 0.10 || dropsChanged {
            lastMainActorSyncTime = now
            lastReportedDrops = drops
            let frameDataBox = SendableBox(value: frameData)
            let avgMs = metalPipeline?.averageFrameTimeMs ?? 0.0
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.syncLiveAutoValues(from: frameDataBox.value)
                if self.cfaLabel != cfaName { self.cfaLabel = cfaName }
                if self.droppedFrames != drops { self.droppedFrames = drops }
                if abs(self.averageFrameTimeMs - avgMs) >= 0.1 { self.averageFrameTimeMs = avgMs }
            }
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
        guard isAutoWhiteBalanceEnabled || isAutoWhiteBalanceAdjusting else { return }
        guard let device = captureController.activeDevice else { return }

        if isAutoWhiteBalanceEnabled {
            // Fast path: use pre-smoothed Kelvin and Tint calculated on the capture queue
            // without performing expensive nonlinear polynomial root-finding on the main thread!
            if let temp = frameData.kelvin, let tintVal = frameData.tint {
                let clampedTemp = max(2000, min(10000, temp))
                if abs(wbKelvin - clampedTemp) >= 25 {
                    wbKelvin = clampedTemp
                    wbStopIndex = ExposureStops.nearestIndex(in: wbStops, to: clampedTemp)
                }
                if abs(wbTint - tintVal) >= 1.0 {
                    wbTint = tintVal
                }
            } else if let gains = frameData.whiteBalanceGains {
                let clamped = clampWhiteBalanceGains(gains, for: device)
                let temperatureAndTint = device.temperatureAndTintValues(for: clamped)
                let temp = max(2000, min(10000, temperatureAndTint.temperature))
                if abs(wbKelvin - temp) >= 25 {
                    wbKelvin = temp
                    wbStopIndex = ExposureStops.nearestIndex(in: wbStops, to: temp)
                }
                if abs(wbTint - temperatureAndTint.tint) >= 1.0 {
                    wbTint = temperatureAndTint.tint
                }
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

