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

/// High-frequency isolated state for lossless sensor-crop zoom.
/// Decoupled from `CameraViewModel` to eliminate full HUD and ContentView invalidations
/// during 60/120 Hz gesture tracking.
@MainActor
final class ZoomMonitor: ObservableObject {
    @Published private(set) var zoomFactor: CGFloat = 1.0
    @Published var isWheelActive: Bool = false
    @Published var isSwitcherExpanded: Bool = false

    weak var metalPipeline: MetalPipeline?
    var onZoomChanged: (@Sendable (Float) -> Void)?
    private var zoomDismissTask: Task<Void, Never>?

    init(metalPipeline: MetalPipeline? = nil) {
        self.metalPipeline = metalPipeline
    }

    /// Sets the zoom factor.
    /// Updates the GPU Metal pipeline at zero latency on every call,
    /// and deadband-filters SwiftUI view updates to eliminate main runloop congestion.
    func setZoom(_ factor: CGFloat) {
        let clamped = max(1.0, min(2.0, factor))
        // 1. Direct zero-latency hardware update
        metalPipeline?.zoomFactor = Float(clamped)
        onZoomChanged?(Float(clamped))

        // 2. Deadband check: publish only on measurable change
        guard abs(zoomFactor - clamped) >= 0.002 else { return }
        zoomFactor = clamped
    }

    func resetZoom() {
        setZoom(1.0)
    }

    func showWheelAndSwitcher() {
        cancelDismiss()
        if !isWheelActive {
            isWheelActive = true
        }
        if !isSwitcherExpanded {
            isSwitcherExpanded = true
        }
    }

    func hideWheelAndSwitcher() {
        cancelDismiss()
        if isWheelActive {
            isWheelActive = false
        }
        if isSwitcherExpanded {
            isSwitcherExpanded = false
        }
    }

    func toggleSwitcher() {
        Haptics.selection()
        if isSwitcherExpanded {
            withAnimation(.spring(response: 0.28, dampingFraction: 0.84)) {
                hideWheelAndSwitcher()
            }
        } else {
            withAnimation(.spring(response: 0.28, dampingFraction: 0.84)) {
                showWheelAndSwitcher()
            }
            scheduleDismiss(delaySeconds: 3.5)
        }
    }

    func showWheel() {
        showWheelAndSwitcher()
    }

    func hideWheel() {
        hideWheelAndSwitcher()
    }

    func cancelDismiss() {
        zoomDismissTask?.cancel()
        zoomDismissTask = nil
    }

    func scheduleDismiss(delaySeconds: Double = 3.5) {
        zoomDismissTask?.cancel()
        guard !UIAccessibility.isVoiceOverRunning else { return }
        zoomDismissTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(delaySeconds * 1_000_000_000))
            guard !Task.isCancelled else { return }
            withAnimation(.spring(response: 0.28, dampingFraction: 0.84)) {
                self.isWheelActive = false
                self.isSwitcherExpanded = false
            }
        }
    }

    func effectiveZoomString(baseMultiplier: Float) -> String {
        let effective = Float(zoomFactor) * baseMultiplier
        if effective >= 10.0 {
            return String(format: "%.0f×", effective)
        } else if abs(effective.rounded() - effective) < 0.05 {
            return String(format: "%.0f×", effective)
        } else {
            return String(format: "%.1f×", effective)
        }
    }

    func equivalentFocalLengthString(deviceType: AVCaptureDevice.DeviceType?) -> String {
        let baseFocal: Float
        switch deviceType {
        case .builtInUltraWideCamera: baseFocal = 13.0
        case .builtInTelephotoCamera: baseFocal = 77.0
        default: baseFocal = 24.0
        }
        let focal = Int((baseFocal * Float(zoomFactor)).rounded())
        return "\(focal)mm"
    }
}

/// Central view model — CaptureController → RawFrameBuffer → MetalPipeline → preview + VideoWriter.
@MainActor
final class CameraViewModel: NSObject, ObservableObject, UIDocumentPickerDelegate {
    // MARK: - Published State

    /// High-speed direct preview feed to MTKView. Eliminates per-frame SwiftUI re-renders.
    let previewFeed = PreviewFeed()
    @Published var isRecording = false
    @Published var isSaving = false
    @Published var controlsLocked = false
    private var wasControlsLockedBeforeRecording = false
    private var wasAutoFocusBeforeRecording = true
    private var wasAutoWBBeforeRecording = false
    @Published var thermalState: ProcessInfo.ThermalState = .nominal
    private var latestLandscapeOrientation: UIInterfaceOrientation?
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
            if selectedFormat == .openGate && (selectedCodec == .proRes422 || selectedCodec == .proRes422HQ) {
                selectedCodec = .hevc
            }
            updateStorageEstimate()
            refreshStatusLine()
        }
    }
    @Published var selectedFPS: CaptureFrameRate = .fps24 {
        didSet {
            guard !controlsLocked, !isRecording else { return }
            captureController.lockCaptureTo24FPS()
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
            if (selectedCodec == .proRes422 || selectedCodec == .proRes422HQ) && selectedFormat == .openGate {
                // ProRes hardware encoding is strictly supported for 16:9 on iOS
                selectedFormat = .hd169
            }
            updateStorageEstimate()
            refreshStatusLine()
        }
    }
    /// User-adjustable denoise strength (0.0–1.0). Default = 1.0.
    @Published var denoiseStrength: Float = 1.0 {
        didSet {
            metalPipeline?.denoiseStrength = denoiseStrength
            UserDefaults.standard.set(denoiseStrength, forKey: "OwLens.DenoiseStrength")
            print("[CameraViewModel] Denoise strength: \(denoiseStrength)")
        }
    }
    @Published private(set) var selectedSaveDestination: VideoSaveDestination = .photos
    @Published var audioSources: [AudioSourceOption] = []
    @Published var selectedAudioSource: AudioSourceOption = .none {
        didSet {
            isAudioMutedUnsafe = (selectedAudioSource.portUID == nil)
            guard !controlsLocked, !isRecording else { return }
            // Skip no-op reassign (same port and matches hardware) — avoids hang loops
            guard oldValue.portUID != selectedAudioSource.portUID ||
                  captureController.currentAudioPortUID != selectedAudioSource.portUID else { return }
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

    // MARK: - Lossless Sensor-Crop Zoom (1.0x ... 2.0x)

    /// Isolated high-frequency zoom monitor. Eliminates full HUD and ContentView invalidations during dragging.
    let zoomMonitor = ZoomMonitor()

    /// Current sensor-crop zoom factor (1.0 ... 2.0 lossless range).
    var zoomFactor: CGFloat {
        zoomMonitor.zoomFactor
    }

    /// Whether the circular zoom wheel UI is actively expanded.
    var isZoomWheelActive: Bool {
        get { zoomMonitor.isWheelActive }
        set { zoomMonitor.isWheelActive = newValue }
    }

    private var pinchStartZoom: CGFloat = 1.0

    /// Formatted effective zoom label (e.g. "1.0×", "1.4×", "2.0×").
    var effectiveZoomString: String {
        zoomMonitor.effectiveZoomString(baseMultiplier: selectedLens?.baseMultiplier ?? 1.0)
    }

    /// Approximate 35mm equivalent focal length string (e.g. "24mm", "35mm", "48mm").
    var equivalentFocalLengthString: String {
        zoomMonitor.equivalentFocalLengthString(deviceType: selectedLens?.deviceType)
    }

    /// Sets the zoom factor smoothly with clamping to 1.0...2.0.
    func setZoomFactor(_ factor: CGFloat) {
        zoomMonitor.setZoom(factor)
    }

    /// Resets zoom back to 1.0 (un-cropped full sensor).
    func resetZoom() {
        zoomMonitor.resetZoom()
    }

    /// Opens or touches the zoom wheel with auto-dismiss timeout.
    func touchZoomWheel() {
        zoomMonitor.showWheel()
        zoomMonitor.scheduleDismiss()
    }

    func scheduleZoomWheelDismiss(delaySeconds: Double = 2.5) {
        zoomMonitor.scheduleDismiss(delaySeconds: delaySeconds)
    }

    func handlePinchChanged(scale: CGFloat) {
        zoomMonitor.cancelDismiss()
        if !zoomMonitor.isWheelActive {
            pinchStartZoom = zoomFactor
            withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                zoomMonitor.showWheel()
            }
        }
        let target = pinchStartZoom * scale
        zoomMonitor.setZoom(target)
    }

    func handlePinchEnded() {
        pinchStartZoom = zoomFactor
        zoomMonitor.scheduleDismiss()
    }

    @Published var showGrid = false
    @Published var showClipping = false {
        didSet {
            let enabled = showClipping
            clippingPreviewEnabled.withLock { $0 = enabled }
        }
    }
    nonisolated private let clippingPreviewEnabled = OSAllocatedUnfairLock(initialState: false)
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
            metalPipeline?.isAutoWBEnabled = isAutoWhiteBalanceEnabled
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
    /// Eight 12 MP Bayer frames bound the backlog to about a third of a second.
    nonisolated(unsafe) private let frameBuffer = RawFrameBuffer(capacity: 8)

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
    private let exposureControlQueue = DispatchQueue(label: "com.owlens.exposure", qos: .userInitiated)
    nonisolated private let processLock = OSAllocatedUnfairLock()
    nonisolated(unsafe) private var freeSlots: Set<Int> = [0, 1, 2]

    private struct RecordedOutput: @unchecked Sendable {
        let pixels: CVPixelBuffer?
        let timestamp: CMTime
        let releaseSlot: () -> Void
    }

    // Each sequence number owns a slot until its output reaches the writer queue.
    nonisolated private let submittedRecordingWork = DispatchGroup()
    nonisolated(unsafe) private var nextDispatchSequenceID: UInt64 = 0
    nonisolated(unsafe) private var nextOutputSequenceID: UInt64 = 0
    nonisolated(unsafe) private var pendingVideoWrites = 0
    nonisolated(unsafe) private var writerQueueDrops = 0
    nonisolated(unsafe) private var gpuFrameCount = 0
    nonisolated(unsafe) private var gpuMilliseconds = 0.0
    nonisolated(unsafe) private var gpuMaxMilliseconds = 0.0
    nonisolated(unsafe) private var gpuFrameFailures = 0
    nonisolated(unsafe) private var reorderBuffer: [UInt64: RecordedOutput] = [:]

    nonisolated(unsafe) private var activeEncodeWidth = 2016
    nonisolated(unsafe) private var activeEncodeHeight = 1512
    nonisolated(unsafe) private var activeFPS: Double = 30
    nonisolated(unsafe) private var activeZoomFactorUnsafe: Float = 1.0
    nonisolated(unsafe) private var isRecordingUnsafe = false
    nonisolated(unsafe) private var activeRecordingSessionID: UInt64 = 0
    nonisolated private let recordingFlip180 = OSAllocatedUnfairLock(initialState: false)
    nonisolated(unsafe) private var isAudioMutedUnsafe = false
    nonisolated(unsafe) private var showScopesUnsafe = true
    private struct PreviewTelemetry {
        var lastScope: CFTimeInterval = 0
        var lastHUD: CFTimeInterval = 0
    }
    nonisolated private let previewTelemetry = OSAllocatedUnfairLock(initialState: PreviewTelemetry())
    nonisolated(unsafe) var isAppActive = true
    /// Measured LSC override from device calibration (set once at setup).
    nonisolated(unsafe) private var lscOverride: LSCCoefficients?

    /// Measured noise coefficients from device calibration (set once at setup).
    nonisolated(unsafe) private var noiseCoeffs: (shot: Float, read: Float) = (0.012, 0.0004)
    /// Noise profile for per-ISO coefficient lookup (nil on unknown device).
    nonisolated(unsafe) private var noiseProfileForISO: NoiseProfile?
    /// Latest calibrated color matrices extracted from DNG metadata.
    nonisolated(unsafe) private var latestColorMatrix: simd_float3x3?
    nonisolated(unsafe) private var latestAppleWideGamutMatrix: simd_float3x3?

    enum ControlPanel: String, Identifiable {
        case exposure, iso, shutter, wb, focus, format, bitrate, denoise, mic, lens, save, logCurve
        var id: String { rawValue }
    }

    // MARK: - Init

    override init() {
        let pipeline = MetalPipeline()
        metalPipeline = pipeline
        zoomMonitor.metalPipeline = pipeline
        super.init()
        zoomMonitor.onZoomChanged = { [weak self] z in
            self?.activeZoomFactorUnsafe = z
        }
        captureController.sharedCommandQueue = pipeline?.commandQueue
        startClockAndBatteryMonitoring()
#if DEBUG
        // Automated unit tests and benchmarks are invoked via test suites rather than on camera startup.
#endif
        loadFilesFolderBookmark()
        if let savedDenoise = UserDefaults.standard.object(forKey: "OwLens.DenoiseStrength") as? Float {
            denoiseStrength = savedDenoise
        } else {
            denoiseStrength = 1.0
        }
        metalPipeline?.denoiseStrength = denoiseStrength
        metalPipeline?.zoomFactor = Float(zoomFactor)
        metalPipeline?.curveType = selectedCurve
        metalPipeline?.thermalState = ProcessInfo.processInfo.thermalState
        activeEncodeWidth = selectedFormat.width
        activeEncodeHeight = selectedFormat.height
        activeFPS = selectedFPS.rawValue
        selectedBitrate = selectedFormat.suggestedBitratePreset
        // Open Gate + HEVC is the universal default. ProRes remains an explicit
        // option only on hardware that supports it, rather than changing framing
        // and codec behind the user's back at launch.
        selectedCodec = .hevc

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

        // Refresh mic list and handle disconnects / connects when route changes
        NotificationCenter.default.publisher(for: AVAudioSession.routeChangeNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] note in
                guard let self else { return }
                let reasonValue = note.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt
                let reason = reasonValue.flatMap { AVAudioSession.RouteChangeReason(rawValue: $0) }
                self.handleAudioRouteChange(reason: reason)
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: AVCaptureDevice.wasConnectedNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.handleAudioRouteChange(reason: .newDeviceAvailable)
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: AVCaptureDevice.wasDisconnectedNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.handleAudioRouteChange(reason: .oldDeviceUnavailable)
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
            let audio = SendableBox(value: sample)
            self.processLock.withLock {
                let sessionID = self.activeRecordingSessionID
                guard self.isRecordingUnsafe, sessionID > 0 else { return }
                self.recordingQueue.async { [weak self] in
                    guard let self, self.processLock.withLock({ self.activeRecordingSessionID == sessionID }) else { return }
                    _ = self.videoWriter.appendAudio(sampleBuffer: audio.value)
                }
            }
        }

        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        NotificationCenter.default.publisher(for: UIDevice.orientationDidChangeNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.rememberLandscapeOrientation()
            }
            .store(in: &cancellables)
        rememberLandscapeOrientation()

        updateStorageEstimate()
        startStorageMonitor()
    }

    var viewfinderOrientation: UIInterfaceOrientation {
        let activeScene = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first { $0.activationState == .foregroundActive }
        if let orientation = activeScene?.effectiveGeometry.interfaceOrientation, orientation.isLandscape {
            return orientation
        }
        return .landscapeRight
    }

    private func rememberLandscapeOrientation() {
        // The display can lag behind the physical device or be rotation-locked.
        // Flat and portrait readings preserve the last useful landscape reading.
        switch UIDevice.current.orientation {
        case .landscapeRight: latestLandscapeOrientation = .landscapeLeft
        case .landscapeLeft: latestLandscapeOrientation = .landscapeRight
        default: break
        }
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
            captureController.lockCaptureTo24FPS()
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
        // 150 Mbps is the quality default for every supported hardware tier.
        // It is the highest validated real-time HEVC rate in this RAW pipeline.
        selectedBitrate = .mbps150
        selectedCodec = .hevc
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
            captureController.lockCaptureTo24FPS()
            refreshAudioSources()
            captureController.startSession()
            applyManualExposureAndWB()
            refreshStatusLine()
            isCameraReady = true
            cameraReadyTime = Date()
            UIApplication.shared.isIdleTimerDisabled = true
            Task.detached(priority: .utility) { Self.cleanStaleTemporaryRecordings() }
            print("[CameraViewModel] Camera session started · \(caps.marketingName) · lenses=\(availableLenses.map(\.shortLabel))")
        } catch {
            errorMessage = error.localizedDescription
            statusText = "Camera failed"
            isCameraReady = false
            cameraReadyTime = nil
            // If session fails due to no Bayer after all, treat as unsupported
            if (error as NSError).code == 4 {
                isDeviceUnsupportedForLog = true
            }
            print("[CameraViewModel] Failed to configure camera: \(error.localizedDescription)")
        }
    }

    func teardownCamera() {
        if isRecording { stopRecording() }
        isCameraReady = false
        cameraReadyTime = nil
        processLock.withLock { isRecordingUnsafe = false }
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
            zoomMonitor.hideWheelAndSwitcher()
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

        var maxAmp: Float = 0
        if let formatDesc = CMSampleBufferGetFormatDescription(sample),
           let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc)?.pointee {
            let isFloat = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0
            if isFloat {
                let sampleCount = totalLength / MemoryLayout<Float32>.size
                guard sampleCount > 0 else { return }
                let step = max(1, sampleCount / 48)
                dataPtr.withMemoryRebound(to: Float32.self, capacity: sampleCount) { ptr in
                    var i = 0
                    while i < sampleCount {
                        let val = abs(ptr[i])
                        if val > maxAmp { maxAmp = val }
                        i += step
                    }
                }
            } else {
                let sampleCount = totalLength / MemoryLayout<Int16>.size
                guard sampleCount > 0 else { return }
                let step = max(1, sampleCount / 48)
                dataPtr.withMemoryRebound(to: Int16.self, capacity: sampleCount) { ptr in
                    var i = 0
                    while i < sampleCount {
                        let val = abs(Float(ptr[i])) / 32768.0
                        if val > maxAmp { maxAmp = val }
                        i += step
                    }
                }
            }
        } else {
            let sampleCount = totalLength / MemoryLayout<Int16>.size
            guard sampleCount > 0 else { return }
            let step = max(1, sampleCount / 48)
            dataPtr.withMemoryRebound(to: Int16.self, capacity: sampleCount) { ptr in
                var i = 0
                while i < sampleCount {
                    let val = abs(Float(ptr[i])) / 32768.0
                    if val > maxAmp { maxAmp = val }
                    i += step
                }
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

    private func handleAudioRouteChange(reason: AVAudioSession.RouteChangeReason?) {
        refreshAudioSources(reason: reason)
        // Delayed secondary check to account for hardware USB descriptor enumeration latency
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { [weak self] in
            self?.refreshAudioSources(reason: reason)
        }
    }

    func refreshAudioSources(reason: AVAudioSession.RouteChangeReason? = nil) {
        let previousIDs = Set(audioSources.map(\.id))
        let sources = captureController.availableAudioSources()
        audioSources = sources

        // Case 1: A new device was connected (or detected)
        if reason == .newDeviceAvailable {
            // Find newly attached device (not previously seen, and not built-in)
            if let newlyConnected = sources.first(where: { !previousIDs.contains($0.id) && $0.portUID != nil && $0.name != "Built-in Mic" })
                ?? sources.first(where: { $0.portUID != nil && $0.name != "Built-in Mic" && $0.id != "Built-In Microphone" }) {
                if selectedAudioSource.id != newlyConnected.id {
                    print("[CameraViewModel] New external mic attached: \(newlyConnected.name)")
                    selectedAudioSource = newlyConnected
                    showToast("Connected: \(newlyConnected.name)")
                    return
                }
            }
        }

        // Case 2: A device was disconnected
        if reason == .oldDeviceUnavailable || !sources.contains(where: { $0.portUID == selectedAudioSource.portUID }) {
            // Previously selected device is gone → fallback to built-in mic
            if let builtIn = sources.first(where: { $0.name == "Built-in Mic" || $0.id == "Built-In Microphone" || $0.portUID?.contains("Built-In") == true }) ?? sources.first {
                if selectedAudioSource.id != builtIn.id {
                    print("[CameraViewModel] External mic disconnected → fallback to: \(builtIn.name)")
                    selectedAudioSource = builtIn
                    showToast("Mic: \(builtIn.name)")
                    return
                }
            }
        }

        // Case 3: Check if active hardware route changed to an external device
        let currentUID = captureController.currentAudioPortUID
        if let currentUID, let active = sources.first(where: { $0.portUID == currentUID }) {
            if active.name != "Built-in Mic" && active.id != "Built-In Microphone" && selectedAudioSource.portUID != active.portUID {
                print("[CameraViewModel] Active route is external mic → \(active.name)")
                selectedAudioSource = active
                showToast("Connected: \(active.name)")
                return
            }
            if selectedAudioSource.id != active.id || selectedAudioSource.name != active.name {
                selectedAudioSource = active
            }
            return
        }

        // Case 4: Keep existing selection if still valid
        if let match = sources.first(where: { $0.portUID == selectedAudioSource.portUID && $0.portUID != nil }) {
            if match.id != selectedAudioSource.id || match.name != selectedAudioSource.name {
                selectedAudioSource = match
            }
            return
        }

        // Case 5: Initial startup fallback
        if let builtIn = sources.first(where: { $0.name == "Built-in Mic" || $0.id == "Built-In Microphone" || $0.portUID?.contains("Built-In") == true }) {
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
        resetZoom()
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
        let targetID = device.uniqueID
        let target = SendableBox(value: device)

        // Preserve the order of exposure updates and discard work for an old lens.
        exposureControlQueue.async { [weak self] in
            guard let self, self.captureController.currentLensUniqueID == targetID else { return }
            let device = target.value
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
                    let r = wbGains.redGain.isNaN ? 1.0 : max(1.0, min(maxGain, wbGains.redGain))
                    let g = wbGains.greenGain.isNaN ? 1.0 : max(1.0, min(maxGain, wbGains.greenGain))
                    let b = wbGains.blueGain.isNaN ? 1.0 : max(1.0, min(maxGain, wbGains.blueGain))
                    let clamped = AVCaptureDevice.WhiteBalanceGains(
                        redGain: r,
                        greenGain: g,
                        blueGain: b
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
                guard activeDevice.uniqueID == targetID else { return }
                self.updateWBParams(from: activeDevice, explicitGains: finalGains)
            }
        }
    }

    /// Freeze active auto white balance directly on hardware at its current values,
    /// ensuring that locking controls or starting recording prevents color drift.
    private func freezeAutoWhiteBalance(on device: AVCaptureDevice) {
        guard isAutoWhiteBalanceEnabled else { return }
        let currentGains = device.deviceWhiteBalanceGains
        let clamped = clampWhiteBalanceGains(currentGains, for: device)
        let tempTint = device.temperatureAndTintValues(for: clamped)
        wbKelvin = max(2000, min(10000, tempTint.temperature))
        wbStopIndex = ExposureStops.nearestIndex(in: wbStops, to: wbKelvin)
        wbTint = max(-50.0, min(50.0, tempTint.tint))
        isAutoWhiteBalanceEnabled = false
        isAutoWhiteBalanceAdjusting = false
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
        if wasAutoFocusBeforeRecording {
            isAutoFocus = true
            isFocusLocked = false
            captureController.restoreContinuousAutoFocus()
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
        let r = gains.redGain.isNaN ? 1.0 : max(1.0, min(maxGain, gains.redGain))
        let g = gains.greenGain.isNaN ? 1.0 : max(1.0, min(maxGain, gains.greenGain))
        let b = gains.blueGain.isNaN ? 1.0 : max(1.0, min(maxGain, gains.blueGain))
        return AVCaptureDevice.WhiteBalanceGains(
            redGain: r,
            greenGain: g,
            blueGain: b
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
            cMatrix = latestAppleWideGamutMatrix ?? WhiteBalanceParams.defaultSensorToAppleWideGamut
        case .sLog3Approx:
            // HEVC is written with BT.2020 Y′CbCr coefficients and BT.2020
            // signaling. Feeding it S-Gamut3.Cine RGB would be a color-space
            // mismatch, so this remains a BT.2020 + S-Log3-transfer profile.
            cMatrix = latestColorMatrix ?? WhiteBalanceParams.defaultSensorToBT2020
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

    private var cameraReadyTime: Date?

    func toggleRecording() {
        guard isAppActive, isCameraReady, !isDeviceUnsupportedForLog, !isSaving,
              !isSwitchingLens, !isSwitchingMic else { return }
        // Guard against any spurious triggers during camera spin-up and initial stabilization
        guard let readyTime = cameraReadyTime, Date().timeIntervalSince(readyTime) > 1.5 else {
            print("[CameraViewModel] Ignored shutter toggle during camera stabilization window")
            return
        }
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
        guard isAppActive, isCameraReady, !isRecording, !isSaving, !isSwitchingLens, !isSwitchingMic else { return }

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
        wasAutoFocusBeforeRecording = isAutoFocus

        if !controlsLocked {
            lockControls()
        }

        // ProRes hardware encoding is strictly supported for standard 16:9 broadcast dimensions (1080p).
        // If Open Gate (2016x1512) is active, safely use HEVC 10-bit so recording never fails.
        var effectiveCodec = selectedCodec
        if (effectiveCodec == .proRes422 || effectiveCodec == .proRes422HQ) && selectedFormat == .openGate {
            print("[CameraViewModel] ProRes hardware encoding does not support Open Gate (2016x1512) — recording in HEVC 10-bit")
            effectiveCodec = .hevc
        }

        activeEncodeWidth = selectedFormat.width
        activeEncodeHeight = selectedFormat.height
        activeFPS = selectedFPS.rawValue
        lockAutoModesForRecording()
        metalPipeline?.curveType = selectedCurve
        metalPipeline?.denoiseStrength = denoiseStrength
        metalPipeline?.use422Encoding = (effectiveCodec == .proRes422 || effectiveCodec == .proRes422HQ)
        metalPipeline?.zoomFactor = Float(zoomFactor)
        frameBuffer.flush()
        processLock.withLock {
            nextDispatchSequenceID = 0
            nextOutputSequenceID = 0
            reorderBuffer.removeAll(keepingCapacity: true)
            writerQueueDrops = 0
            gpuFrameCount = 0
            gpuMilliseconds = 0
            gpuMaxMilliseconds = 0
            gpuFrameFailures = 0
        }
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

        rememberLandscapeOrientation()
        let recordOrientation = latestLandscapeOrientation ?? viewfinderOrientation
        recordingFlip180.withLock { $0 = recordOrientation == .landscapeLeft }

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
                captureClock: captureController.session.synchronizationClock,
                includeAudio: includeAudio,
                curveType: selectedCurve,
                codec: effectiveCodec,
                orientation: recordOrientation
            )
            // VideoWriter may reject ProRes for this exact device/configuration and
            // transparently fall back to HEVC. Keep Metal's output pixel format in
            // lock-step with the writer; 4:2:2 frames fed to a 4:2:0 HEVC adaptor
            // otherwise fail with VideoToolbox's generic "cannot encode" error.
            let writerCodec = videoWriter.activeCodec
            metalPipeline?.use422Encoding = (writerCodec == .proRes422 || writerCodec == .proRes422HQ)
            let width = selectedFormat.width, height = selectedFormat.height, curve = selectedCurve
            // Texture pools belong to the same serial queue that submits GPU work.
            processQueue.sync { metalPipeline?.prewarm(width: width, height: height, curveType: curve) }
            frameIndex = 0
            captureController.setRecordingMode(true)
            let sessionID = captureController.activeRecordingSessionID
            processLock.withLock {
                activeRecordingSessionID = sessionID
                isRecordingUnsafe = true
            }
            isRecording = true
            frameCount = 0
            recordingStartTime = recordingDate
            recordingDuration = "00:00"
            activePanel = nil

            recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    self?.updateRecordingDuration()
                }
            }
            statusText = "REC · \(selectedFormat.shortLabel) · \(selectedFPS.label)fps · \(writerCodec.displayName)"
            let capsLine = capabilities?.diagnosticSummary ?? ""
            print("[CameraViewModel] Recording start \(selectedFormat.width)x\(selectedFormat.height) CFR \(selectedFPS.label) \(writerCodec.displayName) orientation=\(recordOrientation.rawValue)\n\(capsLine)")
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
        processLock.withLock { isRecordingUnsafe = false }
        videoWriter.onLowDiskSpace = nil
        recordingTimer?.invalidate()
        recordingTimer = nil
        statusText = "Saving…"
        captureController.setRecordingMode(false)

        saveBackgroundTask = UIApplication.shared.beginBackgroundTask(withName: "OwLens-FinalizeRecording") { [weak self] in
            Task { @MainActor [weak self] in
                self?.releaseSaveBackgroundTime()
            }
        }

        // Take a barrier on the submission queue before registering for
        // completion so a frame already entering Metal isn't missed.
        processQueue.async { [self] in
            submittedRecordingWork.notify(queue: recordingQueue) {
                self.finishSubmittedRecording()
            }
        }
    }

    nonisolated private func finishSubmittedRecording() {
        let gpuStats = processLock.withLock {
            activeRecordingSessionID = 0
            let mean = gpuMilliseconds / Double(max(1, gpuFrameCount))
            return "gpuFrames=\(gpuFrameCount) gpuFailures=\(gpuFrameFailures) gpuMsMean=\(mean) gpuMsMax=\(gpuMaxMilliseconds) writerQueueDrops=\(writerQueueDrops)"
        }
        let count = Int(frameIndex)
        let rawDrops = frameBuffer.droppedCount
        print("[CameraViewModel] Recording stopped frames=\(count) queueDrops=\(rawDrops) \(gpuStats)")
        videoWriter.finish { [self] url, error in
            processQueue.async { [weak self] in
                self?.metalPipeline?.trimMemory()
            }
            Task { @MainActor in
                self.frameCount = count
                if let url {
                    await self.saveFinishedRecording(at: url)
                } else {
                    self.statusText = "Save failed"
                    self.errorMessage = error?.localizedDescription ?? "Recording ended with no frames"
                    self.endSaveTask()
                }
            }
        }
    }

    @MainActor
    private func endSaveTask() {
        isSaving = false
        releaseSaveBackgroundTime()
        if !wasControlsLockedBeforeRecording {
            unlockControls()
            restoreAutoModesAfterRecording()
        }
    }

    private func releaseSaveBackgroundTime() {
        if saveBackgroundTask != .invalid {
            let task = saveBackgroundTask
            saveBackgroundTask = .invalid
            UIApplication.shared.endBackgroundTask(task)
        }
    }

    private func saveFinishedRecording(at url: URL) async {
        let tempGcsvURL = url.deletingPathExtension().appendingPathExtension("gcsv")

        // Validate the file before attempting any save
        guard await validateVideoFile(at: url) else {
            print("[CameraViewModel] File validation failed for \(url.lastPathComponent)")
            await recoverUnexportedRecording(url, reason: "Couldn’t verify the video")
            endSaveTask()
            return
        }

        switch selectedSaveDestination {
        case .photos:
            let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
            guard status == .authorized || status == .limited else {
                await recoverUnexportedRecording(url, reason: "Photos access denied")
                endSaveTask()
                return
            }
            do {
                try await PHPhotoLibrary.shared().performChanges {
                    PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
                }
                try? FileManager.default.removeItem(at: url)
                try? FileManager.default.removeItem(at: tempGcsvURL)
                statusText = "Saved to Photos"
                showToast("Saved to Photos")
                refreshStatusLine()
            } catch {
                await recoverUnexportedRecording(url, reason: "Photos save failed")
            }
            endSaveTask()
        case .files:
            await saveRecordingToChosenFilesFolder(url)
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

    private func saveRecordingToChosenFilesFolder(_ url: URL) async {
        defer { endSaveTask() }
        guard let folderURL = resolveFilesFolderURL() else {
            await recoverUnexportedRecording(url, reason: "The selected Files folder is unavailable")
            return
        }
        do {
            let destination = try await Task.detached(priority: .utility) {
                let shouldStop = folderURL.startAccessingSecurityScopedResource()
                defer { if shouldStop { folderURL.stopAccessingSecurityScopedResource() } }
                let files = FileManager.default
                let destination = Self.uniqueDestinationURL(in: folderURL, preferredName: url.lastPathComponent)
                try files.copyItem(at: url, to: destination)
                let sidecar = url.deletingPathExtension().appendingPathExtension("gcsv")
                let sidecarTarget = destination.deletingPathExtension().appendingPathExtension("gcsv")
                if (try? files.copyItem(at: sidecar, to: sidecarTarget)) != nil {
                    try? files.removeItem(at: sidecar)
                }
                try? files.removeItem(at: url)
                return destination
            }.value
            statusText = "Saved to Files"
            showToast("Saved to Files")
            refreshStatusLine()
            print("[CameraViewModel] Saved recording to Files: \(destination.path)")
        } catch {
            print("[CameraViewModel] Files save failed: \(error.localizedDescription)")
            await recoverUnexportedRecording(url, reason: "Files save failed")
        }
    }

    private func recoverUnexportedRecording(_ url: URL, reason: String) async {
        let recovered = await Task.detached(priority: .utility) { Self.recoverRecording(at: url) }.value
        statusText = recovered ? "Saved locally" : "Save failed"
        errorMessage = recovered
            ? "\(reason). Find your clip in Files → On My iPhone → OwLens → Recovered Recordings."
            : "\(reason). OwLens couldn’t move the recording to Files."
    }

    nonisolated static func recoverRecording(at url: URL, documents: URL? = nil) -> Bool {
        let files = FileManager.default
        guard let documents = documents ?? files.urls(for: .documentDirectory, in: .userDomainMask).first else { return false }
        let folder = documents.appendingPathComponent("Recovered Recordings", isDirectory: true)
        do {
            try files.createDirectory(at: folder, withIntermediateDirectories: true)
            let destination = uniqueDestinationURL(in: folder, preferredName: url.lastPathComponent)
            try files.moveItem(at: url, to: destination)
            let sidecar = url.deletingPathExtension().appendingPathExtension("gcsv")
            let sidecarTarget = destination.deletingPathExtension().appendingPathExtension("gcsv")
            try? files.moveItem(at: sidecar, to: sidecarTarget)
            return true
        } catch {
            print("[CameraViewModel] Recording recovery failed: \(error.localizedDescription)")
            return false
        }
    }

    nonisolated private static func uniqueDestinationURL(in folderURL: URL, preferredName: String) -> URL {
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
        let recording = processLock.withLock { isRecordingUnsafe }
        guard let frame = recording ? frameBuffer.dequeue() : frameBuffer.dequeueLatest() else {
            releaseProcessingSlot(slot)
            return
        }

        // Check if another frame can begin encoding concurrently in another available slot
        scheduleProcess()

        processFrame(frame, slot: slot) { [weak self] in
            self?.releaseProcessingSlot(slot)
        }
    }

    nonisolated private func releaseProcessingSlot(_ slot: Int) {
        _ = processLock.withLock { freeSlots.insert(slot) }
        scheduleProcess()
    }

    nonisolated private func processFrame(_ frameData: RawFrameData, slot: Int = 0, completion: @escaping () -> Void) {
        guard isAppActive, let pipeline = metalPipeline else {
            completion()
            return
        }

        pipeline.bayerPattern = frameData.cfaPattern
        pipeline.blackLevel = frameData.blackLevel
        pipeline.whiteLevel = frameData.whiteLevel
        pipeline.zoomFactor = activeZoomFactorUnsafe

        pipeline.updateGainMap(lscOverride == nil ? frameData.gainMap : nil)
        if let override = lscOverride {
            pipeline.lscParams = override.asLSCParams
        } else if let gm = frameData.gainMap {
            var gmParams = Self.simd4ToLSCParams(frameData.lscCoefficients)
            gmParams.useGainMap = 1
            gmParams.gainMapScaleOffset = SIMD4<Float>(gm.uvScaleX, gm.uvScaleY, gm.uvOffsetX, gm.uvOffsetY)
            gmParams.outputScale = 1.0
            pipeline.lscParams = gmParams
        } else {
            pipeline.lscParams = Self.simd4ToLSCParams(frameData.lscCoefficients)
        }
        // Apple RAW can keep sensor DN nearly constant as ISO rises and express
        // the remaining exposure in DNG BaselineExposure (EV). Apply that
        // per-frame rendering instruction in scene-linear light before the log OETF.
        pipeline.lscParams.outputScale *= exp2(frameData.baselineExposureStops)
        pipeline.greenBalance = 1.0
        pipeline.iso = frameData.iso

        // Recompute noise coefficients from ISO-dependent profile
        if let prof = noiseProfileForISO {
            noiseCoeffs = prof.coefficients(at: frameData.iso)
        }
        pipeline.noiseShotCoeff = noiseCoeffs.shot
        pipeline.noiseReadCoeff = noiseCoeffs.read

        if let cm = frameData.colorMatrix { latestColorMatrix = cm }
        if let appleWide = frameData.appleWideGamutMatrix { latestAppleWideGamutMatrix = appleWide }

        let cMatrix: simd_float3x3
        switch pipeline.curveType {
        case .linear:
            cMatrix = matrix_identity_float3x3
        case .appleLog2:
            cMatrix = latestAppleWideGamutMatrix ?? WhiteBalanceParams.defaultSensorToAppleWideGamut
        case .sLog3Approx:
            cMatrix = latestColorMatrix ?? WhiteBalanceParams.defaultSensorToBT2020
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

        let sourceSessionID = frameData.recordingSessionID
        let isRecordingFrame = processLock.withLock {
            isRecordingUnsafe && sourceSessionID != 0 && sourceSessionID == activeRecordingSessionID
        }

        if isRecordingFrame {
            // ── Recording ──
            let isThermalElevated = (metalPipeline?.thermalState.rawValue ?? 0) >= ProcessInfo.ThermalState.serious.rawValue
            pipeline.processingQuality = isThermalElevated ? .previewFast : .recordQuality
            let submittedAt = CACurrentMediaTime()
            let flip180 = recordingFlip180.withLock { $0 }
            let sequence = nextDispatchSequenceID
            nextDispatchSequenceID &+= 1
            let group = submittedRecordingWork
            group.enter()
            pipeline.process(frameData.pixelBuffer, encodeWidth: w, encodeHeight: h, encodeAsBGRA: true,
                             slot: slot, flip180: flip180) { [weak self] log, pixels in
                guard let self else { completion(); group.leave(); return }
                let milliseconds = (CACurrentMediaTime() - submittedAt) * 1000
                self.completeRecordedFrame(sequence, pixels: pixels, timestamp: frameData.timestamp,
                                           sessionID: sourceSessionID, milliseconds: milliseconds) {
                    completion()
                    group.leave()
                }
                if let log {
                    self.publishFramePreview(PreviewFeed.Frame(log: log, flipped180: flip180, timestamp: frameData.timestamp),
                                             frameData: frameData)
                }
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
            // The sensor is 4032×3024, while the viewfinder is at most 1920×1440.
            // Reduce Bayer data before demosaicing on every lens. Keep the native
            // sensor path when the clipping overlay needs individual sensel peaks.
            let fastPreview = !clippingPreviewEnabled.withLock { $0 }
            pipeline.processPreviewOnly(frameData.pixelBuffer, encodeWidth: prevW, encodeHeight: prevH,
                                        slot: slot, preferBinnedPreview: fastPreview) { [weak self] log in
                defer { completion() }
                guard let self, let log else { return }
                self.publishFramePreview(PreviewFeed.Frame(log: log, flipped180: false, timestamp: frameData.timestamp),
                                         frameData: frameData)
            }
        }
    }

    nonisolated private func completeRecordedFrame(
        _ sequence: UInt64, pixels: CVPixelBuffer?, timestamp: CMTime,
        sessionID: UInt64, milliseconds: Double, releaseSlot: @escaping () -> Void
    ) {
        let output = RecordedOutput(pixels: pixels, timestamp: timestamp, releaseSlot: releaseSlot)
        let released: [() -> Void] = processLock.withLock {
            guard sessionID == activeRecordingSessionID else { return [releaseSlot] }
            gpuFrameCount += 1
            gpuMilliseconds += milliseconds
            gpuMaxMilliseconds = max(gpuMaxMilliseconds, milliseconds)
            if output.pixels == nil { gpuFrameFailures += 1 }

            reorderBuffer[sequence] = output
            var slots: [() -> Void] = []
            while let next = reorderBuffer.removeValue(forKey: nextOutputSequenceID) {
                if let pixels = next.pixels {
                    enqueueVideoFrame(pixels, timestamp: next.timestamp, sessionID: sessionID)
                }
                nextOutputSequenceID &+= 1
                slots.append(next.releaseSlot)
            }
            return slots
        }
        for release in released { release() }
    }

    /// Called with processLock held to preserve the order of writer submissions.
    nonisolated private func enqueueVideoFrame(_ buffer: CVPixelBuffer, timestamp: CMTime, sessionID: UInt64) {
        guard pendingVideoWrites < 8 else {
            writerQueueDrops += 1
            return
        }
        pendingVideoWrites += 1
        let pixels = SendableBox(value: buffer)
        recordingQueue.async { [weak self] in
            guard let self else { return }
            defer { self.processLock.withLock { self.pendingVideoWrites -= 1 } }
            guard self.processLock.withLock({ self.activeRecordingSessionID == sessionID }) else { return }
            if self.videoWriter.appendFrame(pixelBuffer: pixels.value, captureTime: timestamp) {
                self.frameIndex += 1
            }
        }
    }

    nonisolated private func publishFramePreview(_ frame: PreviewFeed.Frame, frameData: RawFrameData) {
        guard isAppActive, previewFeed.submit(frame) else { return }

        let now = CACurrentMediaTime()
        let scopeInterval: Double = processLock.withLock { isRecordingUnsafe } ? 0.25 : 0.10
        let (updateScope, updateHUD) = previewTelemetry.withLock { state in
            let scope = showScopesUnsafe && now - state.lastScope >= scopeInterval
            let hud = now - state.lastHUD >= 0.10
            if scope { state.lastScope = now }
            if hud { state.lastHUD = now }
            return (scope, hud)
        }
        if updateScope {
            metalPipeline?.makeScopeData(from: frame.log) { [weak self] scope in
                guard let scope else { return }
                Task { @MainActor [weak self] in self?.scopeMonitor.update(scope) }
            }
        }
        guard updateHUD else { return }
        let drops = frameBuffer.droppedCount
        let cfaNames = ["RGGB", "GRBG", "GBRG", "BGGR"]
        let cfa = Int(frameData.cfaPattern)
        let cfaName = cfaNames.indices.contains(cfa) ? cfaNames[cfa] : "?\(cfa)"
        let kelvin = frameData.kelvin
        let tint = frameData.tint
        let gains = SendableBox(value: frameData.whiteBalanceGains)
        let average = metalPipeline?.averageFrameTimeMs ?? 0
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.syncLiveAutoValues(kelvin: kelvin, tint: tint, gains: gains.value)
            if self.cfaLabel != cfaName { self.cfaLabel = cfaName }
            if self.droppedFrames != drops { self.droppedFrames = drops }
            if abs(self.averageFrameTimeMs - average) >= 0.1 { self.averageFrameTimeMs = average }
        }
    }

    /// Convert fixed lens coefficients to the shader's equal-RGB gains.
    nonisolated private static func simd4ToLSCParams(_ coeffs: SIMD4<Float>) -> LSCParams {
        let r2 = coeffs[0]
        let r4 = coeffs[1]
        let cornerBase = 1 + r2
        let cornerGain = 1 + (cornerBase * cornerBase - 1 + r4) / (1 + coeffs.z)
        // Halfway between the previous Balanced and Edge Lift presets in stops.
        let outputScale = max(exp2(-0.325), 1 / sqrt(sqrt(max(cornerGain, 1))))
        return LSCParams(
            radialR: r2,
            radialG: r2,
            radialB: r2,
            radial4R: r4,
            radial4G: r4,
            radial4B: r4,
            azimuthR: 0, azimuthG: 0, azimuthB: 0,
            radialRolloff: coeffs.z,
            outputScale: outputScale
        )
    }

    private func syncLiveAutoValues(kelvin: Float?, tint: Float?, gains: AVCaptureDevice.WhiteBalanceGains?) {
        guard isAutoWhiteBalanceEnabled || isAutoWhiteBalanceAdjusting else { return }
        guard let device = captureController.activeDevice else { return }

        if isAutoWhiteBalanceEnabled {
            // Fast path: use pre-smoothed Kelvin and Tint calculated on the capture queue
            // without performing expensive nonlinear polynomial root-finding on the main thread!
            if let temp = kelvin, let tintVal = tint {
                let clampedTemp = max(2000, min(10000, temp))
                if abs(wbKelvin - clampedTemp) >= 25 {
                    wbKelvin = clampedTemp
                    wbStopIndex = ExposureStops.nearestIndex(in: wbStops, to: clampedTemp)
                }
                let clampedTint = max(-50.0, min(50.0, tintVal))
                if abs(wbTint - clampedTint) >= 1.0 {
                    wbTint = clampedTint
                }
            } else if let gains {
                let clamped = clampWhiteBalanceGains(gains, for: device)
                let temperatureAndTint = device.temperatureAndTintValues(for: clamped)
                let temp = max(2000, min(10000, temperatureAndTint.temperature))
                if abs(wbKelvin - temp) >= 25 {
                    wbKelvin = temp
                    wbStopIndex = ExposureStops.nearestIndex(in: wbStops, to: temp)
                }
                let clampedTint = max(-50.0, min(50.0, temperatureAndTint.tint))
                if abs(wbTint - clampedTint) >= 1.0 {
                    wbTint = clampedTint
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

