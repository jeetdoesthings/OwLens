import AVFoundation
import AudioToolbox
import Combine
import CoreVideo
import QuartzCore

/// Metadata extracted from each RAW photo capture.
/// `pixelBuffer` is always an **owned copy** so the capture pipeline can start the next still immediately.
struct RawFrameData {
    let pixelBuffer: CVPixelBuffer
    let whiteBalanceGains: AVCaptureDevice.WhiteBalanceGains?
    let cfaPattern: Int32
    let blackLevel: Float
    let whiteLevel: Float
    let pixelFormat: OSType
    let lscCoefficients: SIMD4<Float>
    let iso: Float
}

final class CaptureController: NSObject, ObservableObject {
    let session = AVCaptureSession()
    private let photoOutput = AVCapturePhotoOutput()
    var isShutterSoundSuppressionSupported: Bool {
        if #available(iOS 18.0, *) {
            return photoOutput.isShutterSoundSuppressionSupported
        }
        return false
    }
    private let audioOutput = AVCaptureAudioDataOutput()
    private var device: AVCaptureDevice?
    private var videoInput: AVCaptureDeviceInput?
    private var audioInput: AVCaptureDeviceInput?
    private var captureTimer: DispatchSourceTimer?
    private let captureQueue = DispatchQueue(label: "raw.capture.queue", qos: .userInteractive)
    private let audioQueue = DispatchQueue(label: "raw.audio.queue", qos: .userInitiated)

    private let captureLock = NSLock()
    /// Outstanding capturePhoto calls (RAW usually 1; responsive may allow more).
    private var inFlightCaptures = 0
    private var maxInFlight = 1

    var onRawFrameData: ((RawFrameData) -> Void)?
    var onAudioSample: ((CMSampleBuffer) -> Void)?

    private var rawPixelFormat: OSType = 0
    private var formatCFAPattern: Int32 = 0
    private var targetFPS: Double = 24
    private var minFrameInterval: Double = 1.0 / 24.0
    private var lastCaptureStart: CFTimeInterval = 0
    private var selectedAudioPortUID: String?
    private var isReconfiguringAudio = false

    private var cachedBlackLevel: Float?
    private var cachedWhiteLevel: Float?
    private var cachedISO: Float?

    // MARK: - Session Configuration

    func configureSession() throws {
        // Activate .playAndRecord audio session early — this suppresses the
        // system shutter sound that otherwise fires 24×/sec during RAW burst.
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playAndRecord, mode: .videoRecording, options: [.defaultToSpeaker, .mixWithOthers])
            try audioSession.setActive(true)
        } catch {
            print("[CaptureController] Audio session shutter-suppress failed (non-fatal): \(error)")
        }

        // Build graph under begin/commit, then resolve Bayer *after* commit.
        // availableRawPhotoPixelFormatTypes is often empty *inside* beginConfiguration.
        session.beginConfiguration()
        session.sessionPreset = .photo

        let lenses = Self.discoverBackLenses()
        // Prefer Wide (most reliable Bayer), then Tele, then Ultra Wide — single-lens only.
        let ordered = lenses.sorted { a, b in
            Self.singleLensSortKey(a.deviceType) < Self.singleLensSortKey(b.deviceType)
        }

        guard let firstLens = ordered.first,
              let camera = Self.requireSingleLensDevice(uniqueID: firstLens.uniqueID) else {
            session.commitConfiguration()
            throw NSError(domain: "RawLogCam", code: 1, userInfo: [NSLocalizedDescriptionKey: "No single-lens back camera found"])
        }
        self.device = camera
        Self.logSelectedCamera(camera, context: "configureSession")

        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input) else {
            session.commitConfiguration()
            throw NSError(domain: "RawLogCam", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot add camera input"])
        }
        session.addInput(input)
        self.videoInput = input

        guard session.canAddOutput(photoOutput) else {
            session.commitConfiguration()
            throw NSError(domain: "RawLogCam", code: 3, userInfo: [NSLocalizedDescriptionKey: "Cannot add photo output"])
        }
        session.addOutput(photoOutput)

        photoOutput.maxPhotoQualityPrioritization = .speed
        maxInFlight = 1
 
        // Mic + audio output
        if let defaultMic = AVCaptureDevice.default(for: .audio) {
            try attachAudioDevice(defaultMic)
        }
        audioOutput.setSampleBufferDelegate(self, queue: audioQueue)
        if session.canAddOutput(audioOutput) {
            session.addOutput(audioOutput)
        }

        session.commitConfiguration()
        if #available(iOS 18.0, *) {
            print("[CaptureController] iOS 18+ isShutterSoundSuppressionSupported: \(photoOutput.isShutterSoundSuppressionSupported)")
        }
        // ── configuration committed: RAW format list is now meaningful ──

        // Pure .photo preset — do NOT force activeFormat (that was zeroing Bayer).
        try applyDefaultCameraModes(on: camera)
        lockSensorToTargetFPS(on: camera, fps: targetFPS)

        try resolveBayerRAWOrThrow()

        // Burst helpers only after Bayer is confirmed
        enableBurstHelpersIfSafe()
        prepareRAWPhotoResources()

        if let builtIn = AVAudioSession.sharedInstance().availableInputs?.first(where: {
            $0.portType == .builtInMic
        }) {
            selectedAudioPortUID = builtIn.uid
            try? AVAudioSession.sharedInstance().setPreferredInput(builtIn)
        }

        print("[RawLogCam] Ready. singleLensTypes=\(lenses.map { Self.deviceTypeLabel($0.deviceType) }.joined(separator: ", ")) Bayer=\(fourCC(rawPixelFormat)) inFlight=\(maxInFlight)")
    }

    /// Lock sensor cadence without changing activeFormat (keeps photo/RAW path).
    private func lockSensorToTargetFPS(on camera: AVCaptureDevice, fps: Double) {
        let rate = min(max(fps, 1), 30)
        do {
            try camera.lockForConfiguration()
            defer { camera.unlockForConfiguration() }
            let ranges = camera.activeFormat.videoSupportedFrameRateRanges
            guard ranges.contains(where: { $0.minFrameRate <= rate && rate <= $0.maxFrameRate }) else {
                print("[RawLogCam] Sensor FPS \(rate) not in range; leaving default")
                return
            }
            let duration = CMTime(value: 1, timescale: CMTimeScale(rate))
            camera.activeVideoMinFrameDuration = duration
            camera.activeVideoMaxFrameDuration = duration
            let dims = CMVideoFormatDescriptionGetDimensions(camera.activeFormat.formatDescription)
            print("[RawLogCam] Sensor \(dims.width)x\(dims.height) locked @ \(rate)fps (preset .photo)")
        } catch {
            print("[RawLogCam] lockSensorToTargetFPS: \(error)")
        }
    }

    private func enableBurstHelpersIfSafe() {
        let before = bayerFormatsAvailable()
        guard !before.isEmpty else { return }

        if photoOutput.isZeroShutterLagSupported {
            photoOutput.isZeroShutterLagEnabled = true
            if bayerFormatsAvailable().isEmpty {
                photoOutput.isZeroShutterLagEnabled = false
                print("[RawLogCam] ZSL disabled — would remove Bayer RAW")
            }
        }
        if photoOutput.isResponsiveCaptureSupported {
            photoOutput.isResponsiveCaptureEnabled = true
            if bayerFormatsAvailable().isEmpty {
                photoOutput.isResponsiveCaptureEnabled = false
                maxInFlight = 1
                print("[RawLogCam] Responsive capture disabled — would remove Bayer RAW")
            } else {
                maxInFlight = 2
            }
        }
        if photoOutput.isFastCapturePrioritizationSupported {
            photoOutput.isFastCapturePrioritizationEnabled = true
            if bayerFormatsAvailable().isEmpty {
                photoOutput.isFastCapturePrioritizationEnabled = false
                print("[RawLogCam] Fast capture prioritization disabled — Bayer safety")
            }
        }
    }

    private func bayerFormatsAvailable() -> [OSType] {
        photoOutput.availableRawPhotoPixelFormatTypes.filter {
            AVCapturePhotoOutput.isBayerRAWPixelFormat($0)
        }
    }

    /// Must run **after** `session.commitConfiguration()` with photo input+output attached.
    private func resolveBayerRAWOrThrow() throws {
        var allRaw = photoOutput.availableRawPhotoPixelFormatTypes
        print("[RawLogCam] Available RAW formats: \(allRaw.map { fourCC($0) })")

        var bayer = bayerFormatsAvailable()
        print("[RawLogCam] Bayer formats: \(bayer.map { fourCC($0) })")

        if bayer.isEmpty {
            print("[RawLogCam] Bayer empty — soft reset (.photo, no burst opts)")
            session.beginConfiguration()
            session.sessionPreset = .photo
            photoOutput.maxPhotoQualityPrioritization = .speed
            if #available(iOS 17.0, *) {
                if photoOutput.isZeroShutterLagSupported { photoOutput.isZeroShutterLagEnabled = false }
                if photoOutput.isResponsiveCaptureSupported { photoOutput.isResponsiveCaptureEnabled = false }
                if photoOutput.isFastCapturePrioritizationSupported { photoOutput.isFastCapturePrioritizationEnabled = false }
            }
            maxInFlight = 1
            session.commitConfiguration()

            allRaw = photoOutput.availableRawPhotoPixelFormatTypes
            bayer = bayerFormatsAvailable()
            print("[RawLogCam] Bayer after soft reset: \(bayer.map { fourCC($0) }) all=\(allRaw.map { fourCC($0) })")
        }

        // Try other back lenses if needed
        if bayer.isEmpty {
            for lens in Self.discoverBackLenses() {
                guard lens.uniqueID != device?.uniqueID,
                      let cam = AVCaptureDevice(uniqueID: lens.uniqueID) else { continue }
                print("[RawLogCam] Trying lens for Bayer: \(lens.shortLabel)")
                if switchVideoDeviceSync(to: cam) {
                    bayer = bayerFormatsAvailable()
                    if !bayer.isEmpty {
                        print("[RawLogCam] Bayer found on \(lens.shortLabel)")
                        break
                    }
                }
            }
        }

        // Last resort: if *any* RAW exists and Bayer filter fails, still reject ProRAW-only
        // but log everything for debugging
        if bayer.isEmpty && !allRaw.isEmpty {
            print("[RawLogCam] WARNING: RAW present but none classified as Bayer — \(allRaw.map { fourCC($0) })")
        }

        guard let rawFormat = Self.preferredBayerFormat(from: bayer) else {
            throw NSError(
                domain: "RawLogCam",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "No Bayer RAW available. Use Photo preset + Wide camera. Reboot if stuck."]
            )
        }
        self.rawPixelFormat = rawFormat
        self.formatCFAPattern = Self.cfaPattern(forBayerFormat: rawFormat)
        print("[RawLogCam] Selected Bayer RAW: \(fourCC(rawFormat)) CFA=\(formatCFAPattern)")
    }

    /// Synchronous lens swap for setup recovery (capture queue not required).
    private func switchVideoDeviceSync(to camera: AVCaptureDevice) -> Bool {
        guard Self.isAllowedSingleLens(camera.deviceType), !camera.isVirtualDevice else {
            print("[RawLogCam] switchVideoDeviceSync rejected multi-cam/virtual \(Self.deviceTypeLabel(camera.deviceType))")
            return false
        }
        Self.logSelectedCamera(camera, context: "switchVideoDeviceSync")
        do {
            let newInput = try AVCaptureDeviceInput(device: camera)
            session.beginConfiguration()
            if let old = videoInput {
                session.removeInput(old)
            }
            guard session.canAddInput(newInput) else {
                if let old = videoInput { session.addInput(old) }
                session.commitConfiguration()
                return false
            }
            session.addInput(newInput)
            videoInput = newInput
            device = camera
            session.sessionPreset = .photo
            session.commitConfiguration()
            try? applyDefaultCameraModes(on: camera)
            return true
        } catch {
            session.commitConfiguration()
            return false
        }
    }

    /// Warm capture pipeline for repeated RAW stills (WWDC: setPreparedPhotoSettingsArray).
    private func prepareRAWPhotoResources() {
        let prepared = (0..<3).map { _ in makeRAWPhotoSettings() }
        photoOutput.setPreparedPhotoSettingsArray(prepared) { preparedOK, error in
            if let error {
                print("[RawLogCam] prepare RAW settings failed: \(error)")
            } else {
                print("[RawLogCam] RAW photo resources prepared=\(preparedOK)")
            }
        }
    }

    private func makeRAWPhotoSettings() -> AVCapturePhotoSettings {
        let settings = AVCapturePhotoSettings(rawPixelFormatType: rawPixelFormat)
        settings.flashMode = .off
        settings.photoQualityPrioritization = .speed
        if #available(iOS 16.0, *) {
            settings.maxPhotoDimensions = photoOutput.maxPhotoDimensions
        }
        if #available(iOS 18.0, *) {
            if photoOutput.isShutterSoundSuppressionSupported {
                settings.isShutterSoundSuppressionEnabled = true
            }
        }
        return settings
    }

    private func applyDefaultCameraModes(on camera: AVCaptureDevice) throws {
        try camera.lockForConfiguration()
        if camera.isFocusModeSupported(.autoFocus) {
            camera.focusMode = .autoFocus
        }
        if camera.isExposureModeSupported(.continuousAutoExposure) {
            camera.exposureMode = .continuousAutoExposure
        }
        if camera.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
            camera.whiteBalanceMode = .continuousAutoWhiteBalance
        }
        // Slightly higher ISO / shorter shutter can reduce sensor readout wait in auto mode
        camera.unlockForConfiguration()
    }

    // MARK: - Lenses (single physical cameras ONLY)

    /// Device types that are a **single** physical back camera.
    /// Virtual multi-cam (Dual / DualWide / Triple / Continuity) do **not** expose Bayer RAW
    /// the way this pipeline needs — stock Camera app uses those for seamless zoom.
    static let allowedSingleLensTypes: [AVCaptureDevice.DeviceType] = [
        .builtInWideAngleCamera,
        .builtInTelephotoCamera,
        .builtInUltraWideCamera
    ]

    /// Explicitly rejected virtual / multi-camera types (must never be session input).
    static let forbiddenMultiCamTypes: Set<AVCaptureDevice.DeviceType> = {
        var s: Set<AVCaptureDevice.DeviceType> = [
            .builtInDualCamera,
            .builtInDualWideCamera,
            .builtInTripleCamera,
            .builtInTrueDepthCamera
        ]
        if #available(iOS 17.0, *) {
            s.insert(.continuityCamera)
        }
        return s
    }()

    static func isAllowedSingleLens(_ type: AVCaptureDevice.DeviceType) -> Bool {
        allowedSingleLensTypes.contains(type) && !forbiddenMultiCamTypes.contains(type)
    }

    static func deviceTypeLabel(_ type: AVCaptureDevice.DeviceType) -> String {
        switch type {
        case .builtInWideAngleCamera: return "builtInWideAngleCamera"
        case .builtInTelephotoCamera: return "builtInTelephotoCamera"
        case .builtInUltraWideCamera: return "builtInUltraWideCamera"
        case .builtInDualCamera: return "builtInDualCamera(FORBIDDEN)"
        case .builtInDualWideCamera: return "builtInDualWideCamera(FORBIDDEN)"
        case .builtInTripleCamera: return "builtInTripleCamera(FORBIDDEN)"
        case .builtInTrueDepthCamera: return "builtInTrueDepthCamera(FORBIDDEN)"
        default: return type.rawValue
        }
    }

    static func singleLensSortKey(_ type: AVCaptureDevice.DeviceType) -> Int {
        switch type {
        case .builtInWideAngleCamera: return 0
        case .builtInTelephotoCamera: return 1
        case .builtInUltraWideCamera: return 2
        default: return 99
        }
    }

    static func logSelectedCamera(_ device: AVCaptureDevice, context: String) {
        let type = device.deviceType
        let ok = isAllowedSingleLens(type)
        let dims = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
        print("""
        [RawLogCam] CAMERA SELECT context=\(context)
          deviceType=\(deviceTypeLabel(type))
          singleLensOK=\(ok)
          localizedName=\(device.localizedName)
          uniqueID=\(device.uniqueID)
          position=\(device.position.rawValue)
          activeFormat=\(dims.width)x\(dims.height)
          isVirtualDevice=\(device.isVirtualDevice)
        """)
        if !ok || device.isVirtualDevice {
            print("[RawLogCam] ERROR: multi-cam / virtual device selected — Bayer RAW path will fail")
        }
    }

    /// Resolve device by uniqueID and reject multi-cam / virtual.
    static func requireSingleLensDevice(uniqueID: String) -> AVCaptureDevice? {
        guard let device = AVCaptureDevice(uniqueID: uniqueID) else { return nil }
        guard isAllowedSingleLens(device.deviceType), !device.isVirtualDevice else {
            print("[RawLogCam] Rejected camera uniqueID=\(uniqueID) type=\(deviceTypeLabel(device.deviceType)) virtual=\(device.isVirtualDevice)")
            return nil
        }
        return device
    }

    /// Discover **only** single-lens back cameras (never Dual/Triple virtual devices).
    static func discoverBackLenses() -> [LensOption] {
        // Discovery session restricted to single physical types — never Dual/Triple.
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: allowedSingleLensTypes,
            mediaType: .video,
            position: .back
        )

        // Get the wide lens's minimum back FOV zoom factor for relative magnification
        let wideFOV: Float? = discovery.devices
            .first(where: { $0.deviceType == .builtInWideAngleCamera && !$0.isVirtualDevice })
            .flatMap { Float($0.minAvailableVideoZoomFactor) }

        var options: [LensOption] = []
        var seen = Set<String>()

        for device in discovery.devices {
            guard isAllowedSingleLens(device.deviceType) else {
                print("[RawLogCam] discover: skip forbidden type \(deviceTypeLabel(device.deviceType))")
                continue
            }
            guard !device.isVirtualDevice else {
                print("[RawLogCam] discover: skip virtual \(device.localizedName)")
                continue
            }
            guard !seen.contains(device.uniqueID) else { continue }
            seen.insert(device.uniqueID)

            let (name, short): (String, String)
            switch device.deviceType {
            case .builtInUltraWideCamera: name = "Ultra Wide"; short = "0.5×"
            case .builtInWideAngleCamera: name = "Wide"; short = "1×"
            case .builtInTelephotoCamera:
                // Derive actual multiplier relative to wide (2×, 3×, 5× etc.)
                // Uses the telephoto's zoom factor relative to the wide lens
                let multiplier: Int
                if let wFOV = wideFOV, wFOV > 0 {
                    // Telephoto equivalent focal length vs wide
                    // Most iPhones: 12 Pro = 2×, 13 Pro = 3×, 15 Pro Max = 5×
                    let ratio = Float(device.minAvailableVideoZoomFactor) / wFOV
                    if ratio > 1.1 {
                        multiplier = Int(ratio.rounded())
                    } else {
                        // Fallback: use known device characteristics
                        multiplier = 2
                    }
                } else {
                    multiplier = 2
                }
                name = "Telephoto"; short = "\(multiplier)×"
            default:
                print("[RawLogCam] discover: unexpected type \(deviceTypeLabel(device.deviceType)) — skipped")
                continue
            }

            options.append(LensOption(
                id: device.uniqueID,
                name: name,
                shortLabel: short,
                deviceType: device.deviceType,
                uniqueID: device.uniqueID
            ))
            print("[RawLogCam] discover: + \(short) \(deviceTypeLabel(device.deviceType)) id=\(device.uniqueID)")
        }

        // Fallback if DiscoverySession empty: explicit defaults per type (still single-lens only)
        if options.isEmpty {
            print("[RawLogCam] discover: DiscoverySession empty — trying defaultDevice per single-lens type")
            for type in allowedSingleLensTypes {
                guard let device = AVCaptureDevice.default(type, for: .video, position: .back),
                      !device.isVirtualDevice else { continue }
                let name: String
                let short: String
                switch type {
                case .builtInUltraWideCamera: name = "Ultra Wide"; short = "0.5×"
                case .builtInWideAngleCamera: name = "Wide"; short = "1×"
                case .builtInTelephotoCamera: name = "Telephoto"; short = "2×"
                default: continue
                }
                options.append(LensOption(
                    id: device.uniqueID,
                    name: name,
                    shortLabel: short,
                    deviceType: type,
                    uniqueID: device.uniqueID
                ))
                print("[RawLogCam] discover fallback: + \(short) \(deviceTypeLabel(type))")
            }
        }

        // Sort from smallest to biggest focal length: UW (0.5×) → Wide (1×) → Tele (2×+)
        let sortOrder: [AVCaptureDevice.DeviceType: Int] = [
            .builtInUltraWideCamera: 0,
            .builtInWideAngleCamera: 1,
            .builtInTelephotoCamera: 2
        ]
        options.sort { (sortOrder[$0.deviceType] ?? 9) < (sortOrder[$1.deviceType] ?? 9) }

        print("[RawLogCam] discoverBackLenses count=\(options.count) types=[\(options.map { "\($0.shortLabel) \(deviceTypeLabel($0.deviceType))" }.joined(separator: ", "))]")
        return options
    }

    var currentLensUniqueID: String? { device?.uniqueID }

    func selectLens(uniqueID: String, completion: ((Error?) -> Void)? = nil) {
        if uniqueID == device?.uniqueID {
            completion?(nil)
            return
        }

        captureQueue.async { [weak self] in
            guard let self else {
                completion?(nil)
                return
            }

            let wasRunning = self.session.isRunning
            self.captureTimer?.cancel()
            self.captureTimer = nil
            self.waitForInFlightClear(timeout: 0.5)

            let previousInput = self.videoInput
            let previousDevice = self.device

            do {
                guard let camera = Self.requireSingleLensDevice(uniqueID: uniqueID) else {
                    throw NSError(
                        domain: "RawLogCam",
                        code: 7,
                        userInfo: [NSLocalizedDescriptionKey: "Lens not found or is multi-cam virtual (Bayer RAW requires single-lens device)"]
                    )
                }
                Self.logSelectedCamera(camera, context: "selectLens")
                let newInput = try AVCaptureDeviceInput(device: camera)

                self.session.beginConfiguration()
                if let old = self.videoInput {
                    self.session.removeInput(old)
                }
                guard self.session.canAddInput(newInput) else {
                    if let old = previousInput, self.session.canAddInput(old) {
                        self.session.addInput(old)
                        self.videoInput = old
                        self.device = previousDevice
                    }
                    self.session.commitConfiguration()
                    throw NSError(domain: "RawLogCam", code: 8, userInfo: [NSLocalizedDescriptionKey: "Cannot switch to this lens"])
                }
                self.session.addInput(newInput)
                self.videoInput = newInput
                self.device = camera

                self.session.sessionPreset = .photo
                self.session.commitConfiguration()

                // Resolve Bayer *after* commit (list is empty mid-configuration)
                do {
                    try self.applyDefaultCameraModes(on: camera)
                    self.lockSensorToTargetFPS(on: camera, fps: self.targetFPS)
                    try self.resolveBayerRAWOrThrow()
                } catch {
                    // Roll back lens
                    if let old = previousInput, let prev = previousDevice {
                        self.session.beginConfiguration()
                        self.session.removeInput(newInput)
                        if self.session.canAddInput(old) {
                            self.session.addInput(old)
                            self.videoInput = old
                            self.device = prev
                        }
                        self.session.sessionPreset = .photo
                        self.session.commitConfiguration()
                        try? self.applyDefaultCameraModes(on: prev)
                        try? self.resolveBayerRAWOrThrow()
                    }
                    throw error
                }
                
                self.cachedBlackLevel = nil
                self.cachedWhiteLevel = nil
                self.cachedISO = nil
                self.prepareRAWPhotoResources()

                if wasRunning {
                    self.startFrameTimer(fps: self.targetFPS)
                }
                print("[RawLogCam] Lens → \(camera.localizedName)")
                DispatchQueue.main.async { completion?(nil) }
            } catch {
                if wasRunning { self.startFrameTimer(fps: self.targetFPS) }
                print("[RawLogCam] Lens switch failed: \(error)")
                DispatchQueue.main.async { completion?(error) }
            }
        }
    }

    // MARK: - Audio sources

    func availableAudioSources() -> [AudioSourceOption] {
        var options: [AudioSourceOption] = []
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setActive(true, options: [])

        var seen = Set<String>()
        for port in audioSession.availableInputs ?? [] {
            guard !seen.contains(port.uid) else { continue }
            seen.insert(port.uid)
            let name: String
            switch port.portType {
            case .builtInMic: name = "iPhone"
            case .headsetMic: name = port.portName.isEmpty ? "Headset" : port.portName
            case .usbAudio: name = port.portName.isEmpty ? "USB Mic" : port.portName
            case .bluetoothHFP, .bluetoothA2DP: name = port.portName.isEmpty ? "Bluetooth" : port.portName
            default: name = port.portName.isEmpty ? port.portType.rawValue : port.portName
            }
            options.append(AudioSourceOption(id: port.uid, name: name, portUID: port.uid))
        }
        return options
    }

    var currentAudioPortUID: String? { selectedAudioPortUID }

    func selectAudioSource(portUID: String?, completion: ((Error?) -> Void)? = nil) {
        if portUID == selectedAudioPortUID {
            completion?(nil)
            return
        }

        captureQueue.async { [weak self] in
            guard let self else {
                completion?(nil)
                return
            }

            self.isReconfiguringAudio = true
            self.waitForInFlightClear(timeout: 0.4)
            self.captureTimer?.cancel()
            self.captureTimer = nil

            let wasRunning = self.session.isRunning
            let audioSession = AVAudioSession.sharedInstance()

            do {
                try audioSession.setActive(true, options: [])

                if let portUID {
                    if self.audioInput == nil {
                        self.session.beginConfiguration()
                        if let mic = AVCaptureDevice.default(for: .audio) {
                            try self.attachAudioDevice(mic)
                        }
                        self.session.commitConfiguration()
                    }
                    guard let port = audioSession.availableInputs?.first(where: { $0.uid == portUID }) else {
                        self.isReconfiguringAudio = false
                        if wasRunning { self.startFrameTimer(fps: self.targetFPS) }
                        let err = NSError(domain: "RawLogCam", code: 5, userInfo: [NSLocalizedDescriptionKey: "Mic port not found — reconnect"])
                        DispatchQueue.main.async { completion?(err) }
                        return
                    }
                    try audioSession.setPreferredInput(port)
                    self.selectedAudioPortUID = portUID
                    print("[RawLogCam] Audio port → \(port.portName)")
                } else {
                    if self.audioInput != nil {
                        self.session.beginConfiguration()
                        if let existing = self.audioInput {
                            self.session.removeInput(existing)
                        }
                        self.audioInput = nil
                        self.session.commitConfiguration()
                    }
                    self.selectedAudioPortUID = nil
                    print("[RawLogCam] Audio: Off")
                }

                self.isReconfiguringAudio = false
                if wasRunning { self.startFrameTimer(fps: self.targetFPS) }
                DispatchQueue.main.async { completion?(nil) }
            } catch {
                self.isReconfiguringAudio = false
                if wasRunning { self.startFrameTimer(fps: self.targetFPS) }
                print("[RawLogCam] Audio switch failed: \(error)")
                DispatchQueue.main.async { completion?(error) }
            }
        }
    }

    private func attachAudioDevice(_ mic: AVCaptureDevice) throws {
        let input = try AVCaptureDeviceInput(device: mic)
        guard session.canAddInput(input) else {
            throw NSError(domain: "RawLogCam", code: 6, userInfo: [NSLocalizedDescriptionKey: "Cannot add mic input"])
        }
        session.addInput(input)
        self.audioInput = input
    }

    private static func preferredBayerFormat(from formats: [OSType]) -> OSType? {
        let fourteenBit: Set<OSType> = [
            kCVPixelFormatType_14Bayer_RGGB,
            kCVPixelFormatType_14Bayer_GRBG,
            kCVPixelFormatType_14Bayer_GBRG,
            kCVPixelFormatType_14Bayer_BGGR
        ]
        if let match = formats.first(where: { fourteenBit.contains($0) }) {
            return match
        }
        return formats.first
    }

    static func cfaPattern(forBayerFormat format: OSType) -> Int32 {
        cfaPatternOptional(forBayerFormat: format) ?? DeviceCapabilities.defaultBayerPattern.rawValue
    }

    /// nil if FourCC not a known Bayer layout (caller may apply device override).
    static func cfaPatternOptional(forBayerFormat format: OSType) -> Int32? {
        switch format {
        case kCVPixelFormatType_14Bayer_RGGB: return 0
        case kCVPixelFormatType_14Bayer_GRBG: return 1
        case kCVPixelFormatType_14Bayer_GBRG: return 2
        case kCVPixelFormatType_14Bayer_BGGR: return 3
        default:
            let s = fourCCString(format).lowercased()
            if s.contains("rggb") || s.contains("rgg4") { return 0 }
            if s.contains("grbg") || s.contains("grb4") { return 1 }
            if s.contains("gbrg") || s.contains("gbr4") { return 2 }
            if s.contains("bggr") || s.contains("bgg4") { return 3 }
            return nil
        }
    }

    /// Optional per-device CFA override from DeviceCapabilities (nil = use live OSType/DNG).
    var bayerPatternOverride: Int32?

    func currentWhiteBalanceGains() -> AVCaptureDevice.WhiteBalanceGains? {
        device?.deviceWhiteBalanceGains
    }

    var activeDevice: AVCaptureDevice? { device }

    // MARK: - Focus API

    func setManualFocus(lensPosition: Float) {
        guard let device = device else { return }
        do {
            try device.lockForConfiguration()
            device.setFocusModeLocked(lensPosition: lensPosition, completionHandler: nil)
            device.unlockForConfiguration()
        } catch {
            print("[CaptureController] Error setting manual focus: \(error)")
        }
    }

    func setContinuousAutoFocus() {
        guard let device = device else { return }
        do {
            try device.lockForConfiguration()
            if device.isFocusModeSupported(.autoFocus) {
                device.focusMode = .autoFocus
            } else if device.isFocusModeSupported(.autoFocus) {
                device.focusMode = .autoFocus
            }
            device.unlockForConfiguration()
        } catch {
            print("[CaptureController] Error setting auto focus: \(error)")
        }
    }

    func setFocusPointOfInterest(_ point: CGPoint, lock: Bool = false) {
        guard let device = activeDevice ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        do {
            try device.lockForConfiguration()
            if device.isFocusPointOfInterestSupported {
                device.focusPointOfInterest = point
            }
            if lock {
                if device.isFocusModeSupported(.autoFocus) {
                    device.focusMode = .autoFocus
                }
            } else {
                if device.isFocusModeSupported(.autoFocus) {
                    device.focusMode = .autoFocus
                } else if device.isFocusModeSupported(.autoFocus) {
                    device.focusMode = .autoFocus
                }
            }
            device.unlockForConfiguration()
        } catch {
            print("[CaptureController] Failed to set focus point: \(error)")
        }
    }

    // MARK: - Session Lifecycle

    func startSession() {
        captureQueue.async { [weak self] in
            guard let self else { return }
            self.session.startRunning()
            self.startFrameTimer(fps: self.targetFPS)
        }
    }

    func stopSession() {
        captureTimer?.cancel()
        captureTimer = nil
        captureQueue.async { [weak self] in
            self?.session.stopRunning()
        }
    }

    func setCaptureFPS(_ fps: Double) {
        let clamped = max(1, min(30, fps))
        captureQueue.async { [weak self] in
            guard let self else { return }
            self.targetFPS = clamped
            self.minFrameInterval = 1.0 / clamped
            guard self.session.isRunning else { return }
            self.startFrameTimer(fps: clamped)
        }
    }

    // MARK: - Continuous RAW Capture Loop

    private func startFrameTimer(fps: Double) {
        captureTimer?.cancel()
        captureTimer = nil
        minFrameInterval = 1.0 / max(1, fps)
        // Aggressive poll so we never sit idle after a free slot
        let poll = min(1.0 / 120.0, minFrameInterval / 2)
        let timer = DispatchSource.makeTimerSource(queue: captureQueue)
        timer.schedule(deadline: .now(), repeating: poll)
        timer.setEventHandler { [weak self] in
            self?.captureOneRawFrame()
        }
        timer.resume()
        captureTimer = timer
        print("[RawLogCam] Capture burst → \(fps) fps (interval \(String(format: "%.3f", minFrameInterval))s)")
    }

    private func captureOneRawFrame() {
        if isReconfiguringAudio { return }

        let now = CACurrentMediaTime()
        if now - lastCaptureStart < minFrameInterval {
            return
        }

        captureLock.lock()
        if inFlightCaptures >= maxInFlight {
            captureLock.unlock()
            return
        }
        // Responsive capture: also respect captureReadiness when available
        if #available(iOS 17.0, *) {
            if photoOutput.isResponsiveCaptureEnabled {
                switch photoOutput.captureReadiness {
                case .ready, .notReadyMomentarily:
                    break
                default:
                    captureLock.unlock()
                    return
                }
            }
        }
        inFlightCaptures += 1
        lastCaptureStart = now
        captureLock.unlock()

        if !photoOutput.availableRawPhotoPixelFormatTypes.contains(rawPixelFormat) {
            if let first = Self.preferredBayerFormat(
                from: photoOutput.availableRawPhotoPixelFormatTypes.filter {
                    AVCapturePhotoOutput.isBayerRAWPixelFormat($0)
                }
            ) {
                rawPixelFormat = first
                formatCFAPattern = Self.cfaPattern(forBayerFormat: first)
            } else {
                endInFlight()
                return
            }
        }

        // Unique settings object every shot (required). Type matches prepared array.
        let settings = makeRAWPhotoSettings()
        photoOutput.capturePhoto(with: settings, delegate: self)
    }

    private func endInFlight() {
        captureLock.lock()
        inFlightCaptures = max(0, inFlightCaptures - 1)
        captureLock.unlock()
        // Immediately attempt next still
        captureQueue.async { [weak self] in
            self?.captureOneRawFrame()
        }
    }

    private func waitForInFlightClear(timeout: TimeInterval) {
        let deadline = Date().addingTimeInterval(timeout)
        while true {
            captureLock.lock()
            let n = inFlightCaptures
            captureLock.unlock()
            if n == 0 || Date() >= deadline { break }
            Thread.sleep(forTimeInterval: 0.005)
        }
    }

    // MARK: - Bayer buffer copy (critical for stills rate)

    /// Deep-copy Bayer plane so AVCapture can recycle its pool and start the next RAW.
    private static func copyBayerPixelBuffer(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(src)
        let height = CVPixelBufferGetHeight(src)
        let format = CVPixelBufferGetPixelFormatType(src)

        var dst: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
            kCVPixelBufferMetalCompatibilityKey: true
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, format,
            attrs as CFDictionary, &dst
        )
        guard status == kCVReturnSuccess, let dst else { return nil }

        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
            CVPixelBufferUnlockBaseAddress(dst, [])
        }

        guard let srcBase = CVPixelBufferGetBaseAddress(src),
              let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }

        let srcBPR = CVPixelBufferGetBytesPerRow(src)
        let dstBPR = CVPixelBufferGetBytesPerRow(dst)
        let copyW = min(srcBPR, dstBPR)

        if srcBPR == dstBPR {
            memcpy(dstBase, srcBase, srcBPR * height)
        } else {
            for y in 0..<height {
                memcpy(
                    dstBase.advanced(by: y * dstBPR),
                    srcBase.advanced(by: y * srcBPR),
                    copyW
                )
            }
        }
        return dst
    }
}

// MARK: - Photo delegate

extension CaptureController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, willCapturePhotoFor resolvedSettings: AVCaptureResolvedPhotoSettings) {
        AudioServicesDisposeSystemSoundID(1108)
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didCapturePhotoFor resolvedSettings: AVCaptureResolvedPhotoSettings) {
        AudioServicesDisposeSystemSoundID(1108)
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        // Always free the capture slot ASAP after we copy the buffer.
        defer { endInFlight() }

        if let error {
            print("[RawLogCam] Capture error: \(error.localizedDescription)")
            return
        }

        guard let systemBuffer = photo.pixelBuffer else {
            print("[RawLogCam] No pixel buffer on RAW photo")
            return
        }

        // CRITICAL: copy then drop system buffer reference so next RAW can start.
        guard let owned = Self.copyBayerPixelBuffer(systemBuffer) else {
            print("[RawLogCam] Failed to copy Bayer buffer")
            return
        }

        let bufferFormat = CVPixelBufferGetPixelFormatType(owned)
        let metaCFA = extractCFAPattern(from: photo)
        let fromFourCC = AVCapturePhotoOutput.isBayerRAWPixelFormat(bufferFormat)
            ? Self.cfaPatternOptional(forBayerFormat: bufferFormat)
            : nil
        // Priority: device-model override → DNG metadata → OSType FourCC → session default → RGGB
        let cfa: Int32
        if let override = bayerPatternOverride {
            cfa = override
        } else if let metaCFA {
            cfa = metaCFA
        } else if let fromFourCC {
            cfa = fromFourCC
        } else {
            cfa = formatCFAPattern
        }
        let currentISO = device?.iso ?? 0
        let levels: (Float, Float)
        
        if let cb = cachedBlackLevel, let cw = cachedWhiteLevel, let ci = cachedISO, abs(ci - currentISO) < 1.0 {
            levels = (cb, cw)
        } else {
            levels = extractBlackWhiteLevels(from: photo, pixelFormat: rawPixelFormat)
            cachedBlackLevel = levels.0
            cachedWhiteLevel = levels.1
            cachedISO = currentISO
        }
        
        let black = levels.0
        let white = levels.1

        // Disable hardcoded LSC polynomial. Hardcoding 0.35 across all iPhones/lenses
        // will cause severe vignette under/over correction. (Phase 2 feature pending).
        let lsc = SIMD4<Float>(0.0, 0.0, 0.0, 0.0)

        let frameData = RawFrameData(
            pixelBuffer: owned,
            whiteBalanceGains: device?.deviceWhiteBalanceGains,
            cfaPattern: cfa,
            blackLevel: black,
            whiteLevel: white,
            pixelFormat: bufferFormat,
            lscCoefficients: lsc,
            iso: currentISO
        )

        onRawFrameData?(frameData)
    }

    private func extractBlackWhiteLevels(from photo: AVCapturePhoto, pixelFormat: OSType) -> (Float, Float) {
        let fullScale: Float = 65535.0
        var blackRaw: Float = 0
        
        // Dynamically guess fallback based on format, in case metadata fails
        var whiteRaw: Float = 16383 // Assume 14-bit
        if pixelFormat == kCVPixelFormatType_14Bayer_RGGB || pixelFormat == kCVPixelFormatType_14Bayer_GRBG ||
           pixelFormat == kCVPixelFormatType_14Bayer_GBRG || pixelFormat == kCVPixelFormatType_14Bayer_BGGR {
            whiteRaw = 16383
        } else {
            // If it's a 10 or 12 bit bayer format (or custom), fallback appropriately (though iOS prefers 14)
            // But realistically, if DNG metadata exists, it overrides this anyway.
        }

        let metadata = photo.metadata
        if let dng = metadata["{DNG}"] as? [String: Any] {
            if let bl = Self.floatFromMetadata(dng["BlackLevel"]) {
                blackRaw = bl
            } else if let bls = dng["BlackLevel"] as? [Any], let first = Self.floatFromAny(bls.first) {
                blackRaw = first
            }
            if let wl = Self.floatFromMetadata(dng["WhiteLevel"]) {
                whiteRaw = wl
            } else if let wls = dng["WhiteLevel"] as? [Any], let first = Self.floatFromAny(wls.first) {
                whiteRaw = first
            }
        }

        if blackRaw == 0, let tiff = metadata["{TIFF}"] as? [String: Any],
           let bl = Self.floatFromMetadata(tiff["BlackLevel"]) {
            blackRaw = bl
        }

        let black = max(0, blackRaw / fullScale)
        var white = whiteRaw / fullScale
        if white <= black + 1e-6 {
            white = 16383.0 / fullScale
        }
        return (black, white)
    }

    private static func floatFromMetadata(_ value: Any?) -> Float? {
        floatFromAny(value)
    }

    private static func floatFromAny(_ value: Any?) -> Float? {
        switch value {
        case let f as Float: return f
        case let d as Double: return Float(d)
        case let i as Int: return Float(i)
        case let n as NSNumber: return n.floatValue
        case let arr as [Any]: return floatFromAny(arr.first)
        default: return nil
        }
    }

    private func extractCFAPattern(from photo: AVCapturePhoto) -> Int32? {
        let metadata = photo.metadata
        let cfaArray: [Int]? = {
            if let dng = metadata["{DNG}"] as? [String: Any] { return Self.intArray(from: dng["CFAPattern"]) }
            if let tiff = metadata["{TIFF}"] as? [String: Any] { return Self.intArray(from: tiff["CFAPattern"]) }
            if let exif = metadata["{Exif}"] as? [String: Any] { return Self.intArray(from: exif["CFAPattern"]) }
            return nil
        }()
        guard let pattern = cfaArray, pattern.count >= 4 else { return nil }
        let p4 = Array(pattern.prefix(4))
        if p4 == [0, 1, 1, 2] { return 0 }
        if p4 == [1, 0, 2, 1] { return 1 }
        if p4 == [1, 2, 0, 1] { return 2 }
        if p4 == [2, 1, 1, 0] { return 3 }
        return nil
    }

    private static func intArray(from value: Any?) -> [Int]? {
        switch value {
        case let arr as [Int]: return arr
        case let arr as [NSNumber]: return arr.map { $0.intValue }
        case let arr as [Any]: return arr.compactMap { ($0 as? NSNumber)?.intValue ?? ($0 as? Int) }
        case let data as Data: return data.map { Int($0) }
        default: return nil
        }
    }
}

// MARK: - Audio delegate

extension CaptureController: AVCaptureAudioDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        onAudioSample?(sampleBuffer)
    }
}

// MARK: - FourCC helpers

private func fourCC(_ type: OSType) -> String {
    fourCCString(type)
}

private func fourCCString(_ type: OSType) -> String {
    let chars: [UInt8] = [
        UInt8((type >> 24) & 0xFF),
        UInt8((type >> 16) & 0xFF),
        UInt8((type >> 8) & 0xFF),
        UInt8(type & 0xFF)
    ]
    let s = String(bytes: chars, encoding: .ascii) ?? String(type)
    return "'\(s)' (0x\(String(type, radix: 16)))"
}
