import SwiftUI
import AVFoundation
import AVKit
import MediaPlayer
import Combine

@main
struct OwLensApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
        }
    }
}

/// Root container managing camera lifecycle, overlays, gestures, and permissions.
struct RootView: View {
    @StateObject private var viewModel = CameraViewModel()
    @State private var permissionDenied = false
    @State private var showSilentModeWarning = false
    @State private var tapFocusPoint: CGPoint? = nil
    @State private var focusReticleOpacity: Double = 0
    @State private var focusReticleWorkItem: DispatchWorkItem? = nil

    var body: some View {
        ZStack {
            // Hardware Volume Button & Camera Control Shutter Interception (active only after camera is ready)
            if viewModel.isCameraReady {
                HardwareShutterInteractionView {
                    viewModel.toggleRecording()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            }

            if permissionDenied {
                permissionDeniedView
            } else if viewModel.isDeviceUnsupportedForLog {
                unsupportedView
            } else if let pipeline = viewModel.metalPipeline, viewModel.isCameraReady {
                GeometryReader { geo in
                    let videoRect = aspectFitRect(in: geo.size, aspect: viewModel.selectedFormat.aspectRatio)

                    ZStack {
                        // Metal Pipeline Preview (Bayer RAW -> Demosaic -> Rec.709/Log preview + Overlays)
                        CameraPreviewView(
                            metalPipeline: pipeline,
                            previewFeed: viewModel.previewFeed,
                            showClipping: $viewModel.showClipping,
                            showFocusPeaking: $viewModel.showFocusPeaking,
                            showDisplayLUT: viewModel.showDisplayLUT,
                            overlayOnly: false,
                            targetFPS: viewModel.selectedFPS.rawValue
                        )
                        .opacity(1)
                        .allowsHitTesting(true)
                        .ignoresSafeArea()

                        // Rule of Thirds Grid & Spirit Level
                        GridLevelOverlay(
                            showGrid: viewModel.showGrid,
                            showLevel: viewModel.showLevel,
                            videoAspect: viewModel.selectedFormat.aspectRatio,
                            levelMonitor: viewModel.levelMonitor
                        )
                        .ignoresSafeArea()

                        // Red Recording Tally Frame
                        if viewModel.isRecording {
                            RoundedRectangle(cornerRadius: 3, style: .continuous)
                                .strokeBorder(OwLensTheme.recordingRed, lineWidth: 1.5)
                                .frame(width: videoRect.width, height: videoRect.height)
                                .position(x: videoRect.midX, y: videoRect.midY)
                                .allowsHitTesting(false)
                                .transition(.opacity)
                        }

                        // Tap-to-Focus Reticle
                        if let focusPt = tapFocusPoint {
                            cinemaFocusReticle
                                .position(focusPt)
                                .opacity(focusReticleOpacity)
                                .allowsHitTesting(false)
                        }
                    }
                    .contentShape(Rectangle())
                    .onTapGesture(count: 1, coordinateSpace: .local) { loc in
                        if viewModel.activePanel != nil {
                            withAnimation {
                                viewModel.activePanel = nil
                            }
                            return
                        }
                        guard videoRect.contains(loc) else { return }
                        let nx = (loc.x - videoRect.minX) / videoRect.width
                        let ny = (loc.y - videoRect.minY) / videoRect.height

                        let interfaceOrientation = UIApplication.shared.connectedScenes
                            .compactMap { $0 as? UIWindowScene }
                            .first?.interfaceOrientation ?? .landscapeRight

                        let sensorX: CGFloat
                        let sensorY: CGFloat
                        if interfaceOrientation == .landscapeLeft {
                            sensorX = ny
                            sensorY = 1.0 - nx
                        } else {
                            sensorX = 1.0 - ny
                            sensorY = nx
                        }
                        let clampedPoint = CGPoint(
                            x: max(0.0, min(1.0, sensorX)),
                            y: max(0.0, min(1.0, sensorY))
                        )

                        Haptics.impact(.medium)
                        viewModel.setFocusPoint(clampedPoint, lock: true)

                        tapFocusPoint = loc
                        focusReticleWorkItem?.cancel()
                        focusReticleOpacity = 1.0

                        let work = DispatchWorkItem {
                            withAnimation(.easeOut(duration: 0.25)) {
                                focusReticleOpacity = 0
                            }
                        }
                        focusReticleWorkItem = work
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.9, execute: work)
                    }
                }
                .ignoresSafeArea()

                // Interactive HUD Controls & Panels (docked to edges on iPad and iPhone)
                ControlsView(viewModel: viewModel)
                    .ignoresSafeArea()
            } else if viewModel.metalPipeline == nil {
                metalUnavailableView
            } else {
                // Background loading placeholder
                Color.black.ignoresSafeArea()
            }

            // Silent Mode Advisory Banner
            if showSilentModeWarning {
                VStack {
                    Spacer()
                    HStack(spacing: 8) {
                        Image(systemName: "bell.slash.fill")
                            .font(.system(size: 11, weight: .medium))
                        Text("Mute device for silent recording shutter")
                            .font(.appFont(.regular, size: 12))
                        Button {
                            withAnimation {
                                showSilentModeWarning = false
                            }
                        } label: {
                            Image(systemName: "xmark")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundColor(OwLensTheme.textSecondary)
                                .padding(4)
                        }
                        .buttonStyle(.plain)
                    }
                    .foregroundColor(OwLensTheme.textPrimary)
                    .padding(.vertical, 8)
                    .padding(.horizontal, 14)
                    .glassPanel(cornerRadius: OwLensTheme.radiusCard, border: OwLensTheme.glassBorderActive, background: OwLensTheme.glassBaseHeavy)
                    .padding(.bottom, 72)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
                .zIndex(5)
            }
        }
        .statusBarHidden()
        .persistentSystemOverlays(.hidden)
        .onAppear {
            requestCameraPermission()
            if !viewModel.captureController.isShutterSoundSuppressionSupported {
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    withAnimation {
                        showSilentModeWarning = true
                    }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 4.0) {
                        withAnimation {
                            showSilentModeWarning = false
                        }
                    }
                }
            }
        }
        .onDisappear {
            viewModel.teardownCamera()
        }
    }

    // MARK: - Focus Reticle View

    private var cinemaFocusReticle: some View {
        let reticleColor = Color.white
        let bracketLen: CGFloat = 8
        let size: CGFloat = 46

        return ZStack {
            // 4 Corner brackets with drop shadow
            Path { path in
                // Top-Left
                path.move(to: CGPoint(x: 0, y: bracketLen))
                path.addLine(to: CGPoint(x: 0, y: 0))
                path.addLine(to: CGPoint(x: bracketLen, y: 0))

                // Top-Right
                path.move(to: CGPoint(x: size - bracketLen, y: 0))
                path.addLine(to: CGPoint(x: size, y: 0))
                path.addLine(to: CGPoint(x: size, y: bracketLen))

                // Bottom-Left
                path.move(to: CGPoint(x: 0, y: size - bracketLen))
                path.addLine(to: CGPoint(x: 0, y: size))
                path.addLine(to: CGPoint(x: bracketLen, y: size))

                // Bottom-Right
                path.move(to: CGPoint(x: size - bracketLen, y: size))
                path.addLine(to: CGPoint(x: size, y: size))
                path.addLine(to: CGPoint(x: size, y: size - bracketLen))
            }
            .stroke(reticleColor, lineWidth: 1.25)
            .shadow(color: .black.opacity(0.8), radius: 1.5, x: 0, y: 0.5)
            .frame(width: size, height: size)

            // AF LOCK badge illuminated in White
            Text("LOCK")
                .font(.appMono(.bold, size: 7.5))
                .foregroundColor(.black)
                .padding(.horizontal, 4.5)
                .padding(.vertical, 1.5)
                .background(Color.white)
                .clipShape(RoundedRectangle(cornerRadius: 2.5, style: .continuous))
                .shadow(color: .black.opacity(0.6), radius: 1.5, x: 0, y: 0.5)
                .offset(y: -size / 2 - 10)
        }
    }

    // MARK: - Aspect Fit Helper

    private func aspectFitRect(in size: CGSize, aspect: CGFloat) -> CGRect {
        guard size.width > 0, size.height > 0, aspect > 0 else {
            return CGRect(origin: .zero, size: size)
        }
        let viewAspect = size.width / size.height
        if aspect > viewAspect {
            let h = size.width / aspect
            return CGRect(x: 0, y: (size.height - h) / 2, width: size.width, height: h)
        } else {
            let w = size.height * aspect
            return CGRect(x: (size.width - w) / 2, y: 0, width: w, height: size.height)
        }
    }

    // MARK: - Permission Logic

    private var permissionDeniedView: some View {
        VStack(spacing: 16) {
            Image(systemName: "camera.fill")
                .font(.system(size: 40))
                .foregroundColor(OwLensTheme.textMuted)
            Text("Camera Access Required")
                .font(.appFont(.semiBold, size: 20))
                .foregroundColor(.white)
            Text("OwLens requires camera permission to capture uncompressed Bayer RAW sensor streams.")
                .font(.appFont(.regular, size: 13))
                .foregroundColor(OwLensTheme.textSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            Button {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            } label: {
                Text("Open Settings")
                    .font(.appFont(.semiBold, size: 13))
                    .foregroundColor(.black)
                    .padding(.horizontal, 18)
                    .padding(.vertical, 9)
                    .background(Capsule().fill(Color.white))
            }
            .buttonStyle(.plain)
            .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
    }

    private var unsupportedView: some View {
        VStack(spacing: 16) {
            Image(systemName: "xmark.circle.fill")
                .font(.system(size: 44))
                .foregroundColor(OwLensTheme.recordingRed)
            Text("Device Not Supported for Log")
                .font(.appFont(.semiBold, size: 20))
                .foregroundColor(.white)
            Text("Bayer RAW stills are required. This iPhone model does not expose a Bayer RAW sensor stream, so Log recording is unavailable.")
                .font(.appFont(.regular, size: 13))
                .foregroundColor(OwLensTheme.textSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 36)
            if let caps = viewModel.capabilities {
                Text("\(caps.marketingName) · \(caps.machineIdentifier) · \(caps.chipTier.rawValue)")
                    .font(.appMono(.regular, size: 11))
                    .foregroundColor(OwLensTheme.textMuted)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
    }

    private var metalUnavailableView: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.largeTitle)
                .foregroundColor(OwLensTheme.amberWarning)
            Text("Metal GPU Pipeline Unavailable")
                .font(.appFont(.semiBold, size: 20))
                .foregroundColor(.white)
            Text("OwLens requires a physical device with Apple Silicon GPU support.")
                .font(.appFont(.regular, size: 13))
                .foregroundColor(OwLensTheme.textSecondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
    }

    private func requestCameraPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            permissionDenied = false
            viewModel.setupCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                Task { @MainActor in
                    if granted {
                        permissionDenied = false
                        viewModel.setupCamera()
                    } else {
                        permissionDenied = true
                    }
                }
            }
        default:
            permissionDenied = true
        }
    }
}

// MARK: - Hardware Shutter Interception (Volume Buttons & Camera Control)

/// Captures physical volume button presses (Volume Up / Down) and Camera Control events to trigger recording.
/// Uses native AVCaptureEventInteraction (iOS 17.2+) with an off-screen MPVolumeView to suppress the system volume HUD.
struct HardwareShutterInteractionView: UIViewRepresentable {
    let onTrigger: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onTrigger: onTrigger)
    }

    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)

        // 1. Off-screen MPVolumeView suppresses the iOS system volume HUD popup
        let volumeView = MPVolumeView(frame: CGRect(x: -1000, y: -1000, width: 1, height: 1))
        volumeView.alpha = 0.0001
        volumeView.clipsToBounds = true
        view.addSubview(volumeView)

        // 2. AVCaptureEventInteraction for native hardware button handling (iOS 17.2+)
        if #available(iOS 17.2, *) {
            let interaction = AVCaptureEventInteraction(primary: { [weak coordinator = context.coordinator] event in
                if event.phase == .ended {
                    coordinator?.triggerAction()
                }
            }, secondary: { [weak coordinator = context.coordinator] event in
                if event.phase == .ended {
                    coordinator?.triggerAction()
                }
            })
            view.addInteraction(interaction)
        }

        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.onTrigger = onTrigger
    }

    final class Coordinator: NSObject {
        var onTrigger: () -> Void
        private var viewAttachedTime: CFTimeInterval = 0
        private var lastTriggerTime: CFTimeInterval = 0

        init(onTrigger: @escaping () -> Void) {
            self.onTrigger = onTrigger
            self.viewAttachedTime = CACurrentMediaTime()
        }

        func triggerAction() {
            let now = CACurrentMediaTime()
            // Ignore any hardware button events during initial view/hardware stabilization window (2.0s)
            guard now - viewAttachedTime > 2.0 else {
                print("[HardwareShutter] Ignored shutter event during launch stabilization (age: \(now - viewAttachedTime)s)")
                return
            }
            guard now - lastTriggerTime > 0.4 else { return }
            lastTriggerTime = now
            DispatchQueue.main.async { [weak self] in
                self?.onTrigger()
            }
        }
    }
}
