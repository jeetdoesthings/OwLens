import SwiftUI
import AVFoundation

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
    @State private var focusReticleScale: CGFloat = 1.3
    @State private var touchDownDate: Date? = nil

    var body: some View {
        ZStack {
            if permissionDenied {
                permissionDeniedView
            } else if viewModel.isDeviceUnsupportedForLog {
                unsupportedView
            } else if let pipeline = viewModel.metalPipeline, viewModel.isCameraReady {
                GeometryReader { geo in
                    let videoRect = aspectFitRect(in: geo.size, aspect: viewModel.selectedFormat.aspectRatio)

                    ZStack {
                        // Stock ISP Preview (Normal Video / Rec.709 monitoring mode)
                        NormalVideoPreviewView(
                            session: viewModel.captureController.session,
                            lensID: viewModel.selectedLens?.uniqueID,
                            videoAspect: viewModel.selectedFormat.aspectRatio
                        )
                        .opacity(viewModel.previewDisplayMode == .normalVideo ? 1 : 0)
                        .allowsHitTesting(false)
                        .ignoresSafeArea()

                        // Metal Pipeline Preview (Bayer RAW -> Demosaic -> Log preview + Overlays)
                        CameraPreviewView(
                            metalPipeline: pipeline,
                            currentTexture: $viewModel.currentTexture,
                            textureChangeCount: $viewModel.textureChangeCount,
                            showClipping: $viewModel.showClipping,
                            showFocusPeaking: $viewModel.showFocusPeaking,
                            showDisplayLUT: viewModel.showDisplayLUT,
                            overlayOnly: viewModel.previewDisplayMode == .normalVideo
                        )
                        .opacity(viewModel.previewDisplayMode == .log || viewModel.showClipping || viewModel.showFocusPeaking ? 1 : 0)
                        .allowsHitTesting(viewModel.previewDisplayMode == .log)
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



                        // Pro-Cinema Tap-to-Focus Reticle
                        if let focusPt = tapFocusPoint {
                            cinemaFocusReticle
                                .position(focusPt)
                                .scaleEffect(focusReticleScale)
                                .opacity(focusReticleOpacity)
                                .allowsHitTesting(false)
                        }
                    }
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { _ in
                                if touchDownDate == nil {
                                    touchDownDate = Date()
                                }
                            }
                            .onEnded { value in
                                let duration = Date().timeIntervalSince(touchDownDate ?? Date())
                                touchDownDate = nil
                                let isLongPress = duration > 0.35
                                
                                let loc = value.location
                                guard videoRect.contains(loc) else { return }
                                let x = (loc.x - videoRect.minX) / videoRect.width
                                let y = (loc.y - videoRect.minY) / videoRect.height
                                
                                Haptics.impact(isLongPress ? .heavy : .medium)
                                viewModel.setFocusPoint(CGPoint(x: x, y: y), lock: isLongPress)
                                
                                tapFocusPoint = loc
                                focusReticleScale = 1.15
                                focusReticleOpacity = 1.0
                                
                                withAnimation(.spring(response: 0.16, dampingFraction: 0.8)) {
                                    focusReticleScale = 1.0
                                }
                                
                                withAnimation(.easeOut(duration: 0.2).delay(isLongPress ? 1.5 : 0.7)) {
                                    focusReticleOpacity = 0
                                }
                            }
                    )
                }
                .ignoresSafeArea()

                // Interactive HUD Controls & Panels
                ControlsView(viewModel: viewModel)
                    .frame(maxWidth: UIDevice.current.userInterfaceIdiom == .pad ? 750 : .infinity)
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
                    HStack(spacing: 6) {
                        Image(systemName: "bell.slash.fill")
                            .font(.system(size: 11, weight: .medium))
                        Text("Mute your device for silent recording shutter")
                            .font(.geist(.regular, size: 12))
                    }
                    .foregroundColor(OwLensTheme.textPrimary)
                    .padding(.vertical, 8)
                    .padding(.horizontal, 16)
                    .glassPanel(cornerRadius: OwLensTheme.radiusCard, border: OwLensTheme.glassBorderActive, background: OwLensTheme.glassBaseHeavy)
                    .padding(.bottom, 80)
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
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
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
        let isLocked = viewModel.isFocusLocked
        let reticleColor = Color.white
        let bracketLen: CGFloat = 7
        let size: CGFloat = 44

        return ZStack {
            // 4 Corner brackets
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
            .stroke(reticleColor.opacity(0.9), lineWidth: 0.75)
            .shadow(color: .black.opacity(0.5), radius: 1, x: 0, y: 0)
            .frame(width: size, height: size)

            // Center target dot
            Circle()
                .fill(reticleColor.opacity(0.8))
                .shadow(color: .black.opacity(0.5), radius: 1, x: 0, y: 0)
                .frame(width: 2, height: 2)

            // AF LOCK badge if locked
            if isLocked {
                Text("LOCK")
                    .font(.geistMono(.semiBold, size: 7))
                    .foregroundColor(.black)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1.5)
                    .background(Color.white.opacity(0.9))
                    .clipShape(RoundedRectangle(cornerRadius: 2, style: .continuous))
                    .offset(y: -size / 2 - 9)
            }
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
                .font(.geist(.semiBold, size: 20))
                .foregroundColor(.white)
            Text("OwLens requires camera permission to capture uncompressed Bayer RAW sensor streams.")
                .font(.geist(.regular, size: 13))
                .foregroundColor(OwLensTheme.textSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
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
                .font(.geist(.semiBold, size: 20))
                .foregroundColor(.white)
            Text("Bayer RAW stills are required. This iPhone model does not expose a Bayer RAW sensor stream, so Log recording is unavailable.")
                .font(.geist(.regular, size: 13))
                .foregroundColor(OwLensTheme.textSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 36)
            if let caps = viewModel.capabilities {
                Text("\(caps.marketingName) · \(caps.machineIdentifier) · \(caps.chipTier.rawValue)")
                    .font(.geistMono(.regular, size: 11))
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
                .font(.geist(.semiBold, size: 20))
                .foregroundColor(.white)
            Text("OwLens requires a physical device with Apple Silicon GPU support.")
                .font(.geist(.regular, size: 13))
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
