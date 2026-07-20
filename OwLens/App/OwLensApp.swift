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

/// Shows black “OwLens” splash until camera setup finishes (covers the multi-second black gap).
struct RootView: View {
    @StateObject private var viewModel = CameraViewModel()
    @State private var permissionDenied = false
    @State private var showSplash = true
    /// Minimum splash so the brand always reads, even if setup is fast.
    @State private var minSplashElapsed = false
    @State private var showSilentModeWarning = false
    @State private var tapFocusPoint: CGPoint? = nil
    @State private var focusReticleOpacity: Double = 0
    @State private var touchDownDate: Date? = nil

    var body: some View {
        ZStack {
            if permissionDenied {
                permissionDeniedView
            } else if viewModel.isDeviceUnsupportedForLog && !showSplash {
                unsupportedView
            } else if let pipeline = viewModel.metalPipeline, viewModel.isCameraReady {
                GeometryReader { geo in
                    ZStack {
                        NormalVideoPreviewView(
                            session: viewModel.captureController.session,
                            lensID: viewModel.selectedLens?.uniqueID,
                            videoAspect: viewModel.selectedFormat.aspectRatio
                        )
                            .opacity(viewModel.previewDisplayMode == .normalVideo ? 1 : 0)
                            .allowsHitTesting(false)
                            .ignoresSafeArea()

                        CameraPreviewView(
                            metalPipeline: pipeline,
                            currentTexture: $viewModel.currentTexture,
                            showClipping: $viewModel.showClipping,
                            showFocusPeaking: $viewModel.showFocusPeaking,
                            overlayOnly: viewModel.previewDisplayMode == .normalVideo
                        )
                            .opacity(viewModel.previewDisplayMode == .log || viewModel.showClipping || viewModel.showFocusPeaking ? 1 : 0)
                            .allowsHitTesting(viewModel.previewDisplayMode == .log)
                            .ignoresSafeArea()
                            .onTapGesture { location in
                                let normalized = CGPoint(x: location.x / geo.size.width, y: location.y / geo.size.height)
                                viewModel.triggerTapToFocus(at: location, normalized: normalized)
                            }
                        
                        if let focusPt = viewModel.focusPointLocation {
                            Rectangle()
                                .stroke(Color.yellow, lineWidth: 1.5)
                                .frame(width: 60, height: 60)
                                .position(focusPt)
                                .animation(.spring(), value: viewModel.focusPointLocation)
                        }

                        GridLevelOverlay(
                            showGrid: viewModel.showGrid,
                            showLevel: viewModel.showLevel,
                            videoAspect: viewModel.selectedFormat.aspectRatio,
                            levelMonitor: viewModel.levelMonitor
                        )
                        .ignoresSafeArea()

                        if viewModel.showScopes {
                            ScopesOverlay(data: viewModel.scopeData)
                                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
                                .padding(.leading, 18)
                                .padding(.top, 104)
                                .allowsHitTesting(false)
                        }

                        if let focusPt = tapFocusPoint {
                            Rectangle()
                                .stroke(Color.yellow, lineWidth: 1.5)
                                .frame(width: 60, height: 60)
                                .position(focusPt)
                                .opacity(focusReticleOpacity)
                        }

                        if viewModel.isFocusLocked {
                            Text("AF LOCK")
                                .font(.system(size: 12, weight: .bold, design: .rounded))
                                .foregroundColor(.black)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 4)
                                .background(Color.yellow)
                                .cornerRadius(4)
                                .position(x: geo.size.width / 2, y: geo.size.height * 0.15)
                        }
                    }
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                if touchDownDate == nil {
                                    touchDownDate = Date()
                                }
                            }
                            .onEnded { value in
                                let duration = Date().timeIntervalSince(touchDownDate ?? Date())
                                touchDownDate = nil
                                let isLongPress = duration > 0.4
                                
                                let loc = value.location
                                let aspect = viewModel.selectedFormat.aspectRatio
                                let viewAspect = geo.size.width / geo.size.height
                                var frame = CGRect(origin: .zero, size: geo.size)
                                if aspect > viewAspect {
                                    let h = geo.size.width / aspect
                                    frame = CGRect(x: 0, y: (geo.size.height - h) / 2, width: geo.size.width, height: h)
                                } else {
                                    let w = geo.size.height * aspect
                                    frame = CGRect(x: (geo.size.width - w) / 2, y: 0, width: w, height: geo.size.height)
                                }
                                guard frame.contains(loc) else { return }
                                let x = (loc.x - frame.minX) / frame.width
                                let y = (loc.y - frame.minY) / frame.height
                                viewModel.setFocusPoint(CGPoint(x: x, y: y), lock: isLongPress)
                                
                                tapFocusPoint = loc
                                focusReticleOpacity = 1
                                withAnimation(.easeOut(duration: 0.5).delay(1.5)) {
                                    focusReticleOpacity = 0
                                }
                            }
                    )
                }
                .ignoresSafeArea()

                ControlsView(viewModel: viewModel)
                    .frame(maxWidth: UIDevice.current.userInterfaceIdiom == .pad ? 700 : .infinity)
            } else if !showSplash && viewModel.metalPipeline == nil {
                metalUnavailableView
            } else {
                // Still loading or splash hold
                Color.black.ignoresSafeArea()
            }

            if showSplash {
                SplashView()
                    .transition(.opacity)
                    .zIndex(10)
            }

            if showSilentModeWarning {
                VStack {
                    Spacer()
                    Text("It is advised to put your device in silent mode")
                        .font(.custom("Helvetica-Bold", size: 13))
                        .foregroundColor(.white)
                        .padding(.vertical, 10)
                        .padding(.horizontal, 16)
                        .background(Color.black.opacity(0.85))
                        .cornerRadius(20)
                        .padding(.bottom, 120)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
                .zIndex(5)
            }
        }
        .animation(.easeOut(duration: 0.35), value: showSplash)
        .statusBarHidden()
        .persistentSystemOverlays(.hidden)
        .onAppear {
            requestCameraPermission()
            // Branding minimum ~1.2s; camera may take longer — splash stays until ready
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
                minSplashElapsed = true
                dismissSplashIfReady()
            }
        }
        .onChange(of: viewModel.isCameraReady) { _, ready in
            if ready { dismissSplashIfReady() }
        }
        .onChange(of: viewModel.isDeviceUnsupportedForLog) { _, unsupported in
            if unsupported { dismissSplashIfReady() }
        }
        .onChange(of: viewModel.errorMessage) { _, msg in
            if msg != nil { dismissSplashIfReady() }
        }
        .onChange(of: permissionDenied) { _, denied in
            if denied {
                minSplashElapsed = true
                showSplash = false
            }
        }
        .onChange(of: showSplash) { _, isShowing in
            if !isShowing && !viewModel.captureController.isShutterSoundSuppressionSupported {
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
        .onDisappear {
            viewModel.teardownCamera()
        }
    }

    private func dismissSplashIfReady() {
        guard minSplashElapsed else { return }
        // Ready for camera UI, unsupported gate, or hard failure
        let canLeaveSplash = viewModel.isCameraReady
            || viewModel.isDeviceUnsupportedForLog
            || viewModel.errorMessage != nil
        if canLeaveSplash {
            showSplash = false
        }
    }

    private var permissionDeniedView: some View {
        VStack(spacing: 16) {
            Image(systemName: "camera.fill")
                .font(.system(size: 40))
                .foregroundColor(.white.opacity(0.5))
            Text("Camera access required")
                .font(.custom("Helvetica", size: 18))
                .foregroundColor(.white)
            Text("Enable camera permission in Settings for OwLens to capture Bayer RAW.")
                .font(.custom("Helvetica", size: 13))
                .foregroundColor(.white.opacity(0.6))
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
                .foregroundColor(.red.opacity(0.9))
            Text("Device not supported for Log")
                .font(.custom("Helvetica", size: 18))
                .foregroundColor(.white)
            Text("Bayer RAW stills are required. This phone does not expose a Bayer RAW photo format, so log recording is disabled.")
                .font(.custom("Helvetica", size: 13))
                .foregroundColor(.white.opacity(0.7))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 36)
            if let caps = viewModel.capabilities {
                Text("\(caps.marketingName) · \(caps.machineIdentifier) · \(caps.chipTier.rawValue)")
                    .font(.custom("Helvetica", size: 11))
                    .foregroundColor(.white.opacity(0.45))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
    }

    private var metalUnavailableView: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
                .foregroundColor(.yellow)
            Text("Metal GPU not available")
                .font(.custom("Helvetica", size: 18))
                .foregroundColor(.white)
            Text("OwLens requires a physical device with Metal support.")
                .font(.custom("Helvetica", size: 13))
                .foregroundColor(.white.opacity(0.6))
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
