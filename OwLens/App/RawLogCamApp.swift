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

    var body: some View {
        ZStack {
            if permissionDenied {
                permissionDeniedView
            } else if viewModel.isDeviceUnsupportedForLog && !showSplash {
                unsupportedView
            } else if let pipeline = viewModel.metalPipeline, viewModel.isCameraReady {
                CameraPreviewView(
                    metalPipeline: pipeline,
                    currentTexture: $viewModel.currentTexture
                )
                .ignoresSafeArea()

                GridLevelOverlay(
                    showGrid: viewModel.showGrid,
                    showLevel: viewModel.showLevel,
                    videoAspect: viewModel.selectedFormat.aspectRatio,
                    levelMonitor: viewModel.levelMonitor
                )
                .ignoresSafeArea()

                ControlsView(viewModel: viewModel)
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
                    Text("It is advised to put your iPhone in silent mode")
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
            Text("OwLens requires a physical iPhone with Metal support.")
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
