import SwiftUI

/// Full-screen calibration wizard. Layout mirrors the app's horizontal cinema UI.
struct CalibrationView: View {
    @ObservedObject var viewModel: CalibrationViewModel
    let dismiss: () -> Void

    private let chromeRadius: CGFloat = 8

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            content
                .padding(.horizontal, 40)
                .padding(.vertical, 24)
        }
        .statusBarHidden()
        .persistentSystemOverlays(.hidden)
    }

    @ViewBuilder
    private var content: some View {
        switch viewModel.state {
        case .idle, .instructions:
            instructionsView
        case .capturing(let progress):
            progressView(progress: progress, title: "Capturing calibration frames…")
        case .processing:
            progressView(progress: 1.0, title: "Fitting noise model…")
        case .done:
            resultView(title: "Calibration saved", message: "Per-ISO noise profile, LSC, and green balance are now stored for this lens.", isError: false)
        case .failed(let message):
            resultView(title: "Calibration failed", message: message, isError: true)
        }
    }

    private var instructionsView: some View {
        HStack(spacing: 32) {
            VStack(alignment: .leading, spacing: 16) {
                Text("Lens Calibration")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundColor(.white)

                Text("Place a uniform gray card in front of the lens and set exposure so the card reads roughly 50% IRE. Keep the device steady. Calibration captures RAW frames at multiple ISOs to build a per-lens noise model and shading correction.")
                    .font(.system(size: 13, weight: .regular, design: .rounded))
                    .foregroundColor(.white.opacity(0.8))
                    .lineLimit(nil)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 12) {
                    pillButton("Start", icon: "camera.aperture") {
                        Task {
                            await viewModel.startCalibration()
                        }
                    }

                    pillButton("Close", icon: "xmark") {
                        dismiss()
                    }
                    .opacity(0.7)
                }
            }
            .frame(maxWidth: 520)

            Spacer()

            VStack(spacing: 12) {
                instructionPill(icon: "doc.text", text: "Uniform gray card")
                instructionPill(icon: "sun.max", text: "~50% IRE exposure")
                instructionPill(icon: "hand.raised", text: "Hold steady")
                instructionPill(icon: "clock", text: "About 30 seconds")
            }
            .frame(maxWidth: 260)
        }
    }

    private func progressView(progress: Float, title: String) -> some View {
        VStack(spacing: 20) {
            Text(title)
                .font(.system(size: 20, weight: .semibold, design: .rounded))
                .foregroundColor(.white)

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .fill(Color.white.opacity(0.12))
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .fill(Color.yellow)
                        .frame(width: geo.size.width * CGFloat(progress))
                }
            }
            .frame(height: 12)
            .frame(maxWidth: 420)

            Text("\(Int(progress * 100))%")
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                .foregroundColor(.white.opacity(0.7))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func resultView(title: String, message: String, isError: Bool) -> some View {
        VStack(spacing: 18) {
            Image(systemName: isError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                .font(.system(size: 48))
                .foregroundColor(isError ? .orange : .green)

            Text(title)
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .foregroundColor(.white)

            Text(message)
                .font(.system(size: 13, weight: .regular, design: .rounded))
                .foregroundColor(.white.opacity(0.8))
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)

            pillButton("Done", icon: "xmark") {
                viewModel.reset()
                dismiss()
            }
            .padding(.top, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - UI primitives

    private func pillButton(_ title: String, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .semibold))
                Text(title)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
            }
            .foregroundColor(.black)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(Color.white)
            .clipShape(RoundedRectangle(cornerRadius: chromeRadius, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    private func instructionPill(icon: String, text: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 12, weight: .semibold))
                .frame(width: 20)
            Text(text)
                .font(.system(size: 12, weight: .semibold, design: .rounded))
            Spacer()
        }
        .foregroundColor(.white.opacity(0.9))
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Color.white.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: chromeRadius, style: .continuous))
    }
}
