import AVFoundation
import SwiftUI

/// Stock iPhone ISP preview for the normal monitoring mode.
/// Recording still uses the RAW -> Metal -> HEVC path; this is display-only.
struct NormalVideoPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    let lensID: String?

    func makeUIView(context: Context) -> PreviewLayerView {
        let view = PreviewLayerView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspect
        view.updateVideoOrientation()
        context.coordinator.lensID = lensID
        return view
    }

    func updateUIView(_ uiView: PreviewLayerView, context: Context) {
        if uiView.previewLayer.session !== session {
            uiView.previewLayer.session = session
        }
        if context.coordinator.lensID != lensID {
            context.coordinator.lensID = lensID
            uiView.refreshSession(session)
        }
        uiView.updateVideoOrientation()
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    final class Coordinator {
        var lensID: String?
    }
}

final class PreviewLayerView: UIView {
    override init(frame: CGRect) {
        super.init(frame: frame)
        isUserInteractionEnabled = false
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        isUserInteractionEnabled = false
    }

    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        updateVideoOrientation()
    }

    func refreshSession(_ session: AVCaptureSession) {
        previewLayer.session = nil
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.previewLayer.session = session
            self.previewLayer.videoGravity = .resizeAspect
            self.updateVideoOrientation()
        }
    }

    func updateVideoOrientation() {
        guard let connection = previewLayer.connection,
              connection.isVideoOrientationSupported else { return }

        let interfaceOrientation = window?.windowScene?.interfaceOrientation
        switch interfaceOrientation {
        case .landscapeLeft:
            connection.videoOrientation = .landscapeLeft
        case .landscapeRight:
            connection.videoOrientation = .landscapeRight
        default:
            connection.videoOrientation = .landscapeRight
        }
    }
}
