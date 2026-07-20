import AVFoundation
import SwiftUI

/// Stock iPhone ISP preview for the normal monitoring mode.
/// Recording still uses the RAW -> Metal -> HEVC path; this is display-only.
struct NormalVideoPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    let lensID: String?
    let videoAspect: CGFloat

    func makeUIView(context: Context) -> PreviewLayerView {
        let view = PreviewLayerView()
        view.previewLayer.session = session
        view.videoAspect = videoAspect
        view.updateVideoOrientation()
        context.coordinator.lensID = lensID
        context.coordinator.videoAspect = videoAspect
        return view
    }

    func updateUIView(_ uiView: PreviewLayerView, context: Context) {
        if uiView.previewLayer.session !== session {
            uiView.previewLayer.session = session
        }
        if context.coordinator.videoAspect != videoAspect {
            context.coordinator.videoAspect = videoAspect
            uiView.videoAspect = videoAspect
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
        var videoAspect: CGFloat = 4.0 / 3.0
    }
}

final class PreviewLayerView: UIView {
    let previewLayer = AVCaptureVideoPreviewLayer()
    var videoAspect: CGFloat = 4.0 / 3.0 {
        didSet {
            setNeedsLayout()
        }
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        isUserInteractionEnabled = false
        backgroundColor = .black
        previewLayer.videoGravity = .resizeAspectFill
        layer.addSublayer(previewLayer)
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        isUserInteractionEnabled = false
        backgroundColor = .black
        previewLayer.videoGravity = .resizeAspectFill
        layer.addSublayer(previewLayer)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = aspectFitRect(in: bounds, aspect: videoAspect)
        updateVideoOrientation()
    }

    func refreshSession(_ session: AVCaptureSession) {
        previewLayer.session = nil
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.previewLayer.session = session
            self.previewLayer.videoGravity = .resizeAspectFill
            self.previewLayer.frame = self.aspectFitRect(in: self.bounds, aspect: self.videoAspect)
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

    private func aspectFitRect(in bounds: CGRect, aspect: CGFloat) -> CGRect {
        guard bounds.width > 0, bounds.height > 0, aspect > 0 else { return bounds }
        let viewAspect = bounds.width / bounds.height
        if aspect > viewAspect {
            let height = bounds.width / aspect
            return CGRect(x: bounds.minX, y: bounds.midY - height / 2, width: bounds.width, height: height)
        } else {
            let width = bounds.height * aspect
            return CGRect(x: bounds.midX - width / 2, y: bounds.minY, width: width, height: bounds.height)
        }
    }
}
