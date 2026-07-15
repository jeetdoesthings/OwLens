import SwiftUI
import UIKit
import MetalKit
import MetalPerformanceShaders

/// MTKView wrapper — aspect-fits log texture into landscape drawable (no stretch / fake 9:16).
struct CameraPreviewView: UIViewRepresentable {
    let metalPipeline: MetalPipeline
    @Binding var currentTexture: MTLTexture?

    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: metalPipeline.device)
        mtkView.delegate = context.coordinator
        mtkView.framebufferOnly = false
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.preferredFramesPerSecond = 30
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false
        mtkView.autoResizeDrawable = true
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        mtkView.backgroundColor = .black
        mtkView.contentMode = .scaleToFill
        // Avoid UIKit transforming layers into portrait letterbox mid-record
        mtkView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.currentTexture = currentTexture
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(metalPipeline: metalPipeline)
    }

    final class Coordinator: NSObject, MTKViewDelegate {
        let metalPipeline: MetalPipeline
        var currentTexture: MTLTexture?
        private let renderCommandQueue: MTLCommandQueue?
        private let scaler: MPSImageBilinearScale?
        private var fitTexture: MTLTexture?

        init(metalPipeline: MetalPipeline) {
            self.metalPipeline = metalPipeline
            self.renderCommandQueue = metalPipeline.device.makeCommandQueue()
            self.scaler = MPSImageBilinearScale(device: metalPipeline.device)
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            fitTexture = nil
        }

        func draw(in view: MTKView) {
            // Skip Metal when app not active (avoids IOGPU background permission error)
            if UIApplication.shared.applicationState != .active {
                return
            }
            guard let drawable = view.currentDrawable,
                  let commandBuffer = renderCommandQueue?.makeCommandBuffer() else {
                return
            }

            let dest = drawable.texture
            let destW = dest.width
            let destH = dest.height

            // Always clear full drawable black first
            let clearRP = MTLRenderPassDescriptor()
            clearRP.colorAttachments[0].texture = dest
            clearRP.colorAttachments[0].loadAction = .clear
            clearRP.colorAttachments[0].storeAction = .store
            clearRP.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1)
            if let enc = commandBuffer.makeRenderCommandEncoder(descriptor: clearRP) {
                enc.endEncoding()
            }

            guard let texture = currentTexture, let scaler = scaler, destW > 0, destH > 0 else {
                commandBuffer.present(drawable)
                commandBuffer.commit()
                return
            }

            let srcW = CGFloat(texture.width)
            let srcH = CGFloat(texture.height)
            let dstW = CGFloat(destW)
            let dstH = CGFloat(destH)
            guard srcW > 0, srcH > 0 else {
                commandBuffer.present(drawable)
                commandBuffer.commit()
                return
            }

            // Aspect-fit size inside drawable
            let srcAspect = srcW / srcH
            let dstAspect = dstW / dstH
            var fitW: Int
            var fitH: Int
            if srcAspect > dstAspect {
                fitW = destW
                fitH = max(1, Int((dstW / srcAspect).rounded()))
            } else {
                fitH = destH
                fitW = max(1, Int((dstH * srcAspect).rounded()))
            }
            fitW = min(fitW, destW)
            fitH = min(fitH, destH)

            // Intermediate fit texture
            if fitTexture == nil || fitTexture?.width != fitW || fitTexture?.height != fitH {
                let desc = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .bgra8Unorm, width: fitW, height: fitH, mipmapped: false)
                desc.usage = [.shaderWrite, .shaderRead]
                desc.storageMode = .private
                fitTexture = metalPipeline.device.makeTexture(descriptor: desc)
            }
            guard let fitTex = fitTexture else {
                commandBuffer.present(drawable)
                commandBuffer.commit()
                return
            }

            scaler.encode(commandBuffer: commandBuffer, sourceTexture: texture, destinationTexture: fitTex)

            // Blit centered into drawable
            let originX = (destW - fitW) / 2
            let originY = (destH - fitH) / 2
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.copy(
                    from: fitTex,
                    sourceSlice: 0, sourceLevel: 0,
                    sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                    sourceSize: MTLSize(width: fitW, height: fitH, depth: 1),
                    to: dest,
                    destinationSlice: 0, destinationLevel: 0,
                    destinationOrigin: MTLOrigin(x: originX, y: originY, z: 0)
                )
                blit.endEncoding()
            }

            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }
}
