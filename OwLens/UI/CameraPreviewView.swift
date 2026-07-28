import SwiftUI
import UIKit
import MetalKit
import MetalPerformanceShaders
import QuartzCore

/// MTKView wrapper — aspect-fits log texture into landscape drawable (no stretch / fake 9:16)
struct CameraPreviewView: UIViewRepresentable {
    let metalPipeline: MetalPipeline
    @Binding var currentTexture: MTLTexture?
    @Binding var showClipping: Bool
    @Binding var showFocusPeaking: Bool
    var overlayOnly: Bool = false
 
    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: metalPipeline.device)
        mtkView.delegate = context.coordinator
        mtkView.framebufferOnly = true
        mtkView.colorPixelFormat = .bgra8Unorm
        // Run the display link at a fixed rate so draw(in:) is called reliably, even when
        // SwiftUI's binding diff can't detect MTLTexture identity changes (pooled textures).
        // The lastDrawnTexture guard below skips redundant draws when nothing changed.
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false
        mtkView.preferredFramesPerSecond = 30
        mtkView.autoResizeDrawable = true
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: overlayOnly ? 0 : 1)
        mtkView.backgroundColor = overlayOnly ? .clear : .black
        mtkView.isOpaque = !overlayOnly
        mtkView.layer.isOpaque = !overlayOnly
        (mtkView.layer as? CAMetalLayer)?.isOpaque = !overlayOnly
        mtkView.contentMode = .scaleToFill
        // Avoid UIKit transforming layers into portrait letterbox mid-record
        mtkView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        return mtkView
    }
 
    func updateUIView(_ uiView: MTKView, context: Context) {
        // Propagate latest texture to the coordinator — the display link ticks at 30 Hz
        // so draw(in:) picks it up naturally within ~33 ms of the change.
        context.coordinator.currentTexture = currentTexture
        context.coordinator.showClipping = showClipping
        context.coordinator.showFocusPeaking = showFocusPeaking
        context.coordinator.overlayOnly = overlayOnly
        uiView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: overlayOnly ? 0 : 1)
        uiView.backgroundColor = overlayOnly ? .clear : .black
        uiView.isOpaque = !overlayOnly
        uiView.layer.isOpaque = !overlayOnly
        (uiView.layer as? CAMetalLayer)?.isOpaque = !overlayOnly
        // No need for setNeedsDisplay: the display link runs continuously and draw(in:)
        // uses the lastDrawnTexture guard to skip redundant work.
    }
 
    func makeCoordinator() -> Coordinator {
        Coordinator(metalPipeline: metalPipeline)
    }
 
    final class Coordinator: NSObject, MTKViewDelegate {
        let metalPipeline: MetalPipeline
        var currentTexture: MTLTexture?
        var showClipping: Bool = false
        var showFocusPeaking: Bool = false
        var overlayOnly: Bool = false
        /// Tracks the last texture we rendered so we can skip redundant draws when the
        /// display link ticks faster than new frames arrive or when the same pooled
        /// texture object is reused.
        private var lastDrawnTexture: MTLTexture?
        private let renderCommandQueue: MTLCommandQueue?
        private let renderPipeline: MTLRenderPipelineState?
 
        init(metalPipeline: MetalPipeline) {
            self.metalPipeline = metalPipeline
            self.renderCommandQueue = metalPipeline.device.makeCommandQueue()
            
            let library = metalPipeline.device.makeDefaultLibrary()
            if let vertexFunc = library?.makeFunction(name: "fullscreenVertex"),
               let fragmentFunc = library?.makeFunction(name: "displayFragment") {
                let descriptor = MTLRenderPipelineDescriptor()
                descriptor.vertexFunction = vertexFunc
                descriptor.fragmentFunction = fragmentFunc
                descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
                self.renderPipeline = try? metalPipeline.device.makeRenderPipelineState(descriptor: descriptor)
            } else {
                self.renderPipeline = nil
            }
        }
 
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            // Handled dynamically in draw()
        }
 
        func draw(in view: MTKView) {
            // Skip Metal when app not active (avoids IOGPU background permission error)
            if UIApplication.shared.applicationState != .active {
                lastDrawnTexture = nil
                return
            }
            // Same texture as last draw — nothing new to render. The display link ticks
            // at 30 fps but frames arrive at ~24 fps, so we skip ~6 draws per second.
            if let currentTexture, let lastDrawnTexture, currentTexture === lastDrawnTexture {
                return
            }
            guard let drawable = view.currentDrawable,
                  let commandBuffer = renderCommandQueue?.makeCommandBuffer() else {
                return
            }
 
            let dest = drawable.texture
            let destW = dest.width
            let destH = dest.height
 
            guard let texture = currentTexture, destW > 0, destH > 0 else {
                if let renderPassDescriptor = view.currentRenderPassDescriptor,
                   let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                    renderEncoder.endEncoding()
                }
                commandBuffer.present(drawable)
                commandBuffer.commit()
                return
            }
 
            let srcW = CGFloat(texture.width)
            let srcH = CGFloat(texture.height)
            let dstW = CGFloat(destW)
            let dstH = CGFloat(destH)
            guard srcW > 0, srcH > 0 else {
                if let renderPassDescriptor = view.currentRenderPassDescriptor,
                   let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                    renderEncoder.endEncoding()
                }
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
 
            let originX = (destW - fitW) / 2
            let originY = (destH - fitH) / 2
            
            if let renderPassDescriptor = view.currentRenderPassDescriptor,
               let renderPipeline = renderPipeline {
                renderPassDescriptor.colorAttachments[0].loadAction = .clear
                renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(
                    red: 0,
                    green: 0,
                    blue: 0,
                    alpha: overlayOnly ? 0 : 1
                )
                guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
                    commandBuffer.present(drawable)
                    commandBuffer.commit()
                    return
                }
                
                renderEncoder.setRenderPipelineState(renderPipeline)
                renderEncoder.setFragmentTexture(texture, index: 0)
                
                var offset = SIMD2<Int32>(Int32(originX), Int32(originY))
                renderEncoder.setFragmentBytes(&offset, length: MemoryLayout<SIMD2<Int32>>.size, index: 0)
                
                var size = SIMD2<Int32>(Int32(fitW), Int32(fitH))
                renderEncoder.setFragmentBytes(&size, length: MemoryLayout<SIMD2<Int32>>.size, index: 1)
                
                var clipping: Int32 = showClipping ? 1 : 0
                renderEncoder.setFragmentBytes(&clipping, length: MemoryLayout<Int32>.size, index: 2)
                
                var peaking: Int32 = showFocusPeaking ? 1 : 0
                renderEncoder.setFragmentBytes(&peaking, length: MemoryLayout<Int32>.size, index: 3)
                
                var overlay: Int32 = overlayOnly ? 1 : 0
                renderEncoder.setFragmentBytes(&overlay, length: MemoryLayout<Int32>.size, index: 4)
                
                // Draw full-screen triangle
                renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                renderEncoder.endEncoding()
            }
 
            commandBuffer.present(drawable)
            commandBuffer.commit()
            lastDrawnTexture = currentTexture
        }
    }
}
