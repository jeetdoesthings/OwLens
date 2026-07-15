import SwiftUI
import UIKit
import MetalKit
import MetalPerformanceShaders

/// MTKView wrapper — aspect-fits log texture into landscape drawable (no stretch / fake 9:16)
struct CameraPreviewView: UIViewRepresentable {
    let metalPipeline: MetalPipeline
    @Binding var currentTexture: MTLTexture?
    @Binding var showClipping: Bool
 
    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: metalPipeline.device)
        mtkView.delegate = context.coordinator
        mtkView.framebufferOnly = true
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
        context.coordinator.showClipping = showClipping
    }
 
    func makeCoordinator() -> Coordinator {
        Coordinator(metalPipeline: metalPipeline)
    }
 
    final class Coordinator: NSObject, MTKViewDelegate {
        let metalPipeline: MetalPipeline
        var currentTexture: MTLTexture?
        var showClipping: Bool = false
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
 
            let originX = (destW - fitW) / 2
            let originY = (destH - fitH) / 2
            
            if let renderPassDescriptor = view.currentRenderPassDescriptor,
               let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor),
               let renderPipeline = renderPipeline {
                
                renderEncoder.setRenderPipelineState(renderPipeline)
                renderEncoder.setFragmentTexture(texture, index: 0)
                
                var offset = SIMD2<Int32>(Int32(originX), Int32(originY))
                renderEncoder.setFragmentBytes(&offset, length: MemoryLayout<SIMD2<Int32>>.size, index: 0)
                
                var size = SIMD2<Int32>(Int32(fitW), Int32(fitH))
                renderEncoder.setFragmentBytes(&size, length: MemoryLayout<SIMD2<Int32>>.size, index: 1)
                
                var clipping: Int32 = showClipping ? 1 : 0
                renderEncoder.setFragmentBytes(&clipping, length: MemoryLayout<Int32>.size, index: 2)
                
                // Draw full-screen triangle
                renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                renderEncoder.endEncoding()
            }
 
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }
}
