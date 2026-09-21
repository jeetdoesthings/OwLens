import SwiftUI
import UIKit
import MetalKit
import QuartzCore

/// Thread-safe texture delivery pipe from the Metal pipeline to MTKView.
/// Completely bypasses SwiftUI's view body evaluation loop at 24/30 fps,
/// scheduling direct layer updates onto the MTKView.
final class PreviewFeed: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var currentTexture: MTLTexture?
    private weak var targetView: MTKView?
    private weak var coordinator: CameraPreviewView.Coordinator?

    func register(view: MTKView, coordinator: CameraPreviewView.Coordinator) {
        lock.lock()
        defer { lock.unlock() }
        self.targetView = view
        self.coordinator = coordinator
        if let currentTexture {
            coordinator.currentTexture = currentTexture
            view.setNeedsDisplay()
        }
    }

    func unregister() {
        lock.lock()
        defer { lock.unlock() }
        self.targetView = nil
        self.coordinator = nil
    }

    func submit(texture: MTLTexture) {
        lock.lock()
        currentTexture = texture
        let view = targetView
        let coord = coordinator
        lock.unlock()

        guard let view, let coord else { return }
        if Thread.isMainThread {
            coord.currentTexture = texture
            view.setNeedsDisplay()
        } else {
            DispatchQueue.main.async {
                coord.currentTexture = texture
                view.setNeedsDisplay()
            }
        }
    }
}

/// MTKView wrapper — aspect-fits log texture into landscape drawable (no stretch / fake 9:16)
struct CameraPreviewView: UIViewRepresentable {
    let metalPipeline: MetalPipeline
    let previewFeed: PreviewFeed
    @Binding var showClipping: Bool
    @Binding var showFocusPeaking: Bool
    var showDisplayLUT: Bool = false
    var overlayOnly: Bool = false
 
    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: metalPipeline.device)
        mtkView.delegate = context.coordinator
        mtkView.framebufferOnly = true
        mtkView.colorPixelFormat = .bgra8Unorm
        // Drive rendering directly from incoming processed frames (setNeedsDisplay)
        // rather than an unsynchronized 30Hz timer, eliminating 3:2 pulldown judder.
        mtkView.enableSetNeedsDisplay = true
        mtkView.isPaused = true
        mtkView.autoResizeDrawable = true
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: overlayOnly ? 0 : 1)
        mtkView.backgroundColor = overlayOnly ? .clear : .black
        mtkView.isOpaque = !overlayOnly
        mtkView.layer.isOpaque = !overlayOnly
        (mtkView.layer as? CAMetalLayer)?.isOpaque = !overlayOnly
        mtkView.contentMode = .scaleToFill
        // Avoid UIKit transforming layers into portrait letterbox mid-record
        mtkView.autoresizingMask = [.flexibleWidth, .flexibleHeight]

        previewFeed.register(view: mtkView, coordinator: context.coordinator)
        return mtkView
    }

    static func dismantleUIView(_ uiView: MTKView, coordinator: Coordinator) {
        coordinator.previewFeed?.unregister()
    }
 
    func updateUIView(_ uiView: MTKView, context: Context) {
        if context.coordinator.previewFeed !== previewFeed {
            context.coordinator.previewFeed?.unregister()
            context.coordinator.previewFeed = previewFeed
            previewFeed.register(view: uiView, coordinator: context.coordinator)
        }
        context.coordinator.showClipping = showClipping
        context.coordinator.showFocusPeaking = showFocusPeaking
        context.coordinator.showDisplayLUT = showDisplayLUT
        context.coordinator.isAppActive = UIApplication.shared.applicationState == .active
        if context.coordinator.overlayOnly != overlayOnly {
            context.coordinator.overlayOnly = overlayOnly
            uiView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: overlayOnly ? 0 : 1)
            uiView.backgroundColor = overlayOnly ? .clear : .black
            uiView.isOpaque = !overlayOnly
            uiView.layer.isOpaque = !overlayOnly
            (uiView.layer as? CAMetalLayer)?.isOpaque = !overlayOnly
        }
        uiView.setNeedsDisplay()
    }
 
    func makeCoordinator() -> Coordinator {
        let coord = Coordinator(metalPipeline: metalPipeline)
        coord.previewFeed = previewFeed
        return coord
    }
 
    struct DisplayUniforms {
        var destOffset: SIMD2<Int32>
        var destSize: SIMD2<Int32>
        var showClipping: Int32
        var showFocusPeaking: Int32
        var overlayOnly: Int32
        var showDisplayLUT: Int32
        var curveType: Int32
    }

    final class Coordinator: NSObject, MTKViewDelegate {
        let metalPipeline: MetalPipeline
        weak var previewFeed: PreviewFeed?
        var currentTexture: MTLTexture?
        var showClipping: Bool = false
        var showFocusPeaking: Bool = false
        var showDisplayLUT: Bool = false
        var overlayOnly: Bool = false
        /// Cached app state — updated from MainActor via updateUIView, read on render thread.
        var isAppActive: Bool = true
        /// No redundant-draw guard needed: the display link runs at 30 fps and frames
        /// arrive at ~24 fps. The ~6 extra fullscreen blits per second are negligible.
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
            // isAppActive is set on MainActor via updateUIView, read safely here on render thread
            guard isAppActive else { return }

            // When overlayOnly is true (hardware Rec.709 preview is active) and no overlays are requested,
            // skip all texture sampling and fragment shader passes entirely to free 100% GPU bandwidth.
            if overlayOnly && !showClipping && !showFocusPeaking {
                if let drawable = view.currentDrawable,
                   let commandBuffer = renderCommandQueue?.makeCommandBuffer() {
                    if let renderPassDescriptor = view.currentRenderPassDescriptor,
                       let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                        renderEncoder.endEncoding()
                    }
                    commandBuffer.present(drawable)
                    commandBuffer.commit()
                }
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
                
                var uniforms = DisplayUniforms(
                    destOffset: SIMD2<Int32>(Int32(originX), Int32(originY)),
                    destSize: SIMD2<Int32>(Int32(fitW), Int32(fitH)),
                    showClipping: showClipping ? 1 : 0,
                    showFocusPeaking: showFocusPeaking ? 1 : 0,
                    overlayOnly: overlayOnly ? 1 : 0,
                    showDisplayLUT: showDisplayLUT ? 1 : 0,
                    curveType: Int32(metalPipeline.curveType.rawValue)
                )
                renderEncoder.setFragmentBytes(&uniforms, length: MemoryLayout<DisplayUniforms>.stride, index: 0)
                
                // Draw full-screen triangle
                renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                renderEncoder.endEncoding()
            }
 
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }
}
