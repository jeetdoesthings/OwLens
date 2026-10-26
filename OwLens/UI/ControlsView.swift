import SwiftUI
import AVFoundation

/// Landscape cinema-camera HUD layered over the live viewfinder.
/// Monochromatic redesign: every element earns its pixel.
struct ControlsView: View {
    @ObservedObject var viewModel: CameraViewModel

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Lens Switcher & Circular Zoom Wheel (Underneath all other UI controls)
                lensAndZoomWheelLayer(in: geo)
                    .zIndex(0)

                // Left Monitoring Tools Rail
                leftToolRail(in: geo)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
                    .zIndex(1)

                // Top HUD Status Bar
                topStatusBar(in: geo)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    .zIndex(2)

                // Bottom Exposure Deck & Transient Panels
                VStack(spacing: 8) {
                    Spacer(minLength: 0)
                    transientStatusArea
                    bottomExposureDeck(in: geo)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
                .zIndex(3)

                // Right Record Grip (Shutter Trigger & Scopes on top)
                rightRecordGrip(in: geo)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .trailing)
                    .zIndex(4)

                // Interactive Toast Pill
                if let toast = viewModel.activeToast {
                    let isOffState = toast.contains("OFF") || toast.contains("Disabled")
                    let isLockState = toast.contains("Lock")
                    HStack(spacing: 6) {
                        Image(systemName: isLockState ? "lock.fill" : (isOffState ? "minus.circle.fill" : "checkmark.circle.fill"))
                            .font(.system(size: isLockState ? 10 : 11, weight: .semibold))
                            .foregroundColor(isLockState ? Color.white : (isOffState ? OwLensTheme.textSecondary : OwLensTheme.audioNominal))
                        Text(toast)
                            .font(.appFont(.medium, size: 12))
                            .foregroundColor(.white)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 7)
                    .glassPanel(cornerRadius: OwLensTheme.radiusCard, border: OwLensTheme.glassBorderActive, background: OwLensTheme.glassBaseHeavy)
                    .padding(.top, max(48, ScreenGeometry.topPadding(in: geo) + 38))
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .zIndex(30)
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
        }
        .opacity(viewModel.isHUDHidden ? 0 : 1)
        .simultaneousGesture(
            MagnificationGesture()
                .onChanged { scale in
                    viewModel.handlePinchChanged(scale: scale)
                }
                .onEnded { _ in
                    viewModel.handlePinchEnded()
                }
        )
        .animation(.easeInOut(duration: 0.22), value: viewModel.isHUDHidden)
        .animation(.spring(response: 0.28, dampingFraction: 0.82), value: viewModel.activePanel)
        .animation(.spring(response: 0.28, dampingFraction: 0.82), value: viewModel.isRecording)
        .animation(.spring(response: 0.28, dampingFraction: 0.82), value: viewModel.controlsLocked)
        .animation(.spring(response: 0.28, dampingFraction: 0.82), value: viewModel.showScopes)
    }

    // MARK: - Screen Safe Area & Notch Geometry Helper

    private enum ScreenGeometry {
        static var windowInsets: UIEdgeInsets {
            let activeScene = UIApplication.shared.connectedScenes
                .compactMap({ $0 as? UIWindowScene })
                .first(where: { $0.activationState == .foregroundActive }) ?? UIApplication.shared.connectedScenes
                .compactMap({ $0 as? UIWindowScene })
                .first

            var insets = activeScene?.windows.first(where: { $0.isKeyWindow })?.safeAreaInsets
                ?? activeScene?.windows.first?.safeAreaInsets
                ?? .zero

            // If bottom has home indicator (>= 20pt) but left/right report < 30pt (e.g. edge case or early layout pass),
            // determine the notch side from interface orientation so the UI is NEVER under the notch:
            if insets.bottom >= 20 && insets.left < 30 && insets.right < 30 {
                let orientation = activeScene?.interfaceOrientation ?? .landscapeRight
                if orientation == .landscapeLeft {
                    // Home button on left -> Notch is on the RIGHT
                    insets.right = 59
                } else {
                    // Home button on right -> Notch is on the LEFT
                    insets.left = 59
                }
            }

            return insets
        }

        /// Effective safe padding for leading edge (clears notch/Dynamic Island + generous margin)
        static func leadingPadding(in geo: GeometryProxy) -> CGFloat {
            let insets = windowInsets
            let notch = max(geo.safeAreaInsets.leading, insets.left)
            return notch >= 30 ? (notch + 20) : 28
        }

        /// Effective safe padding for trailing edge (clears notch/Dynamic Island + generous margin)
        static func trailingPadding(in geo: GeometryProxy) -> CGFloat {
            let insets = windowInsets
            let notch = max(geo.safeAreaInsets.trailing, insets.right)
            return notch >= 30 ? (notch + 20) : 28
        }

        /// Effective safe padding for top edge
        static func topPadding(in geo: GeometryProxy) -> CGFloat {
            let insets = windowInsets
            let safeTop = max(geo.safeAreaInsets.top, insets.top)
            return max(12, safeTop + 4)
        }

        /// Effective safe padding for bottom edge
        static func bottomPadding(in geo: GeometryProxy) -> CGFloat {
            let insets = windowInsets
            let safeBottom = max(geo.safeAreaInsets.bottom, insets.bottom)
            return max(16, safeBottom)
        }
    }

    // MARK: - Top Status Bar

    private func topStatusBar(in geo: GeometryProxy) -> some View {
        let safeLeading = ScreenGeometry.leadingPadding(in: geo)
        let safeTrailing = ScreenGeometry.trailingPadding(in: geo)
        let safeTop = ScreenGeometry.topPadding(in: geo)

        return ZStack {
            // Layer 1: True Mathematical Center (Pinned directly to geo.size.width / 2)
            centerStatusGroup
                .frame(maxWidth: .infinity, alignment: .center)

            // Layer 2: Leading Telemetry
            leftStatusGroup
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.leading, safeLeading)

            // Layer 3: Trailing Save & Battery
            rightStatusGroup
                .frame(maxWidth: .infinity, alignment: .trailing)
                .padding(.trailing, safeTrailing)
        }
        .padding(.top, safeTop)
    }

    private var isTopLocked: Bool {
        viewModel.controlsLocked || viewModel.isRecording || viewModel.isSaving || viewModel.isDeviceUnsupportedForLog
    }

    private var leftStatusGroup: some View {
        HStack(spacing: 6) {
            if viewModel.isRecording {
                // Unified Cinema Recording & Media Telemetry Capsule
                HStack(spacing: 7) {
                    HStack(spacing: 5) {
                        Circle()
                            .fill(OwLensTheme.recordingRed)
                            .frame(width: 7, height: 7)
                        Text("REC")
                            .font(.appFont(.bold, size: 10))
                            .foregroundColor(OwLensTheme.recordingRed)
                        Text(viewModel.recordingDuration)
                            .font(.appMono(.bold, size: 12))
                            .foregroundColor(.white)
                    }

                    Rectangle()
                        .fill(Color.white.opacity(0.20))
                        .frame(width: 1, height: 12)

                    HStack(spacing: 4) {
                        Image(systemName: "internaldrive")
                            .font(.system(size: 9, weight: .semibold))
                        Text("REM \(viewModel.estimatedRecordTimeText)")
                            .font(.appMono(.semiBold, size: 10))
                    }
                    .foregroundColor(viewModel.remainingRecordSeconds < 60 ? OwLensTheme.recordingRed : (viewModel.remainingRecordSeconds < 180 ? OwLensTheme.amberWarning : OwLensTheme.textSecondary))
                }
                .padding(.horizontal, 9)
                .frame(height: 30)
                .glassPanel(
                    cornerRadius: OwLensTheme.radiusCard,
                    border: viewModel.remainingRecordSeconds < 60 ? OwLensTheme.glassBorderRed : OwLensTheme.glassBorderRed,
                    background: OwLensTheme.glassBaseHeavy
                )
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
            } else {
                // Not recording: Show storage remaining estimate
                HStack(spacing: 4) {
                    Image(systemName: "internaldrive")
                        .font(.system(size: 9, weight: .medium))
                    Text("REM \(viewModel.estimatedRecordTimeText)")
                        .font(.appMono(.medium, size: 10))
                }
                .foregroundColor(OwLensTheme.textSecondary)
                .padding(.horizontal, 8)
                .frame(height: 30)
                .glassPanel(cornerRadius: OwLensTheme.radiusCard, border: OwLensTheme.glassBorder, background: OwLensTheme.glassBase)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
            }

            if viewModel.droppedFrames > 0 {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 9, weight: .semibold))
                    Text("\(viewModel.droppedFrames) DROPS")
                        .font(.appMono(.semiBold, size: 9))
                }
                .foregroundColor(OwLensTheme.amberWarning)
                .padding(.horizontal, 8)
                .frame(height: 30)
                .glassPanel(cornerRadius: OwLensTheme.radiusCard, border: OwLensTheme.amberWarning.opacity(0.4))
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
            }
        }
    }

    private var timeIndicator: some View {
        Text(viewModel.currentTimeString)
            .font(.appMono(.bold, size: 10))
            .foregroundColor(OwLensTheme.textPrimary)
            .padding(.horizontal, 7)
            .frame(height: 30)
            .glassPanel(cornerRadius: OwLensTheme.radiusCard)
            .lineLimit(1)
            .fixedSize(horizontal: true, vertical: false)
    }

    private var batteryIndicator: some View {
        let level = viewModel.batteryLevel
        let isCharging = viewModel.isBatteryCharging
        let percent: Int? = level >= 0 ? Int(max(0, min(100, (level * 100).rounded()))) : nil
        let isLow = level >= 0 && level < 0.20
        let statusColor: Color = isLow ? OwLensTheme.recordingRed : (isCharging ? OwLensTheme.audioNominal : OwLensTheme.textPrimary)

        return HStack(spacing: 4) {
            HStack(spacing: 0) {
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2.5, style: .continuous)
                        .strokeBorder(statusColor.opacity(0.50), lineWidth: 1)
                        .background(
                            RoundedRectangle(cornerRadius: 2.5, style: .continuous)
                                .fill(Color.black.opacity(0.35))
                        )
                        .frame(width: 20, height: 11)

                    if let pct = percent {
                        RoundedRectangle(cornerRadius: 1.5, style: .continuous)
                            .fill(statusColor)
                            .frame(width: max(2, CGFloat(pct) / 100.0 * 16), height: 7)
                            .padding(.leading, 2)
                    }
                }

                // Terminal nub
                Capsule()
                    .fill(statusColor.opacity(0.50))
                    .frame(width: 1.5, height: 4)
                    .offset(x: 0.5)
            }

            HStack(spacing: 1) {
                if isCharging {
                    Image(systemName: "bolt.fill")
                        .font(.system(size: 7, weight: .bold))
                        .foregroundColor(OwLensTheme.audioNominal)
                }
                Text(percent != nil ? "\(percent!)%" : "--%")
                    .font(.appMono(.bold, size: 9))
                    .foregroundColor(isLow ? OwLensTheme.recordingRed : OwLensTheme.textPrimary)
            }
        }
        .padding(.horizontal, 7)
        .frame(height: 30)
        .glassPanel(cornerRadius: OwLensTheme.radiusCard)
        .lineLimit(1)
        .fixedSize(horizontal: true, vertical: false)
    }

    private var centerStatusGroup: some View {
        HStack(spacing: 6) {
            // Resolution / Format Button (OG 4:3 / 1080 16:9)
            formatPill

            // Log Profile Selector Button (Single tap cycles, long-press opens drawer)
            logCurvePill

            // Denoise Selector Button (Tap opens drawer)
            denoisePill
        }
    }

    private var formatPill: some View {
        HStack(spacing: 3) {
            Text(viewModel.selectedFormat.shortLabel)
                .font(.appFont(.bold, size: 12))
                .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textPrimary)
                .lineLimit(1)
                .fixedSize()

            Text(viewModel.selectedFormat == .openGate ? "4:3" : "16:9")
                .font(.appMono(.medium, size: 8))
                .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textSecondary)
                .lineLimit(1)
                .fixedSize()
        }
        .padding(.horizontal, 7)
        .frame(height: 30)
        .glassPanel(cornerRadius: OwLensTheme.radiusCard)
        .contentShape(Rectangle())
        .onTapGesture {
            guard !isTopLocked else { return }
            Haptics.selection()
            viewModel.cycleFormat()
        }
        .onLongPressGesture(minimumDuration: 0.45) {
            guard !isTopLocked else { return }
            Haptics.impact(.medium)
            viewModel.togglePanel(.format)
        }
        .disabled(isTopLocked)
        .opacity(isTopLocked ? 0.4 : 1.0)
    }

    private var logCurvePill: some View {
        HStack(spacing: 5) {
            Text(viewModel.selectedCurve.shortLabel)
                .font(.appFont(.bold, size: 11))
                .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textPrimary)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
            Text("10-BIT")
                .font(.appMono(.bold, size: 8))
                .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : .white)
                .padding(.horizontal, 4)
                .padding(.vertical, 1.5)
                .background(
                    RoundedRectangle(cornerRadius: 3, style: .continuous)
                        .fill(Color.white.opacity(0.18))
                )
                .lineLimit(1)
                .fixedSize()
        }
        .padding(.horizontal, 8)
        .frame(height: 30)
        .glassPanel(cornerRadius: OwLensTheme.radiusCard)
        .contentShape(Rectangle())
        .onTapGesture {
            guard !isTopLocked else { return }
            Haptics.selection()
            viewModel.toggleLogCurve()
        }
        .onLongPressGesture(minimumDuration: 0.45) {
            guard !isTopLocked else { return }
            Haptics.impact(.medium)
            viewModel.togglePanel(.logCurve)
        }
        .disabled(isTopLocked)
        .opacity(isTopLocked ? 0.4 : 1.0)
    }

    private var denoisePill: some View {
        Button {
            guard !isTopLocked else { return }
            Haptics.selection()
            viewModel.togglePanel(.denoise)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "sparkles")
                    .font(.system(size: 9, weight: .semibold))
                Text(viewModel.denoiseStrength > 0.001 ? "DN \(String(format: "%.1f", viewModel.denoiseStrength))" : "DN OFF")
                    .font(.appMono(.bold, size: 9))
                    .lineLimit(1)
                    .fixedSize()
            }
            .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : (viewModel.denoiseStrength > 0.001 ? OwLensTheme.accent : OwLensTheme.textSecondary))
            .padding(.horizontal, 7)
            .frame(height: 30)
            .glassPanel(cornerRadius: OwLensTheme.radiusCard)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(isTopLocked)
        .opacity(isTopLocked ? 0.4 : 1.0)
    }

    private var rightStatusGroup: some View {
        HStack(spacing: 6) {
            timeIndicator

            batteryIndicator

            // Save Destination (Photos vs Files - tap cycles, long-press opens drawer)
            HStack {
                Image(systemName: saveDestinationIcon(viewModel.selectedSaveDestination))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textSecondary)
            }
            .frame(width: 34, height: 30)
            .glassPanel(cornerRadius: OwLensTheme.radiusCard)
            .contentShape(Rectangle())
            .onTapGesture {
                guard !isTopLocked else { return }
                Haptics.selection()
                toggleSaveDestination()
            }
            .onLongPressGesture(minimumDuration: 0.45) {
                guard !isTopLocked else { return }
                Haptics.impact(.medium)
                viewModel.togglePanel(.save)
            }
            .disabled(isTopLocked)
            .opacity(isTopLocked ? 0.4 : 1.0)
        }
    }

    // MARK: - Left Monitoring Tools Rail

    private func leftToolRail(in geo: GeometryProxy) -> some View {
        let safeLeading = ScreenGeometry.leadingPadding(in: geo)

        return VStack(spacing: 4) {
            monitoringToolButton(
                text: "709",
                isActive: viewModel.showDisplayLUT
            ) {
                viewModel.toggleDisplayLUT()
            }

            monitoringToolButton(
                systemName: "grid",
                isActive: viewModel.showGrid
            ) {
                viewModel.toggleGrid()
            }

            monitoringToolButton(
                systemName: "level",
                isActive: viewModel.showLevel
            ) {
                viewModel.toggleLevel()
            }

            monitoringToolButton(
                systemName: "sun.max.fill",
                isActive: viewModel.showClipping
            ) {
                viewModel.toggleClipping()
            }

            monitoringToolButton(
                systemName: "viewfinder",
                isActive: viewModel.showFocusPeaking
            ) {
                viewModel.toggleFocusPeaking()
            }

            monitoringToolButton(
                systemName: "chart.xyaxis.line",
                isActive: viewModel.showScopes
            ) {
                viewModel.toggleScopes()
            }
        }
        .padding(4)
        .glassPanel(cornerRadius: OwLensTheme.radiusLg, background: OwLensTheme.glassBaseHeavy)
        .padding(.leading, safeLeading)
    }

    private func monitoringToolButton(
        systemName: String? = nil,
        text: String? = nil,
        isActive: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button {
            Haptics.selection()
            action()
        } label: {
            Group {
                if let text {
                    Text(text)
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                } else if let systemName {
                    Image(systemName: systemName)
                        .font(.system(size: 14, weight: .medium))
                }
            }
            .foregroundColor(isActive ? .black : OwLensTheme.textSecondary)
            .frame(width: 38, height: 38)
            .background(
                RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                    .fill(isActive ? OwLensTheme.glassActive : Color.clear)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .frame(width: 44, height: 42)
    }

    // MARK: - Bottom Exposure Deck

    private func bottomExposureDeck(in geo: GeometryProxy) -> some View {
        let safeBottom = ScreenGeometry.bottomPadding(in: geo)

        return HStack(spacing: 6) {
            // ISO & Shutter Angle
            deckTile(
                title: "EXPOSURE",
                value: exposureSummaryValue,
                subvalue: viewModel.shutterSpeedText.isEmpty ? "MANUAL" : viewModel.shutterSpeedText,
                isSelected: viewModel.activePanel == .exposure,
                isDisabled: exposureControlsDisabled,
                width: 98
            ) {
                viewModel.togglePanel(.exposure)
            }

            // White Balance Kelvin
            deckTile(
                title: "WB",
                value: "\(Int(viewModel.wbKelvin))K",
                subvalue: viewModel.isAutoWhiteBalanceEnabled ? "AUTO" : "MANUAL",
                isSelected: viewModel.activePanel == .wb,
                isDisabled: exposureControlsDisabled,
                width: 72
            ) {
                viewModel.togglePanel(.wb)
            }

            // Focus Mode (AF-L vs MF)
            deckTile(
                title: "FOCUS",
                value: viewModel.isFocusLocked ? "AF-L" : (viewModel.isAutoFocus ? "TAP" : "MF"),
                subvalue: viewModel.isFocusLocked ? "LOCKED" : (viewModel.isAutoFocus ? "AF" : String(format: "%.2f", viewModel.focusLensPosition)),
                isSelected: viewModel.activePanel == .focus,
                isDisabled: exposureControlsDisabled,
                width: 64
            ) {
                viewModel.togglePanel(.focus)
            }

            // Codec & Bitrate
            deckTile(
                title: viewModel.selectedCodec == .hevc ? "BITRATE" : "CODEC",
                value: viewModel.selectedCodec == .hevc ? "\(viewModel.selectedBitrate.label)M" : (viewModel.selectedCodec == .proRes422HQ ? "HQ" : "422"),
                subvalue: viewModel.selectedCodec == .hevc ? "HEVC" : "PRORES",
                isSelected: viewModel.activePanel == .bitrate,
                isDisabled: viewModel.isRecording || viewModel.isSaving || viewModel.controlsLocked,
                width: 68
            ) {
                viewModel.togglePanel(.bitrate)
            }

            // Audio Source with Live VU Meter
            audioDeckTile
        }
        .padding(.horizontal, 40)
        .padding(.bottom, safeBottom)
    }

    private var audioDeckTile: some View {
        AudioDeckTile(
            audioMonitor: viewModel.audioMonitor,
            isSelected: viewModel.activePanel == .mic,
            isDisabled: viewModel.isRecording || viewModel.isSaving || viewModel.controlsLocked,
            isMuted: viewModel.selectedAudioSource.portUID == nil,
            micShortName: micShortName
        ) {
            Haptics.selection()
            viewModel.togglePanel(.mic)
        }
    }

    private func deckTile(
        title: String,
        value: String,
        subvalue: String,
        isSelected: Bool,
        isDisabled: Bool,
        width: CGFloat,
        action: @escaping () -> Void
    ) -> some View {
        Button {
            Haptics.selection()
            action()
        } label: {
            VStack(spacing: 2) {
                Text(title)
                    .font(.appMono(.medium, size: 7))
                    .foregroundColor(isSelected ? .black.opacity(0.50) : OwLensTheme.textMuted)
                Text(value)
                    .font(.appMono(.semiBold, size: 12))
                    .foregroundColor(isSelected ? .black : (isDisabled ? OwLensTheme.textDisabled : OwLensTheme.textPrimary))
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
                Text(subvalue)
                    .font(.appMono(.regular, size: 7))
                    .foregroundColor(isSelected ? .black.opacity(0.40) : OwLensTheme.textMuted)
                    .lineLimit(1)
            }
            .frame(width: width, height: 44)
            .background(
                RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                    .fill(isSelected ? OwLensTheme.glassActive : (isDisabled ? OwLensTheme.glassBase.opacity(0.3) : OwLensTheme.glassBaseHeavy))
            )
            .overlay(
                RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                    .strokeBorder(isSelected ? Color.clear : OwLensTheme.glassBorder, lineWidth: 0.5)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
    }

    private var exposureSummaryValue: String {
        "\(Int(viewModel.isoValue))/\(Int(viewModel.shutterValue))°"
    }

    private var exposureControlsDisabled: Bool {
        viewModel.controlsLocked || viewModel.isRecording || viewModel.isSaving || viewModel.isDeviceUnsupportedForLog
    }

    // MARK: - Transient Status & Floating Panels

    private var transientStatusArea: some View {
        VStack(spacing: 6) {
            if let err = viewModel.errorMessage {
                Button {
                    withAnimation {
                        viewModel.errorMessage = nil
                    }
                } label: {
                    messageBadge(icon: "xmark.octagon.fill", text: err, color: OwLensTheme.recordingRed)
                }
                .buttonStyle(.plain)
            }

            if viewModel.thermalState != .nominal {
                messageBadge(icon: "thermometer.medium", text: thermalMessage, color: thermalColor)
            }

            if let panel = viewModel.activePanel, !viewModel.isRecording {
                floatingAdjustmentPanel(panel)
                    .transition(.asymmetric(
                        insertion: .scale(scale: 0.97).combined(with: .opacity),
                        removal: .opacity
                    ))
                    .frame(maxWidth: 560)
            }
        }
        .padding(.horizontal, 80)
    }

    private func messageBadge(icon: String, text: String, color: Color) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .medium))
            Text(text)
                .font(.appFont(.medium, size: 11))
                .lineLimit(1)
        }
        .foregroundColor(color)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .glassPanel(cornerRadius: OwLensTheme.radiusCard, border: color.opacity(0.25), background: OwLensTheme.glassBaseHeavy)
    }

    // MARK: - Floating Adjustment Drawer Panels

    @ViewBuilder
    private func floatingAdjustmentPanel(_ panel: CameraViewModel.ControlPanel) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            switch panel {
            case .exposure, .iso, .shutter:
                exposureDrawerContent

            case .wb:
                whiteBalanceDrawerContent

            case .focus:
                focusDrawerContent

            case .format:
                formatDrawerContent

            case .bitrate:
                bitrateDrawerContent

            case .mic:
                micDrawerContent

            case .save:
                saveDrawerContent

            case .lens:
                lensDrawerContent

            case .logCurve:
                logCurveDrawerContent

            case .denoise:
                denoiseDrawerContent
            }
        }
        .padding(12)
        .glassPanel(cornerRadius: OwLensTheme.radiusLg, border: OwLensTheme.glassBorderActive, background: Color.black.opacity(0.32))
        .shadow(color: Color.black.opacity(0.35), radius: 14, x: 0, y: 5)
    }

    // MARK: - Drawer Sub-views

    private var logCurveDrawerContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            drawerHeader(title: "LOG COLOR PROFILE")

            VStack(spacing: 8) {
                ForEach(LogCurveType.uiCases, id: \.self) { curve in
                    let isSelected = viewModel.selectedCurve == curve
                    Button {
                        Haptics.selection()
                        viewModel.selectedCurve = curve
                        viewModel.showToast("\(curve.displayName) · 10-Bit BT.2020")
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 3) {
                                HStack(spacing: 6) {
                                    Text(curve.displayName)
                                        .font(.appFont(isSelected ? .semiBold : .medium, size: 12))
                                        .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                                    Text("10-BIT")
                                        .font(.appMono(.bold, size: 8))
                                        .foregroundColor(isSelected ? .black.opacity(0.75) : .white)
                                        .padding(.horizontal, 4)
                                        .padding(.vertical, 1.5)
                                        .background(
                                            RoundedRectangle(cornerRadius: 3, style: .continuous)
                                                .fill(isSelected ? Color.black.opacity(0.12) : Color.white.opacity(0.18))
                                        )
                                }
                                Text(curve == .appleLog2 ? "Native Apple Log OETF · ITU-R BT.2020 · 12 stops dynamic range" : "Sony S-Log3 OETF · S-Gamut3.Cine · Standardized Cine EI")
                                    .font(.appFont(.regular, size: 9))
                                    .foregroundColor(isSelected ? .black.opacity(0.60) : OwLensTheme.textSecondary)
                            }
                            Spacer()
                            if isSelected {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.black)
                                    .font(.system(size: 14, weight: .semibold))
                            }
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                                .fill(isSelected ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                                .strokeBorder(isSelected ? Color.clear : OwLensTheme.glassBorder, lineWidth: 0.5)
                        )
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var denoiseDrawerContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            drawerHeader(title: "NOISE REDUCTION")

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("DENOISE STRENGTH")
                        .font(.appMono(.medium, size: 8))
                        .foregroundColor(OwLensTheme.textMuted)
                    Spacer()
                    Text(viewModel.denoiseStrength <= 0.001 ? "OFF" : String(format: "%.2f", viewModel.denoiseStrength))
                        .font(.appMono(.bold, size: 10))
                        .foregroundColor(viewModel.denoiseStrength > 0.001 ? OwLensTheme.accent : OwLensTheme.textMuted)
                }

                HStack(spacing: 10) {
                    Text("0.0")
                        .font(.appMono(.regular, size: 9))
                        .foregroundColor(OwLensTheme.textMuted)

                    Slider(value: $viewModel.denoiseStrength, in: 0.0...1.0, step: 0.05)
                        .tint(OwLensTheme.accent)

                    Text("1.0")
                        .font(.appMono(.regular, size: 9))
                        .foregroundColor(OwLensTheme.textMuted)
                }
            }
        }
    }

    private var exposureDrawerContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            drawerHeader(title: "EXPOSURE")

            VStack(spacing: 8) {
                // ISO Stepped Stop Control
                HStack(spacing: 10) {
                    Text("ISO")
                        .font(.appMono(.semiBold, size: 10))
                        .foregroundColor(OwLensTheme.textSecondary)
                        .frame(width: 36, alignment: .leading)

                    stopStepper(
                        index: $viewModel.isoStopIndex,
                        count: viewModel.isoStops.count,
                        label: String(format: "%.0f", viewModel.isoValue),
                        onNudge: { viewModel.nudgeISO($0) }
                    )
                }

                // Shutter Angle Stepper Control & Cinema Snap Targets
                HStack(spacing: 10) {
                    Text("ANG")
                        .font(.appMono(.semiBold, size: 10))
                        .foregroundColor(OwLensTheme.textSecondary)
                        .frame(width: 36, alignment: .leading)

                    shutterStepper
                }

                // Quick Cinema Angle Presets
                HStack(spacing: 4) {
                    ForEach([180.0, 172.8, 90.0, 45.0], id: \.self) { angle in
                        let isMatch = abs(viewModel.shutterValue - Float(angle)) < 1.0
                        Button {
                            Haptics.selection()
                            viewModel.setShutterAngleWithSnapping(Float(angle))
                        } label: {
                            Text(angle == 180.0 ? "180°" : String(format: "%.1f°", angle))
                                .font(.appMono(isMatch ? .semiBold : .regular, size: 9))
                                .foregroundColor(isMatch ? .black : OwLensTheme.textSecondary)
                                .padding(.horizontal, 7)
                                .padding(.vertical, 4)
                                .background(
                                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                                        .fill(isMatch ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                                 )
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    private var whiteBalanceDrawerContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            drawerHeaderRow(
                title: "WHITE BALANCE",
                isAutoOn: $viewModel.isAutoWhiteBalanceEnabled,
                autoLabel: "AUTO"
            )

            if viewModel.isAutoWhiteBalanceEnabled {
                HStack(spacing: 6) {
                    Circle()
                        .fill(viewModel.isAutoWhiteBalanceAdjusting ? OwLensTheme.amberWarning : OwLensTheme.audioNominal)
                        .frame(width: 5, height: 5)
                    Text("LIVE AWB: \(Int(viewModel.wbKelvin))K · Tint \(Int(viewModel.wbTint))\(viewModel.isAutoWhiteBalanceAdjusting ? " (Adjusting)" : "")")
                        .font(.appMono(.medium, size: 10))
                        .foregroundColor(OwLensTheme.textSecondary)
                }
                .padding(.top, 2)
            } else {
                VStack(spacing: 8) {
                    // Kelvin Stop Stepper
                    stopStepper(
                        index: $viewModel.wbStopIndex,
                        count: viewModel.wbStops.count,
                        label: String(format: "%.0fK", viewModel.wbKelvin),
                        onNudge: { viewModel.nudgeWB($0) }
                    )

                    // Quick Presets with Icons
                    HStack(spacing: 4) {
                        ForEach([
                            ("Tungsten", "lightbulb.fill", Float(3200)),
                            ("Fluorescent", "sun.haze.fill", Float(4000)),
                            ("Daylight", "sun.max.fill", Float(5600)),
                            ("Shade", "cloud.sun.fill", Float(7000))
                        ], id: \.0) { item in
                            let isMatch = abs(viewModel.wbKelvin - item.2) < 150
                            Button {
                                Haptics.selection()
                                viewModel.wbStopIndex = ExposureStops.nearestIndex(in: viewModel.wbStops, to: item.2)
                            } label: {
                                HStack(spacing: 3) {
                                    Image(systemName: item.1)
                                        .font(.system(size: 8))
                                    Text("\(Int(item.2))K")
                                        .font(.appFont(isMatch ? .semiBold : .regular, size: 9))
                                }
                                .foregroundColor(isMatch ? .black : OwLensTheme.textSecondary)
                                .padding(.horizontal, 7)
                                .padding(.vertical, 4)
                                .background(
                                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                                        .fill(isMatch ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                                 )
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }

                    // Tint Stepper Control (Green <-> Magenta)
                    HStack(spacing: 10) {
                        Text("TINT")
                            .font(.appMono(.semiBold, size: 10))
                            .foregroundColor(OwLensTheme.textSecondary)
                            .frame(width: 36, alignment: .leading)

                        tintStepper
                    }
                }
            }
        }
    }

    private var focusDrawerContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            drawerHeader(title: "FOCUS")

            HStack(spacing: 6) {
                Button {
                    Haptics.selection()
                    viewModel.isAutoFocus = true
                    viewModel.isFocusLocked = false
                } label: {
                    Text("TAP AF")
                        .font(.appFont(viewModel.isAutoFocus ? .semiBold : .regular, size: 10))
                        .foregroundColor(viewModel.isAutoFocus ? .black : OwLensTheme.textSecondary)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 6)
                        .background(
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .fill(viewModel.isAutoFocus ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                        )
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

                Button {
                    Haptics.selection()
                    viewModel.isAutoFocus = false
                } label: {
                    Text("MF")
                        .font(.appFont(!viewModel.isAutoFocus ? .semiBold : .regular, size: 10))
                        .foregroundColor(!viewModel.isAutoFocus ? .black : OwLensTheme.textSecondary)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 6)
                        .background(
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .fill(!viewModel.isAutoFocus ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                        )
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

                Spacer()
            }

            if viewModel.isAutoFocus {
                Text(viewModel.isFocusLocked ? "Focus is locked. Tap preview to refocus." : "Tap preview to lock focus on subject.")
                    .font(.appFont(.regular, size: 10.5))
                    .foregroundColor(OwLensTheme.textSecondary)
                    .padding(.top, 2)
            } else {
                HStack(spacing: 10) {
                    Text("LENS")
                        .font(.appMono(.semiBold, size: 10))
                        .foregroundColor(OwLensTheme.textSecondary)
                        .frame(width: 36, alignment: .leading)

                    focusStepper
                }
                .transition(.opacity)
            }
        }
    }

    private var formatDrawerContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            drawerHeader(title: "FORMAT")

            // Resolution Section
            VStack(alignment: .leading, spacing: 5) {
                Text("RESOLUTION")
                    .font(.appMono(.medium, size: 8))
                    .foregroundColor(OwLensTheme.textMuted)

                HStack(spacing: 6) {
                    ForEach(RecordingFormat.allCases) { fmt in
                        let isSelected = viewModel.selectedFormat == fmt
                        Button {
                            Haptics.selection()
                            viewModel.selectedFormat = fmt
                        } label: {
                            VStack(spacing: 3) {
                                Text(fmt.displayName)
                                    .font(.appFont(isSelected ? .semiBold : .regular, size: 11))
                                    .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                                Text(fmt.detailLabel)
                                    .font(.appMono(.regular, size: 9))
                                    .foregroundColor(isSelected ? .black.opacity(0.5) : OwLensTheme.textMuted)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                                    .fill(isSelected ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                                    .strokeBorder(isSelected ? Color.clear : OwLensTheme.glassBorder, lineWidth: 0.5)
                            )
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    private var bitrateDrawerContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            drawerHeader(title: "CODEC & BITRATE")

            // Codec Selection Section
            VStack(alignment: .leading, spacing: 5) {
                Text("CODEC")
                    .font(.appMono(.medium, size: 8))
                    .foregroundColor(OwLensTheme.textMuted)

                HStack(spacing: 6) {
                    ForEach(VideoCodecOption.availableCases) { codec in
                        let isSelected = viewModel.selectedCodec == codec
                        Button {
                            Haptics.selection()
                            viewModel.selectedCodec = codec
                        } label: {
                            VStack(spacing: 2) {
                                Text(codec.displayName)
                                    .font(.appFont(isSelected ? .semiBold : .regular, size: 11))
                                    .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                                Text(codec == .hevc ? "10-bit H.265" : "Intra-frame")
                                    .font(.appMono(.regular, size: 8))
                                    .foregroundColor(isSelected ? .black.opacity(0.5) : OwLensTheme.textMuted)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                                    .fill(isSelected ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                                    .strokeBorder(isSelected ? Color.clear : OwLensTheme.glassBorder, lineWidth: 0.5)
                            )
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }

                if !VideoCodecOption.isProResSupported {
                    Text("ProRes hardware encoding requires iPhone 13 Pro or newer (A15+).")
                        .font(.appFont(.regular, size: 9.5))
                        .foregroundColor(OwLensTheme.textMuted)
                        .padding(.top, 2)
                }
            }

            // Bitrate Section (Active for HEVC; informational for ProRes)
            if viewModel.selectedCodec == .hevc {
                VStack(alignment: .leading, spacing: 5) {
                    Text("HEVC BITRATE")
                        .font(.appMono(.medium, size: 8))
                        .foregroundColor(OwLensTheme.textMuted)

                    HStack(spacing: 6) {
                        ForEach(BitratePreset.allCases.filter { $0.rawValue <= viewModel.selectedFormat.maxBitratePreset.rawValue }) { bit in
                            let isSelected = viewModel.selectedBitrate == bit
                            Button {
                                Haptics.selection()
                                viewModel.selectedBitrate = bit
                            } label: {
                                VStack(spacing: 2) {
                                    Text(bit.displayName)
                                        .font(.appMono(isSelected ? .semiBold : .regular, size: 11))
                                        .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                                    Text(bit == viewModel.selectedFormat.suggestedBitratePreset ? "Rec" : "Target")
                                        .font(.appMono(.regular, size: 8))
                                        .foregroundColor(isSelected ? .black.opacity(0.5) : OwLensTheme.textMuted)
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 8)
                                .background(
                                    RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                                        .fill(isSelected ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                                )
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            } else {
                Text(viewModel.selectedCodec == .proRes422HQ ? "ProRes 422 HQ records at ~235 Mbps intra-frame mastering quality." : "ProRes 422 records at ~156 Mbps intra-frame mastering quality.")
                    .font(.appFont(.regular, size: 10.5))
                    .foregroundColor(OwLensTheme.textSecondary)
                    .padding(.top, 2)
            }
        }
    }

    private var micDrawerContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            drawerHeader(title: "AUDIO INPUT")

            if viewModel.isSwitchingMic {
                HStack(spacing: 8) {
                    ProgressView().tint(.white).scaleEffect(0.8)
                    Text("Configuring audio…")
                        .font(.appFont(.regular, size: 11))
                        .foregroundColor(OwLensTheme.textSecondary)
                }
                .padding(.vertical, 8)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 5) {
                        ForEach(viewModel.audioSources) { src in
                            let isSelected = viewModel.selectedAudioSource.id == src.id
                            Button {
                                Haptics.selection()
                                viewModel.selectedAudioSource = src
                            } label: {
                                HStack(spacing: 5) {
                                    Image(systemName: "mic.fill")
                                        .font(.system(size: 10))
                                    Text(src.name)
                                        .font(.appFont(isSelected ? .semiBold : .regular, size: 11))
                                }
                                .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(
                                    RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                                        .fill(isSelected ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                                )
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
        }
    }

    private var saveDrawerContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            drawerHeader(title: "DESTINATION")

            HStack(spacing: 6) {
                ForEach(VideoSaveDestination.allCases) { dest in
                    let isSelected = viewModel.selectedSaveDestination == dest
                    Button {
                        Haptics.selection()
                        viewModel.chooseSaveDestination(dest)
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: saveDestinationIcon(dest))
                                .font(.system(size: 12))
                            Text(dest.label)
                                .font(.appFont(isSelected ? .semiBold : .regular, size: 11))
                        }
                        .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                                .fill(isSelected ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                        )
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var lensDrawerContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            drawerHeader(title: "LENS")

            HStack(spacing: 6) {
                ForEach(viewModel.availableLenses) { lens in
                    let isSelected = viewModel.selectedLens?.id == lens.id
                    Button {
                        Haptics.selection()
                        viewModel.selectedLens = lens
                    } label: {
                        VStack(spacing: 2) {
                            Text(lens.shortLabel)
                                .font(.appFont(isSelected ? .semiBold : .regular, size: 12))
                                .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                            Text(lens.name)
                                .font(.appFont(.regular, size: 8))
                                .foregroundColor(isSelected ? .black.opacity(0.5) : OwLensTheme.textMuted)
                                .lineLimit(1)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                                .fill(isSelected ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                        )
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.isSwitchingLens)
                }
            }
        }
    }

    // MARK: - Drawer Common Helpers

    private func drawerHeader(title: String) -> some View {
        HStack {
            Text(title)
                .font(.appMono(.semiBold, size: 9))
                .foregroundColor(OwLensTheme.textMuted)
                .tracking(1.5)

            Spacer()

            Button {
                Haptics.selection()
                viewModel.activePanel = nil
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(OwLensTheme.textSecondary)
                    .frame(width: 44, height: 44)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
        .frame(height: 32)
    }

    private func drawerHeaderRow(title: String, isAutoOn: Binding<Bool>, autoLabel: String) -> some View {
        HStack(spacing: 8) {
            Text(title)
                .font(.appMono(.semiBold, size: 9))
                .foregroundColor(OwLensTheme.textMuted)
                .tracking(1.5)

            Spacer()

            Button {
                Haptics.selection()
                isAutoOn.wrappedValue.toggle()
            } label: {
                HStack(spacing: 4) {
                    Circle()
                        .fill(isAutoOn.wrappedValue ? OwLensTheme.audioNominal : OwLensTheme.textDisabled)
                        .frame(width: 6, height: 6)
                    Text(autoLabel)
                        .font(.appMono(.bold, size: 9))
                        .foregroundColor(isAutoOn.wrappedValue ? OwLensTheme.textPrimary : OwLensTheme.textMuted)
                }
                .padding(.horizontal, 9)
                .padding(.vertical, 5)
                .background(
                    Capsule()
                        .fill(isAutoOn.wrappedValue ? OwLensTheme.glassActiveBg : OwLensTheme.glassBase)
                )
                .overlay(
                    Capsule()
                        .strokeBorder(isAutoOn.wrappedValue ? OwLensTheme.glassBorderActive : OwLensTheme.glassBorder, lineWidth: 0.5)
                )
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            Button {
                Haptics.selection()
                viewModel.activePanel = nil
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(OwLensTheme.textSecondary)
                    .frame(width: 44, height: 44)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
        .frame(height: 32)
    }

    private func stopStepper(
        index: Binding<Int>,
        count: Int,
        label: String,
        onNudge: @escaping (Int) -> Void
    ) -> some View {
        let maxIndex = max(0, count - 1)
        return HStack(spacing: 8) {
            Button {
                Haptics.impact(.light)
                onNudge(-1)
            } label: {
                Image(systemName: "minus")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(OwLensTheme.textPrimary)
                    .frame(width: 32, height: 32)
                    .background(Circle().fill(OwLensTheme.glassBase))
                    .overlay(Circle().strokeBorder(OwLensTheme.glassBorder, lineWidth: 0.5))
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .frame(width: 44, height: 44)
            .disabled(index.wrappedValue <= 0)

            Slider(
                value: Binding(
                    get: { Double(index.wrappedValue) },
                    set: { index.wrappedValue = Int($0.rounded()) }
                ),
                in: 0...Double(maxIndex),
                step: 1
            )
            .tint(OwLensTheme.textPrimary)

            Button {
                Haptics.impact(.light)
                onNudge(1)
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(OwLensTheme.textPrimary)
                    .frame(width: 32, height: 32)
                    .background(Circle().fill(OwLensTheme.glassBase))
                    .overlay(Circle().strokeBorder(OwLensTheme.glassBorder, lineWidth: 0.5))
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .frame(width: 44, height: 44)
            .disabled(index.wrappedValue >= maxIndex)

            Text(label)
                .font(.appMono(.semiBold, size: 13))
                .foregroundColor(OwLensTheme.textPrimary)
                .frame(width: 58, alignment: .trailing)
        }
    }

    private var shutterStepper: some View {
        HStack(spacing: 8) {
            Button {
                Haptics.impact(.light)
                viewModel.nudgeShutterAngle(-1)
            } label: {
                Image(systemName: "minus")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(OwLensTheme.textPrimary)
                    .frame(width: 32, height: 32)
                    .background(Circle().fill(OwLensTheme.glassBase))
                    .overlay(Circle().strokeBorder(OwLensTheme.glassBorder, lineWidth: 0.5))
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .frame(width: 44, height: 44)
            .disabled(viewModel.shutterValue <= viewModel.shutterRange.lowerBound)

            Slider(value: Binding(get: { viewModel.shutterValue }, set: {
                viewModel.setShutterAngleWithSnapping($0)
            }), in: viewModel.shutterRange)
            .tint(OwLensTheme.textPrimary)

            Button {
                Haptics.impact(.light)
                viewModel.nudgeShutterAngle(1)
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(OwLensTheme.textPrimary)
                    .frame(width: 32, height: 32)
                    .background(Circle().fill(OwLensTheme.glassBase))
                    .overlay(Circle().strokeBorder(OwLensTheme.glassBorder, lineWidth: 0.5))
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .frame(width: 44, height: 44)
            .disabled(viewModel.shutterValue >= viewModel.shutterRange.upperBound)

            VStack(alignment: .trailing, spacing: 1) {
                Text(String(format: "%.0f°", viewModel.shutterValue))
                    .font(.appMono(.semiBold, size: 12))
                    .foregroundColor(OwLensTheme.textPrimary)
                Text(viewModel.shutterSpeedText)
                    .font(.appMono(.regular, size: 8.5))
                    .foregroundColor(OwLensTheme.textSecondary)
            }
            .frame(width: 58, alignment: .trailing)
        }
    }

    private var tintStepper: some View {
        HStack(spacing: 8) {
            Button {
                Haptics.impact(.light)
                viewModel.nudgeTint(-5)
            } label: {
                Image(systemName: "minus")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(OwLensTheme.textPrimary)
                    .frame(width: 32, height: 32)
                    .background(Circle().fill(OwLensTheme.glassBase))
                    .overlay(Circle().strokeBorder(OwLensTheme.glassBorder, lineWidth: 0.5))
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .frame(width: 44, height: 44)
            .disabled(viewModel.wbTint <= -50)

            Slider(value: $viewModel.wbTint, in: -50.0...50.0, step: 1.0)
                .tint(OwLensTheme.textPrimary)

            Button {
                Haptics.impact(.light)
                viewModel.nudgeTint(5)
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(OwLensTheme.textPrimary)
                    .frame(width: 32, height: 32)
                    .background(Circle().fill(OwLensTheme.glassBase))
                    .overlay(Circle().strokeBorder(OwLensTheme.glassBorder, lineWidth: 0.5))
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .frame(width: 44, height: 44)
            .disabled(viewModel.wbTint >= 50)

            Text(String(format: "%+.0f", viewModel.wbTint))
                .font(.appMono(.semiBold, size: 12))
                .foregroundColor(viewModel.wbTint == 0 ? OwLensTheme.textSecondary : (viewModel.wbTint > 0 ? Color(red: 1.0, green: 0.45, blue: 0.75) : Color(red: 0.4, green: 0.9, blue: 0.5)))
                .frame(width: 58, alignment: .trailing)
        }
    }

    private var focusStepper: some View {
        HStack(spacing: 8) {
            Button {
                Haptics.impact(.light)
                viewModel.nudgeFocus(-1)
            } label: {
                Image(systemName: "minus")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(OwLensTheme.textPrimary)
                    .frame(width: 32, height: 32)
                    .background(Circle().fill(OwLensTheme.glassBase))
                    .overlay(Circle().strokeBorder(OwLensTheme.glassBorder, lineWidth: 0.5))
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .frame(width: 44, height: 44)
            .disabled(viewModel.focusLensPosition <= 0.0)

            Slider(value: Binding(get: { viewModel.focusLensPosition }, set: {
                viewModel.isAutoFocus = false
                viewModel.focusLensPosition = $0
            }), in: 0.0...1.0)
            .tint(OwLensTheme.textPrimary)

            Button {
                Haptics.impact(.light)
                viewModel.nudgeFocus(1)
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(OwLensTheme.textPrimary)
                    .frame(width: 32, height: 32)
                    .background(Circle().fill(OwLensTheme.glassBase))
                    .overlay(Circle().strokeBorder(OwLensTheme.glassBorder, lineWidth: 0.5))
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .frame(width: 44, height: 44)
            .disabled(viewModel.focusLensPosition >= 1.0)

            Text(String(format: "%.2f", viewModel.focusLensPosition))
                .font(.appMono(.semiBold, size: 12))
                .foregroundColor(OwLensTheme.textPrimary)
                .frame(width: 58, alignment: .trailing)
        }
    }

    // MARK: - Right Record Grip

    private func rightRecordGrip(in geo: GeometryProxy) -> some View {
        let safeTrailing = ScreenGeometry.trailingPadding(in: geo)

        return ZStack(alignment: .trailing) {
            // Shutter / Record Trigger
            recordButton
                .frame(maxHeight: .infinity, alignment: .center)

            // Scopes / Histogram Monitor - Positioned directly ABOVE the centered record button
            if viewModel.showScopes {
                ScopesContainerView(monitor: viewModel.scopeMonitor)
                    .alignmentGuide(VerticalAlignment.center) { d in
                        d[.bottom] + 46 // 34pt (half record button) + 12pt clearance gap
                    }
                    .transition(.asymmetric(
                        insertion: .opacity.combined(with: .scale(scale: 0.95)),
                        removal: .opacity
                    ))
            }
        }
        .padding(.trailing, safeTrailing)
    }

    // MARK: - Lens Switcher & Zoom Wheel Layer (Underneath all other UI)

    private func lensAndZoomWheelLayer(in geo: GeometryProxy) -> some View {
        let safeTrailing = ScreenGeometry.trailingPadding(in: geo)

        return HStack(spacing: 0) {
            Spacer()
            LensAndZoomWheelControl(viewModel: viewModel, zoomMonitor: viewModel.zoomMonitor)
                .padding(.trailing, safeTrailing + 104 + 12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
    }

    private var recordButton: some View {
        Button {
            viewModel.toggleRecording()
        } label: {
            ZStack {
                // Outer Ring with subtle pulsing glow when recording
                Circle()
                    .strokeBorder(viewModel.isRecording ? OwLensTheme.recordingRed : OwLensTheme.textPrimary, lineWidth: 2.5)
                    .frame(width: 64, height: 64)
                    .shadow(color: viewModel.isRecording ? OwLensTheme.recordingRed.opacity(0.65) : Color.black.opacity(0.3), radius: viewModel.isRecording ? 10 : 3)

                if viewModel.isSaving {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .scaleEffect(1.2)
                } else if viewModel.isRecording {
                    // Red Stop Square
                    RoundedRectangle(cornerRadius: 5, style: .continuous)
                        .fill(OwLensTheme.recordingRed)
                        .frame(width: 22, height: 22)
                } else {
                    // Inner Record Circle
                    Circle()
                        .fill(
                            viewModel.isDeviceUnsupportedForLog
                                ? Color.gray.opacity(0.3)
                                : OwLensTheme.recordingRed
                        )
                        .frame(width: 48, height: 48)
                }
            }
            .frame(width: 104, height: 68)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(viewModel.isDeviceUnsupportedForLog || viewModel.isSaving)
    }

    // MARK: - Utilities & Formatters

    private func toggleSaveDestination() {
        guard !viewModel.isRecording else { return }
        let next: VideoSaveDestination = viewModel.selectedSaveDestination == .photos ? .files : .photos
        viewModel.chooseSaveDestination(next)
    }

    private func saveDestinationIcon(_ destination: VideoSaveDestination) -> String {
        switch destination {
        case .photos: return "photo.on.rectangle"
        case .files: return "folder"
        }
    }

    private var micShortName: String {
        let name = viewModel.selectedAudioSource.name
        if name.isEmpty || viewModel.selectedAudioSource.portUID == nil { return "MIC" }
        if name == "Built-in Mic" || name == "iPhone" || name == "Built-In Microphone" { return "BUILT-IN" }
        if name.count <= 8 { return name.uppercased() }
        return String(name.prefix(7)).uppercased() + "…"
    }

    private var thermalColor: Color {
        switch viewModel.thermalState {
        case .fair: return OwLensTheme.textSecondary
        case .serious: return OwLensTheme.textPrimary
        case .critical: return OwLensTheme.recordingRed
        default: return OwLensTheme.textMuted
        }
    }

    private var thermalMessage: String {
        switch viewModel.thermalState {
        case .fair: return "Device Warming"
        case .serious: return "Thermal Throttling Imminent"
        case .critical: return "Critical Temperature"
        default: return ""
        }
    }
}

// MARK: - iPhone Cinema Lens Switcher & Smooth Circular Zoom Wheel

private struct LensAndZoomWheelControl: View {
    @ObservedObject var viewModel: CameraViewModel
    @ObservedObject var zoomMonitor: ZoomMonitor

    @State private var isArcDragging: Bool = false
    @State private var arcDragInitialY: CGFloat = 0.0
    @State private var arcDragStartZoom: CGFloat = 1.0
    @State private var arcLastHapticStep: Int = 0

    /// Lenses sorted descending so Tele (2x) is on TOP, Wide (1x) in MIDDLE, Ultra-Wide (.5) on BOTTOM.
    private var orderedLenses: [LensOption] {
        viewModel.availableLenses.sorted(by: { $0.baseMultiplier > $1.baseMultiplier })
    }

    private var isExpanded: Bool {
        zoomMonitor.isSwitcherExpanded
    }

    var body: some View {
        HStack(alignment: .center, spacing: 6) {
            if isExpanded || zoomMonitor.isWheelActive {
                tinyZoomReadout
                    .transition(.opacity)

                circularArcWheel
                    .transition(.opacity.combined(with: .scale(scale: 0.95)))
            }

            verticalLensButtons
        }
        .animation(.spring(response: 0.28, dampingFraction: 0.84), value: isExpanded)
        .animation(.spring(response: 0.28, dampingFraction: 0.84), value: zoomMonitor.isWheelActive)
        .onChange(of: viewModel.activePanel) { panel in
            if panel != nil && isExpanded {
                withAnimation(.spring(response: 0.24, dampingFraction: 0.85)) {
                    zoomMonitor.hideWheelAndSwitcher()
                }
            }
        }
    }

    // MARK: - Vertical Lens Buttons

    private var verticalLensButtons: some View {
        VStack(spacing: 4) {
            ForEach(orderedLenses) { lens in
                let isSelected = (viewModel.selectedLens?.id == lens.id) || (viewModel.selectedLens == nil && lens.baseMultiplier == 1.0)
                if isExpanded || isSelected {
                    LensButtonView(
                        lens: lens,
                        isSelected: isSelected,
                        viewModel: viewModel,
                        zoomMonitor: zoomMonitor,
                        onTap: {
                            if isSelected {
                                zoomMonitor.toggleSwitcher()
                            } else {
                                handleSelectOtherLens(lens)
                            }
                        },
                        onDragStart: {
                            zoomMonitor.cancelDismiss()
                            if !zoomMonitor.isSwitcherExpanded {
                                withAnimation(.spring(response: 0.28, dampingFraction: 0.84)) {
                                    zoomMonitor.showWheelAndSwitcher()
                                }
                            }
                        },
                        onDragEnd: {
                            zoomMonitor.scheduleDismiss(delaySeconds: 2.5)
                        }
                    )
                    .transition(
                        .asymmetric(
                            insertion: .opacity.combined(with: .scale(scale: 0.5)),
                            removal: .opacity.combined(with: .scale(scale: 0.5))
                        )
                    )
                }
            }
        }
        .padding(.horizontal, 3)
        .padding(.vertical, 4)
        .background(
            Capsule()
                .fill(OwLensTheme.glassBaseHeavy)
        )
        .overlay(
            Capsule()
                .strokeBorder(OwLensTheme.glassBorder, lineWidth: 0.5)
        )
        .animation(.spring(response: 0.28, dampingFraction: 0.84), value: isExpanded)
        .animation(.spring(response: 0.28, dampingFraction: 0.84), value: viewModel.selectedLens?.id)
    }

    private func handleSelectOtherLens(_ lens: LensOption) {
        guard !viewModel.isSwitchingLens else { return }

        if viewModel.isRecording {
            Haptics.notification(.warning)
            viewModel.activeToast = "Lens locked during recording"
            zoomMonitor.scheduleDismiss(delaySeconds: 3.5)
        } else {
            Haptics.selection()
            withAnimation(.spring(response: 0.28, dampingFraction: 0.84)) {
                viewModel.selectedLens = lens
                zoomMonitor.resetZoom()
            }
            zoomMonitor.scheduleDismiss(delaySeconds: 3.5)
        }
    }

    // MARK: - Tiny Zoom Readout

    private var tinyZoomReadout: some View {
        Button {
            Haptics.selection()
            if zoomMonitor.zoomFactor > 1.02 {
                withAnimation(.spring(response: 0.25, dampingFraction: 0.85)) {
                    zoomMonitor.resetZoom()
                }
                zoomMonitor.scheduleDismiss(delaySeconds: 3.5)
            } else {
                withAnimation(.spring(response: 0.25, dampingFraction: 0.85)) {
                    zoomMonitor.hideWheelAndSwitcher()
                }
            }
        } label: {
            VStack(alignment: .trailing, spacing: 1) {
                Text(zoomMonitor.effectiveZoomString(baseMultiplier: viewModel.selectedLens?.baseMultiplier ?? 1.0))
                    .font(.appMono(.bold, size: 12))
                    .foregroundColor(OwLensTheme.accent)
                    .shadow(color: Color.black.opacity(0.8), radius: 2)

                Text(zoomMonitor.equivalentFocalLengthString(deviceType: viewModel.selectedLens?.deviceType))
                    .font(.appMono(.medium, size: 9))
                    .foregroundColor(.white.opacity(0.70))
                    .shadow(color: Color.black.opacity(0.8), radius: 2)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Circular Arc Wheel (With ample margins - no left clipping)

    private var circularArcWheel: some View {
        ZStack {
            Canvas { context, size in
                let width = size.width
                let height = size.height
                let cy = height * 0.5
                let radius: CGFloat = 110.0
                // Center cx is at 140, so apex (cx - radius) is at X = 30.0 pt (comfortably inside width 96)
                let cx: CGFloat = 140.0

                // 1. Arc background track
                var arcPath = Path()
                let maxAngleDeg: CGFloat = 36.0
                let startAngle = Angle.degrees(Double(180.0 - maxAngleDeg))
                let endAngle = Angle.degrees(Double(180.0 + maxAngleDeg))
                arcPath.addArc(
                    center: CGPoint(x: cx, y: cy),
                    radius: radius,
                    startAngle: startAngle,
                    endAngle: endAngle,
                    clockwise: false
                )
                context.stroke(
                    arcPath,
                    with: .color(Color.white.opacity(0.15)),
                    lineWidth: 1.0
                )

                // 2. Ticks from 1.0 to 2.0 at 0.05 step
                let currentZoom = zoomMonitor.zoomFactor
                let anglePerUnit: CGFloat = 65.0

                for i in 0...20 {
                    let tickVal = 1.0 + CGFloat(i) * 0.05
                    let deltaThetaDeg = (tickVal - currentZoom) * anglePerUnit
                    guard abs(deltaThetaDeg) <= maxAngleDeg else { continue }

                    let rad = deltaThetaDeg * .pi / 180.0
                    let cosR = cos(rad)
                    let sinR = sin(rad)

                    let isKeyStop = (i == 0 || i == 10 || i == 20)
                    let isTenth = (i % 2 == 0)

                    let tickLength: CGFloat = isKeyStop ? 12.0 : (isTenth ? 8.0 : 5.0)
                    let strokeWidth: CGFloat = isKeyStop ? 2.0 : (isTenth ? 1.4 : 1.0)

                    let pOuter = CGPoint(
                        x: cx - radius * cosR,
                        y: cy - radius * sinR
                    )
                    let pInner = CGPoint(
                        x: cx - (radius - tickLength) * cosR,
                        y: cy - (radius - tickLength) * sinR
                    )

                    let normDist = abs(deltaThetaDeg) / maxAngleDeg
                    let edgeFade = max(0.0, 1.0 - pow(normDist, 2.5))

                    var tickPath = Path()
                    tickPath.move(to: pOuter)
                    tickPath.addLine(to: pInner)

                    let tickColor: Color
                    if isKeyStop {
                        tickColor = Color.white.opacity(edgeFade * 0.95)
                    } else if isTenth {
                        tickColor = Color.white.opacity(edgeFade * 0.65)
                    } else {
                        tickColor = Color.white.opacity(edgeFade * 0.35)
                    }

                    context.stroke(
                        tickPath,
                        with: .color(tickColor),
                        style: StrokeStyle(lineWidth: strokeWidth, lineCap: .round)
                    )
                }
            }
            .frame(width: 96, height: 210)

            // Fixed Center Needle Indicator (Apex at X = 30; needle runs from X = 12 to X = 28)
            HStack(spacing: 0) {
                Spacer().frame(width: 12)

                Circle()
                    .fill(OwLensTheme.accent)
                    .frame(width: 4, height: 4)
                    .shadow(color: OwLensTheme.accent.opacity(0.8), radius: 3)

                Rectangle()
                    .fill(OwLensTheme.accent)
                    .frame(width: 14, height: 2.5)
                    .cornerRadius(1.25)
                    .shadow(color: OwLensTheme.accent.opacity(0.6), radius: 2)

                Spacer(minLength: 0)
            }
            .frame(width: 96)
        }
        .frame(width: 96, height: 210)
        .contentShape(Rectangle())
        .gesture(arcWheelGesture)
    }

    // MARK: - Arc Wheel Drag Gesture

    private var arcWheelGesture: some Gesture {
        DragGesture(minimumDistance: 1, coordinateSpace: .global)
            .onChanged { value in
                if !isArcDragging {
                    isArcDragging = true
                    arcDragInitialY = value.location.y
                    arcDragStartZoom = zoomMonitor.zoomFactor
                    arcLastHapticStep = Int(round(arcDragStartZoom * 20))
                    zoomMonitor.cancelDismiss()
                }

                // PULLING DOWN ZOOMS IN (linear, direct, smooth, progressive):
                let deltaY = value.location.y - arcDragInitialY
                let sensitivity: CGFloat = 1.0 / 220.0
                let targetZoom = max(1.0, min(2.0, arcDragStartZoom + deltaY * sensitivity))
                zoomMonitor.setZoom(targetZoom)

                let step = Int(round(targetZoom * 20))
                if step != arcLastHapticStep {
                    Haptics.selection()
                    arcLastHapticStep = step
                }
            }
            .onEnded { _ in
                isArcDragging = false
                zoomMonitor.scheduleDismiss(delaySeconds: 2.5)
            }
    }
}

// MARK: - Isolated Lens Button (Seamless Tap & Drag Gesture Handling)

private struct LensButtonView: View {
    let lens: LensOption
    let isSelected: Bool
    @ObservedObject var viewModel: CameraViewModel
    @ObservedObject var zoomMonitor: ZoomMonitor
    let onTap: () -> Void
    let onDragStart: () -> Void
    let onDragEnd: () -> Void

    @State private var touchStartTime: CFTimeInterval = 0.0
    @State private var touchStartLocation: CGPoint = .zero
    @State private var isDragEngaged: Bool = false
    @State private var dragAnchorY: CGFloat = 0.0
    @State private var dragStartZoom: CGFloat = 1.0
    @State private var lastHapticStep: Int = 0

    private var isZoomed: Bool {
        isSelected && zoomMonitor.zoomFactor > 1.02
    }

    private var labelText: String {
        if isZoomed {
            return zoomMonitor.effectiveZoomString(baseMultiplier: lens.baseMultiplier)
        } else if lens.baseMultiplier == 0.5 {
            return "0.5"
        } else if lens.baseMultiplier == 1.0 {
            return "1"
        } else if lens.baseMultiplier == 2.0 {
            return "2"
        } else {
            return lens.shortLabel.replacingOccurrences(of: "×", with: "")
        }
    }

    var body: some View {
        ZStack {
            if isZoomed {
                Circle()
                    .fill(Color.black.opacity(0.85))
                    .overlay(
                        Circle()
                            .strokeBorder(OwLensTheme.accent, lineWidth: 1.5)
                    )
                    .shadow(color: OwLensTheme.accent.opacity(0.4), radius: 3)

                Text(labelText)
                    .font(.appMono(.bold, size: 9))
                    .foregroundColor(OwLensTheme.accent)
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
            } else if isSelected {
                Circle()
                    .fill(Color.white.opacity(0.24))
                    .overlay(
                        Circle()
                            .strokeBorder(Color.white.opacity(0.45), lineWidth: 0.8)
                    )

                Text(labelText)
                    .font(.appFont(.bold, size: 11))
                    .foregroundColor(.white)
                    .lineLimit(1)
            } else {
                Circle()
                    .fill(OwLensTheme.glassBase)
                    .overlay(
                        Circle()
                            .strokeBorder(OwLensTheme.glassBorder, lineWidth: 0.5)
                    )

                Text(labelText)
                    .font(.appFont(.medium, size: 11))
                    .foregroundColor(OwLensTheme.textSecondary)
                    .lineLimit(1)
            }
        }
        .frame(width: 32, height: 32)
        .contentShape(Circle())
        .gesture(lensGesture)
    }

    private var lensGesture: some Gesture {
        DragGesture(minimumDistance: 0, coordinateSpace: .global)
            .onChanged { value in
                let now = CACurrentMediaTime()
                if touchStartTime == 0 {
                    touchStartTime = now
                    touchStartLocation = value.location
                    dragStartZoom = zoomMonitor.zoomFactor
                    dragAnchorY = value.location.y
                    lastHapticStep = Int(round(dragStartZoom * 20))
                }

                let deltaY = value.location.y - touchStartLocation.y

                // Drag engages immediately when finger moves by 3pt (~0.75 mm), or on hold (> 0.25s)
                if !isDragEngaged {
                    if abs(deltaY) >= 3.0 || (now - touchStartTime >= 0.25) {
                        isDragEngaged = true
                        dragAnchorY = value.location.y // ZERO jump: anchor precisely where drag begins!
                        dragStartZoom = zoomMonitor.zoomFactor
                        lastHapticStep = Int(round(dragStartZoom * 20))
                        onDragStart()
                    }
                }

                if isDragEngaged {
                    // PULLING DOWN ZOOMS IN (linear, direct, smooth, progressive):
                    let pullDelta = value.location.y - dragAnchorY
                    let sensitivity: CGFloat = 1.0 / 220.0
                    let targetZoom = max(1.0, min(2.0, dragStartZoom + pullDelta * sensitivity))
                    zoomMonitor.setZoom(targetZoom)

                    let step = Int(round(targetZoom * 20))
                    if step != lastHapticStep {
                        Haptics.selection()
                        lastHapticStep = step
                    }
                }
            }
            .onEnded { value in
                let now = CACurrentMediaTime()
                let totalDelta = hypot(value.location.x - touchStartLocation.x, value.location.y - touchStartLocation.y)
                let duration = now - touchStartTime

                if isDragEngaged {
                    isDragEngaged = false
                    onDragEnd()
                } else if totalDelta < 8.0 && duration < 0.6 {
                    onTap()
                }

                touchStartTime = 0
                touchStartLocation = .zero
            }
    }
}

/// Isolated scopes container that re-renders only its contents on 10 Hz scope updates,
/// shielding the parent HUD from redundant body evaluations.
private struct ScopesContainerView: View {
    @ObservedObject var monitor: ScopeMonitor

    var body: some View {
        ScopesOverlay(data: monitor.scopeData)
    }
}

/// Isolated audio deck tile that re-renders only when the live VU meter updates,
/// shielding the parent HUD from 50–100 Hz CoreAudio buffer invalidations.
private struct AudioDeckTile: View {
    @ObservedObject var audioMonitor: AudioMonitor
    let isSelected: Bool
    let isDisabled: Bool
    let isMuted: Bool
    let micShortName: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 2) {
                HStack(spacing: 3) {
                    Text("AUDIO")
                        .font(.appMono(.medium, size: 7))
                        .foregroundColor(isSelected ? .black.opacity(0.50) : OwLensTheme.textMuted)
                    if !isMuted {
                        Circle()
                            .fill(audioMonitor.level > 0.85 ? OwLensTheme.audioPeak : (audioMonitor.level > 0.60 ? OwLensTheme.audioWarning : OwLensTheme.audioNominal))
                            .frame(width: 3.5, height: 3.5)
                    }
                }

                Text(micShortName)
                    .font(.appMono(.semiBold, size: 12))
                    .foregroundColor(isSelected ? .black : (isDisabled ? OwLensTheme.textDisabled : OwLensTheme.textPrimary))
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)

                if isMuted {
                    Text("MUTED")
                        .font(.appMono(.regular, size: 7))
                        .foregroundColor(isSelected ? .black.opacity(0.40) : OwLensTheme.textMuted)
                } else {
                    // Mini live VU meter bar
                    GeometryReader { geo in
                        let fillWidth = max(2, min(geo.size.width, geo.size.width * CGFloat(audioMonitor.level)))
                        ZStack(alignment: .leading) {
                            Capsule()
                                .fill(Color.white.opacity(0.15))
                            
                            // Full-span gradient masked to level so low levels stay purely green
                            LinearGradient(
                                colors: [OwLensTheme.audioNominal, OwLensTheme.audioWarning, OwLensTheme.audioPeak],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                            .mask(
                                HStack {
                                    Rectangle()
                                        .frame(width: fillWidth)
                                    Spacer(minLength: 0)
                                }
                            )
                            .clipShape(Capsule())
                        }
                    }
                    .frame(height: 2.5)
                    .padding(.horizontal, 6)
                }
            }
            .frame(width: 72, height: 44)
            .background(
                RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                    .fill(isSelected ? OwLensTheme.glassActive : (isDisabled ? OwLensTheme.glassBase.opacity(0.3) : OwLensTheme.glassBaseHeavy))
            )
            .overlay(
                RoundedRectangle(cornerRadius: OwLensTheme.radiusCard, style: .continuous)
                    .strokeBorder(isSelected ? Color.clear : OwLensTheme.glassBorder, lineWidth: 0.5)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
    }
}

