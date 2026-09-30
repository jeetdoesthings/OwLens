import SwiftUI
import AVFoundation

/// Landscape cinema-camera HUD layered over the live viewfinder.
/// Monochromatic redesign: every element earns its pixel.
struct ControlsView: View {
    @ObservedObject var viewModel: CameraViewModel

    var body: some View {
        ZStack {
            // Top HUD Status Bar
            topStatusBar
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)

            // Left Monitoring Tools Rail
            leftToolRail
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)

            // Bottom Exposure Deck & Transient Panels
            VStack(spacing: 8) {
                Spacer(minLength: 0)
                transientStatusArea
                bottomExposureDeck
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)

            // Right Record Grip (Record button + Scopes below)
            rightRecordGrip
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .trailing)

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
                .padding(.top, 48)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                .transition(.move(edge: .top).combined(with: .opacity))
                .zIndex(30)
            }
        }
        .opacity(viewModel.isHUDHidden ? 0 : 1)
        .animation(.easeInOut(duration: 0.22), value: viewModel.isHUDHidden)
        .animation(.spring(response: 0.28, dampingFraction: 0.82), value: viewModel.activePanel)
        .animation(.spring(response: 0.28, dampingFraction: 0.82), value: viewModel.isRecording)
        .animation(.spring(response: 0.28, dampingFraction: 0.82), value: viewModel.controlsLocked)
        .animation(.spring(response: 0.28, dampingFraction: 0.82), value: viewModel.showScopes)
    }

    // MARK: - Top Status Bar

    private var topStatusBar: some View {
        HStack(spacing: 8) {
            leftStatusGroup
            Spacer(minLength: 4)
            centerStatusGroup
            Spacer(minLength: 4)
            rightStatusGroup
        }
        .padding(.leading, 20)
        .padding(.trailing, viewModel.showScopes ? 126 : 28)
        .padding(.top, 10)
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

            // Live Device Battery Gauge with Percentage Beside
            batteryIndicator

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

    private var batteryIndicator: some View {
        let level = viewModel.batteryLevel
        let isCharging = viewModel.isBatteryCharging
        let percent = Int(max(0, min(100, (level * 100).rounded())))
        let isLow = level < 0.20
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
                        .frame(width: 22, height: 12)

                    RoundedRectangle(cornerRadius: 1.5, style: .continuous)
                        .fill(statusColor)
                        .frame(width: max(2, CGFloat(level) * 18), height: 8)
                        .padding(.leading, 2)
                }

                // Terminal nub
                Capsule()
                    .fill(statusColor.opacity(0.50))
                    .frame(width: 1.5, height: 5)
                    .offset(x: 0.5)
            }

            HStack(spacing: 1) {
                if isCharging {
                    Image(systemName: "bolt.fill")
                        .font(.system(size: 7, weight: .bold))
                        .foregroundColor(OwLensTheme.audioNominal)
                }
                Text("\(percent)%")
                    .font(.appMono(.bold, size: 9))
                    .foregroundColor(isLow ? OwLensTheme.recordingRed : OwLensTheme.textPrimary)
            }
        }
        .frame(height: 30)
        .padding(.horizontal, 4)
        .lineLimit(1)
        .fixedSize(horizontal: true, vertical: false)
    }

    private var centerStatusGroup: some View {
        HStack(spacing: 6) {
            // Lens Switcher Pill
            lensSwitcherPill

            // Format Button (OG / 1080 - single tap cycles, long-press opens drawer)
            HStack(spacing: 0) {
                Text(viewModel.selectedFormat.shortLabel)
                    .font(.appFont(.bold, size: 12))
                    .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textPrimary)
                    .lineLimit(1)
                    .fixedSize()
            }
            .padding(.horizontal, 8)
            .frame(minWidth: 38)
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

            // FPS Button (24 / 30 - single tap cycles, long-press opens drawer)
            HStack(spacing: 0) {
                Text(viewModel.selectedFPS.label)
                    .font(.appMono(.bold, size: 12))
                    .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textPrimary)
                    .lineLimit(1)
                    .fixedSize()
            }
            .padding(.horizontal, 8)
            .frame(minWidth: 32)
            .frame(height: 30)
            .glassPanel(cornerRadius: OwLensTheme.radiusCard)
            .contentShape(Rectangle())
            .onTapGesture {
                guard !isTopLocked else { return }
                Haptics.selection()
                viewModel.cycleFPS()
            }
            .onLongPressGesture(minimumDuration: 0.45) {
                guard !isTopLocked else { return }
                Haptics.impact(.medium)
                viewModel.togglePanel(.fps)
            }
            .disabled(isTopLocked)
            .opacity(isTopLocked ? 0.4 : 1.0)

            // Log Profile Selector Button (Single tap cycles, long-press opens drawer)
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
    }

    private var lensSwitcherPill: some View {
        HStack(spacing: 2) {
            ForEach(viewModel.availableLenses) { lens in
                let isSelected = viewModel.selectedLens?.id == lens.id
                Button {
                    Haptics.selection()
                    viewModel.selectedLens = lens
                } label: {
                    Text(lens.shortLabel)
                        .font(.appFont(isSelected ? .semiBold : .regular, size: 11))
                        .foregroundColor(isSelected ? (isTopLocked ? .white : .black) : (isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textSecondary))
                        .frame(minWidth: 32)
                        .frame(height: 24)
                        .padding(.horizontal, 4)
                        .background(
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .fill(isSelected ? (isTopLocked ? OwLensTheme.glassBaseHeavy : OwLensTheme.glassActive) : Color.clear)
                        )
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(isTopLocked || viewModel.isSwitchingLens)
            }
        }
        .padding(3)
        .frame(height: 30)
        .glassPanel(cornerRadius: OwLensTheme.radiusCard)
        .opacity(isTopLocked ? 0.4 : (viewModel.isSwitchingLens ? 0.5 : 1.0))
        .animation(.easeInOut(duration: 0.15), value: viewModel.isSwitchingLens)
    }

    private var rightStatusGroup: some View {
        HStack(spacing: 6) {
            // Save Destination (Photos vs Files - tap cycles, long-press opens drawer)
            HStack {
                Image(systemName: saveDestinationIcon(viewModel.selectedSaveDestination))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textSecondary)
            }
            .frame(width: 32, height: 30)
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

    private var leftToolRail: some View {
        VStack(spacing: 4) {
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
        .padding(.leading, 20)
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

    private var bottomExposureDeck: some View {
        HStack(spacing: 6) {
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

            // Bitrate
            deckTile(
                title: "BITRATE",
                value: "\(viewModel.selectedBitrate.label)M",
                subvalue: viewModel.selectedCodec.rawValue.uppercased(),
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
        .padding(.bottom, 20)
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

            case .format, .fps:
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
                        .fill(OwLensTheme.audioNominal)
                        .frame(width: 5, height: 5)
                    Text("LIVE AWB: \(Int(viewModel.wbKelvin))K · Tint \(Int(viewModel.wbTint))")
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

            // Frame Rate Section
            VStack(alignment: .leading, spacing: 5) {
                Text("FRAME RATE")
                    .font(.appMono(.medium, size: 8))
                    .foregroundColor(OwLensTheme.textMuted)

                HStack(spacing: 6) {
                    ForEach(CaptureFrameRate.allCases) { rate in
                        let isSelected = viewModel.selectedFPS == rate
                        Button {
                            Haptics.selection()
                            viewModel.selectedFPS = rate
                        } label: {
                            VStack(spacing: 2) {
                                Text(rate.displayName)
                                    .font(.appFont(isSelected ? .semiBold : .regular, size: 11))
                                    .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                                Text(rate == .fps24 ? "Cinema" : "Broadcast")
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
            }
        }
    }

    private var fpsDrawerContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            drawerHeader(title: "FRAME RATE")

            HStack(spacing: 6) {
                ForEach(CaptureFrameRate.allCases) { rate in
                    let isSelected = viewModel.selectedFPS == rate
                    Button {
                        Haptics.selection()
                        viewModel.selectedFPS = rate
                    } label: {
                        VStack(spacing: 2) {
                            Text(rate.displayName)
                                .font(.appFont(isSelected ? .semiBold : .regular, size: 11))
                                .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                            Text(rate == .fps24 ? "Cinema" : "Broadcast")
                                .font(.appFont(.regular, size: 8))
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
    }

    private var bitrateDrawerContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            drawerHeader(title: "BITRATE")

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
                            Text(bit == viewModel.selectedFormat.suggestedBitratePreset ? "Rec" : "HEVC")
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
                                    Image(systemName: src.portUID == nil ? "mic.slash.fill" : "mic.fill")
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

    private var rightRecordGrip: some View {
        ZStack(alignment: .trailing) {
            // Shutter / Record Trigger - Locked in the exact physical vertical center
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
        .padding(.trailing, 22)
    }

    private var recordButton: some View {
        Button {
            guard !viewModel.isSaving else { return }
            if viewModel.isRecording {
                Haptics.notification(.success)
                viewModel.stopRecording()
            } else {
                Haptics.notification(.success)
                viewModel.startRecording()
            }
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
        if viewModel.selectedAudioSource.portUID == nil { return "OFF" }
        let name = viewModel.selectedAudioSource.name
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

