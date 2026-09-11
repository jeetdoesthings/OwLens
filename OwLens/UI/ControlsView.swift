import SwiftUI
import AVFoundation

/// Landscape cinema-camera HUD layered over the live viewfinder.
/// Monochromatic redesign: every element earns its pixel.
struct ControlsView: View {
    @ObservedObject var viewModel: CameraViewModel

    @State private var showLockNotice = false
    @State private var lockNoticeTimer: Timer?

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

            // Right Record Grip
            rightRecordGrip
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .trailing)
        }
        .animation(.spring(response: 0.28, dampingFraction: 0.82), value: viewModel.activePanel)
        .animation(.spring(response: 0.28, dampingFraction: 0.82), value: viewModel.isRecording)
        .animation(.spring(response: 0.28, dampingFraction: 0.82), value: viewModel.controlsLocked)
    }

    // MARK: - Top Status Bar

    private var topStatusBar: some View {
        HStack(spacing: 8) {
            leftStatusGroup
            Spacer(minLength: 8)
            centerStatusGroup
            Spacer(minLength: 8)
            rightStatusGroup
        }
        .padding(.leading, 64)
        .padding(.trailing, 92)
        .padding(.top, 10)
    }

    private var isTopLocked: Bool {
        viewModel.controlsLocked || viewModel.isRecording || viewModel.isDeviceUnsupportedForLog
    }

    private var leftStatusGroup: some View {
        HStack(spacing: 6) {
            if viewModel.isRecording {
                HStack(spacing: 6) {
                    Circle()
                        .fill(OwLensTheme.recordingRed)
                        .frame(width: 7, height: 7)
                    Text("REC")
                        .font(.geist(.bold, size: 10))
                        .foregroundColor(OwLensTheme.recordingRed)
                    Text(viewModel.recordingDuration)
                        .font(.geistMono(.bold, size: 12))
                        .foregroundColor(.white)
                }
                .padding(.horizontal, 10)
                .frame(height: 30)
                .glassPanel(cornerRadius: OwLensTheme.radiusPill, border: OwLensTheme.glassBorderRed, background: OwLensTheme.glassBaseHeavy)
            } else {
                Button {
                    Haptics.impact(.medium)
                    if viewModel.controlsLocked {
                        viewModel.unlockControls()
                    } else {
                        viewModel.lockControls()
                    }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: viewModel.controlsLocked ? "lock.fill" : "lock.open")
                            .font(.system(size: 9, weight: .semibold))
                        Text(viewModel.controlsLocked ? "LOCKED" : "UNLOCK")
                            .font(.geistMono(.semiBold, size: 9))
                    }
                    .foregroundColor(viewModel.controlsLocked ? OwLensTheme.textPrimary : OwLensTheme.textSecondary)
                    .padding(.horizontal, 10)
                    .frame(height: 30)
                    .glassPanel(
                        cornerRadius: OwLensTheme.radiusPill,
                        border: viewModel.controlsLocked ? OwLensTheme.glassBorderActive : OwLensTheme.glassBorder
                    )
                }
                .buttonStyle(.plain)
            }

            if viewModel.droppedFrames > 0 {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 9, weight: .semibold))
                    Text("\(viewModel.droppedFrames) DROPS")
                        .font(.geistMono(.semiBold, size: 9))
                }
                .foregroundColor(OwLensTheme.textPrimary)
                .padding(.horizontal, 8)
                .frame(height: 30)
                .glassPanel(cornerRadius: OwLensTheme.radiusPill, border: OwLensTheme.glassBorderActive)
            }
        }
    }

    private var centerStatusGroup: some View {
        HStack(spacing: 6) {
            // Lens Switcher Pill
            lensSwitcherPill

            // Resolution & FPS Pill
            Button {
                Haptics.selection()
                viewModel.togglePanel(.format)
            } label: {
                HStack(spacing: 4) {
                    Text(viewModel.selectedFormat.shortLabel)
                        .font(.geist(.semiBold, size: 11))
                        .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textPrimary)
                    Text("·")
                        .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textMuted)
                    Text("\(viewModel.selectedFPS.label)fps")
                        .font(.geistMono(.medium, size: 10))
                        .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textSecondary)
                }
                .padding(.horizontal, 10)
                .frame(height: 30)
                .glassPanel(
                    cornerRadius: OwLensTheme.radiusPill,
                    border: (viewModel.activePanel == .format || viewModel.activePanel == .fps) ? OwLensTheme.glassBorderActive : OwLensTheme.glassBorder,
                    background: (viewModel.activePanel == .format || viewModel.activePanel == .fps) ? OwLensTheme.glassActiveBg : OwLensTheme.glassBase
                )
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(isTopLocked)
            .opacity(isTopLocked ? 0.4 : 1.0)

            // Curve Badge
            Button {
                Haptics.selection()
                viewModel.togglePanel(.log)
            } label: {
                Text(shortCurveName(viewModel.selectedCurve))
                    .font(.geist(.semiBold, size: 10))
                    .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textPrimary)
                    .padding(.horizontal, 9)
                    .frame(height: 30)
                    .glassPanel(
                        cornerRadius: OwLensTheme.radiusPill,
                        border: viewModel.activePanel == .log ? OwLensTheme.glassBorderActive : OwLensTheme.glassBorder,
                        background: viewModel.activePanel == .log ? OwLensTheme.glassActiveBg : OwLensTheme.glassBase
                    )
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
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
                        .font(.geist(isSelected ? .semiBold : .regular, size: 11))
                        .foregroundColor(isSelected ? (isTopLocked ? .white : .black) : (isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textSecondary))
                        .frame(minWidth: 32)
                        .frame(height: 24)
                        .padding(.horizontal, 4)
                        .background(
                            Capsule(style: .continuous)
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
        .glassPanel(cornerRadius: OwLensTheme.radiusPill)
        .opacity(isTopLocked ? 0.4 : (viewModel.isSwitchingLens ? 0.5 : 1.0))
        .animation(.easeInOut(duration: 0.15), value: viewModel.isSwitchingLens)
    }

    private var rightStatusGroup: some View {
        HStack(spacing: 6) {
            // Denoise Indicator
            Button {
                Haptics.selection()
                viewModel.togglePanel(.denoise)
            } label: {
                HStack(spacing: 3) {
                    Image(systemName: "sparkles")
                        .font(.system(size: 9, weight: .medium))
                    Text(String(format: "%.1f", viewModel.denoiseStrength))
                        .font(.geistMono(.medium, size: 10))
                }
                .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textSecondary)
                .padding(.horizontal, 8)
                .frame(height: 30)
                .glassPanel(
                    cornerRadius: OwLensTheme.radiusPill,
                    border: viewModel.activePanel == .denoise ? OwLensTheme.glassBorderActive : OwLensTheme.glassBorder,
                    background: viewModel.activePanel == .denoise ? OwLensTheme.glassActiveBg : OwLensTheme.glassBase
                )
            }
            .buttonStyle(.plain)
            .disabled(isTopLocked)
            .opacity(isTopLocked ? 0.4 : 1.0)

            // Save Destination (Photos vs Files)
            Button {
                Haptics.selection()
                toggleSaveDestination()
            } label: {
                Image(systemName: saveDestinationIcon(viewModel.selectedSaveDestination))
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(isTopLocked ? OwLensTheme.textDisabled : OwLensTheme.textSecondary)
                    .frame(width: 30, height: 30)
                    .glassPanel(cornerRadius: OwLensTheme.radiusPill)
            }
            .buttonStyle(.plain)
            .disabled(isTopLocked)
            .opacity(isTopLocked ? 0.4 : 1.0)

            // CFA Pattern Tag
            Text(viewModel.cfaLabel)
                .font(.geistMono(.medium, size: 9))
                .foregroundColor(OwLensTheme.textMuted)
                .padding(.horizontal, 8)
                .frame(height: 30)
                .glassPanel(cornerRadius: OwLensTheme.radiusPill)
                .opacity(isTopLocked ? 0.4 : 1.0)
        }
    }

    // MARK: - Left Monitoring Tools Rail

    private var leftToolRail: some View {
        VStack(spacing: 4) {
            monitoringToolButton(
                systemName: viewModel.previewDisplayMode == .log ? "camera.metering.matrix" : "camera.viewfinder",
                isActive: viewModel.previewDisplayMode == .normalVideo
            ) {
                viewModel.togglePreviewDisplayMode()
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
        .padding(.leading, 12)
    }

    private func monitoringToolButton(
        systemName: String,
        isActive: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button {
            Haptics.selection()
            action()
        } label: {
            Image(systemName: systemName)
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(isActive ? .black : OwLensTheme.textSecondary)
                .frame(width: 34, height: 34)
                .background(
                    RoundedRectangle(cornerRadius: OwLensTheme.radiusSm, style: .continuous)
                        .fill(isActive ? OwLensTheme.glassActive : Color.clear)
                )
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Bottom Exposure Deck

    private var bottomExposureDeck: some View {
        HStack(spacing: 5) {
            // ISO & Shutter Angle
            deckTile(
                title: "EXPOSURE",
                value: exposureSummaryValue,
                subvalue: viewModel.isAutoExposureEnabled ? "AUTO" : "MANUAL",
                isSelected: viewModel.activePanel == .exposure,
                isDisabled: exposureControlsDisabled,
                width: 86
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
                width: 68
            ) {
                viewModel.togglePanel(.wb)
            }

            // Focus Mode (AF vs MF)
            deckTile(
                title: "FOCUS",
                value: viewModel.isAutoFocus ? "AF" : "MF",
                subvalue: viewModel.isAutoFocus ? "CONT" : String(format: "%.2f", viewModel.focusLensPosition),
                isSelected: viewModel.activePanel == .focus,
                isDisabled: exposureControlsDisabled,
                width: 58
            ) {
                viewModel.togglePanel(.focus)
            }

            // Format & Aspect
            deckTile(
                title: "FORMAT",
                value: "\(viewModel.selectedFormat.shortLabel)·\(viewModel.selectedFPS.label)",
                subvalue: viewModel.selectedFormat.detailLabel,
                isSelected: viewModel.activePanel == .format || viewModel.activePanel == .fps,
                isDisabled: viewModel.isRecording || viewModel.controlsLocked,
                width: 70
            ) {
                viewModel.togglePanel(.format)
            }

            // Bitrate
            deckTile(
                title: "BITRATE",
                value: "\(viewModel.selectedBitrate.label)M",
                subvalue: "HEVC",
                isSelected: viewModel.activePanel == .bitrate,
                isDisabled: viewModel.isRecording || viewModel.controlsLocked,
                width: 58
            ) {
                viewModel.togglePanel(.bitrate)
            }

            // Audio Source
            deckTile(
                title: "AUDIO",
                value: micShortName,
                subvalue: viewModel.selectedAudioSource.portUID == nil ? "MUTED" : "ACTIVE",
                isSelected: viewModel.activePanel == .mic,
                isDisabled: viewModel.isRecording || viewModel.controlsLocked,
                width: 64
            ) {
                viewModel.togglePanel(.mic)
            }
        }
        .padding(.leading, 64)
        .padding(.trailing, 92)
        .padding(.bottom, 10)
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
                    .font(.geistMono(.medium, size: 7))
                    .foregroundColor(isSelected ? .black.opacity(0.50) : OwLensTheme.textMuted)
                Text(value)
                    .font(.geistMono(.semiBold, size: 12))
                    .foregroundColor(isSelected ? .black : (isDisabled ? OwLensTheme.textDisabled : OwLensTheme.textPrimary))
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
                Text(subvalue)
                    .font(.geistMono(.regular, size: 7))
                    .foregroundColor(isSelected ? .black.opacity(0.40) : OwLensTheme.textMuted)
                    .lineLimit(1)
            }
            .frame(width: width, height: 42)
            .background(
                RoundedRectangle(cornerRadius: OwLensTheme.radiusSm, style: .continuous)
                    .fill(isSelected ? OwLensTheme.glassActive : (isDisabled ? OwLensTheme.glassBase.opacity(0.3) : OwLensTheme.glassBaseHeavy))
            )
            .overlay(
                RoundedRectangle(cornerRadius: OwLensTheme.radiusSm, style: .continuous)
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
        viewModel.controlsLocked || viewModel.isRecording || viewModel.isDeviceUnsupportedForLog
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

            if showLockNotice {
                messageBadge(
                    icon: "lock.fill",
                    text: "Lock controls to start recording",
                    color: OwLensTheme.textSecondary
                )
                .transition(.scale.combined(with: .opacity))
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
                .font(.geist(.medium, size: 11))
                .lineLimit(1)
        }
        .foregroundColor(color)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .glassPanel(cornerRadius: OwLensTheme.radiusPill, border: color.opacity(0.25), background: OwLensTheme.glassBaseHeavy)
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

            case .log:
                curveDrawerContent

            case .mic:
                micDrawerContent

            case .denoise:
                denoiseDrawerContent

            case .save:
                saveDrawerContent

            case .lens:
                lensDrawerContent
            }
        }
        .padding(12)
        .glassPanel(cornerRadius: OwLensTheme.radiusLg, border: OwLensTheme.glassBorderActive, background: OwLensTheme.glassBaseHeavy)
        .shadow(color: Color.black.opacity(0.4), radius: 12, x: 0, y: 4)
    }

    // MARK: - Drawer Sub-views

    private var exposureDrawerContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            drawerHeaderRow(
                title: "EXPOSURE",
                isAutoOn: $viewModel.isAutoExposureEnabled,
                autoLabel: "AUTO"
            )

            VStack(spacing: 8) {
                // ISO Stepped Stop Control
                HStack(spacing: 10) {
                    Text("ISO")
                        .font(.geistMono(.semiBold, size: 10))
                        .foregroundColor(OwLensTheme.textSecondary)
                        .frame(width: 36, alignment: .leading)

                    stopStepper(
                        index: $viewModel.isoStopIndex,
                        count: viewModel.isoStops.count,
                        label: String(format: "%.0f", viewModel.isoValue),
                        onNudge: { viewModel.nudgeISO($0) }
                    )
                }

                // Shutter Angle Slider & Cinema Snap Targets
                HStack(spacing: 10) {
                    Text("ANG")
                        .font(.geistMono(.semiBold, size: 10))
                        .foregroundColor(OwLensTheme.textSecondary)
                        .frame(width: 36, alignment: .leading)

                    HStack(spacing: 8) {
                        Slider(value: Binding(get: { viewModel.shutterValue }, set: {
                            viewModel.setShutterAngleWithSnapping($0)
                        }), in: viewModel.shutterRange)
                        .tint(OwLensTheme.textPrimary)

                        Text(String(format: "%.0f°", viewModel.shutterValue))
                            .font(.geistMono(.semiBold, size: 12))
                            .foregroundColor(OwLensTheme.textPrimary)
                            .frame(width: 50, alignment: .trailing)
                    }
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
                                .font(.geistMono(isMatch ? .semiBold : .regular, size: 9))
                                .foregroundColor(isMatch ? .black : OwLensTheme.textSecondary)
                                .padding(.horizontal, 7)
                                .padding(.vertical, 4)
                                .background(
                                    Capsule(style: .continuous)
                                        .fill(isMatch ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                                )
                                .contentShape(Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .opacity(viewModel.isAutoExposureEnabled ? 0.3 : 1.0)
            .disabled(viewModel.isAutoExposureEnabled)
        }
    }

    private var whiteBalanceDrawerContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            drawerHeaderRow(
                title: "WHITE BALANCE",
                isAutoOn: $viewModel.isAutoWhiteBalanceEnabled,
                autoLabel: "AUTO"
            )

            VStack(spacing: 8) {
                // Kelvin Stop Stepper
                stopStepper(
                    index: $viewModel.wbStopIndex,
                    count: viewModel.wbStops.count,
                    label: String(format: "%.0fK", viewModel.wbKelvin),
                    onNudge: { viewModel.nudgeWB($0) }
                )

                // Quick Presets
                HStack(spacing: 4) {
                    ForEach([
                        ("3200K", Float(3200)),
                        ("4000K", Float(4000)),
                        ("5600K", Float(5600)),
                        ("7000K", Float(7000))
                    ], id: \.0) { item in
                        let isMatch = abs(viewModel.wbKelvin - item.1) < 150
                        Button {
                            Haptics.selection()
                            viewModel.wbStopIndex = ExposureStops.nearestIndex(in: viewModel.wbStops, to: item.1)
                        } label: {
                            Text(item.0)
                                .font(.geist(isMatch ? .semiBold : .regular, size: 9))
                                .foregroundColor(isMatch ? .black : OwLensTheme.textSecondary)
                                .padding(.horizontal, 7)
                                .padding(.vertical, 4)
                                .background(
                                    Capsule(style: .continuous)
                                        .fill(isMatch ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                                )
                                .contentShape(Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .opacity(viewModel.isAutoWhiteBalanceEnabled ? 0.3 : 1.0)
            .disabled(viewModel.isAutoWhiteBalanceEnabled)
        }
    }

    private var focusDrawerContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            drawerHeader(title: "FOCUS")

            HStack(spacing: 6) {
                Button {
                    Haptics.selection()
                    viewModel.isAutoFocus = true
                } label: {
                    Text("AF")
                        .font(.geist(viewModel.isAutoFocus ? .semiBold : .regular, size: 10))
                        .foregroundColor(viewModel.isAutoFocus ? .black : OwLensTheme.textSecondary)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 6)
                        .background(
                            Capsule(style: .continuous)
                                .fill(viewModel.isAutoFocus ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                        )
                        .contentShape(Capsule())
                }
                .buttonStyle(.plain)

                Button {
                    Haptics.selection()
                    viewModel.isAutoFocus = false
                } label: {
                    Text("MF")
                        .font(.geist(!viewModel.isAutoFocus ? .semiBold : .regular, size: 10))
                        .foregroundColor(!viewModel.isAutoFocus ? .black : OwLensTheme.textSecondary)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 6)
                        .background(
                            Capsule(style: .continuous)
                                .fill(!viewModel.isAutoFocus ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                        )
                        .contentShape(Capsule())
                }
                .buttonStyle(.plain)

                Spacer()
            }

            if !viewModel.isAutoFocus {
                HStack(spacing: 8) {
                    Text("Near")
                        .font(.geistMono(.regular, size: 9))
                        .foregroundColor(OwLensTheme.textMuted)

                    Slider(value: Binding(get: { viewModel.focusLensPosition }, set: {
                        viewModel.isAutoFocus = false
                        viewModel.focusLensPosition = $0
                    }), in: 0.0...1.0)
                    .tint(OwLensTheme.textPrimary)

                    Text("Far")
                        .font(.geistMono(.regular, size: 9))
                        .foregroundColor(OwLensTheme.textMuted)

                    Text(String(format: "%.2f", viewModel.focusLensPosition))
                        .font(.geistMono(.semiBold, size: 12))
                        .foregroundColor(OwLensTheme.textPrimary)
                        .frame(width: 40, alignment: .trailing)
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
                    .font(.geistMono(.medium, size: 8))
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
                                    .font(.geist(isSelected ? .semiBold : .regular, size: 11))
                                    .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                                Text(fmt.detailLabel)
                                    .font(.geistMono(.regular, size: 9))
                                    .foregroundColor(isSelected ? .black.opacity(0.5) : OwLensTheme.textMuted)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: OwLensTheme.radiusSm, style: .continuous)
                                    .fill(isSelected ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: OwLensTheme.radiusSm, style: .continuous)
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
                    .font(.geistMono(.medium, size: 8))
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
                                    .font(.geist(isSelected ? .semiBold : .regular, size: 11))
                                    .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                                Text(rate == .fps24 ? "Cinema" : "Broadcast")
                                    .font(.geistMono(.regular, size: 8))
                                    .foregroundColor(isSelected ? .black.opacity(0.5) : OwLensTheme.textMuted)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: OwLensTheme.radiusSm, style: .continuous)
                                    .fill(isSelected ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: OwLensTheme.radiusSm, style: .continuous)
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
                                .font(.geist(isSelected ? .semiBold : .regular, size: 11))
                                .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                            Text(rate == .fps24 ? "Cinema" : "Broadcast")
                                .font(.geist(.regular, size: 8))
                                .foregroundColor(isSelected ? .black.opacity(0.5) : OwLensTheme.textMuted)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: OwLensTheme.radiusSm, style: .continuous)
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
                                .font(.geistMono(isSelected ? .semiBold : .regular, size: 11))
                                .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                            Text(bit == viewModel.selectedFormat.suggestedBitratePreset ? "Rec" : "HEVC")
                                .font(.geistMono(.regular, size: 8))
                                .foregroundColor(isSelected ? .black.opacity(0.5) : OwLensTheme.textMuted)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: OwLensTheme.radiusSm, style: .continuous)
                                .fill(isSelected ? OwLensTheme.glassActive : OwLensTheme.glassBase)
                        )
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var curveDrawerContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            drawerHeader(title: "LOG CURVE")

            HStack(spacing: 6) {
                ForEach(LogCurveType.uiCases) { curve in
                    let isSelected = viewModel.selectedCurve == curve
                    Button {
                        Haptics.selection()
                        viewModel.selectedCurve = curve
                    } label: {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(shortCurveName(curve))
                                .font(.geist(isSelected ? .semiBold : .medium, size: 12))
                                .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                            Text(curveDescription(curve))
                                .font(.geist(.regular, size: 9))
                                .foregroundColor(isSelected ? .black.opacity(0.5) : OwLensTheme.textMuted)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                        .background(
                            RoundedRectangle(cornerRadius: OwLensTheme.radiusSm, style: .continuous)
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
                        .font(.geist(.regular, size: 11))
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
                                        .font(.geist(isSelected ? .semiBold : .regular, size: 11))
                                }
                                .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(
                                    RoundedRectangle(cornerRadius: OwLensTheme.radiusSm, style: .continuous)
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

    private var denoiseDrawerContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            drawerHeader(title: "DENOISE")

            VStack(spacing: 6) {
                HStack(spacing: 8) {
                    Text("Off")
                        .font(.geistMono(.regular, size: 9))
                        .foregroundColor(OwLensTheme.textMuted)

                    Slider(value: $viewModel.denoiseStrength, in: 0.0...1.0)
                        .tint(OwLensTheme.textPrimary)

                    Text("Max")
                        .font(.geistMono(.regular, size: 9))
                        .foregroundColor(OwLensTheme.textMuted)

                    Text(String(format: "%.2f", viewModel.denoiseStrength))
                        .font(.geistMono(.semiBold, size: 12))
                        .foregroundColor(OwLensTheme.textPrimary)
                        .frame(width: 42, alignment: .trailing)
                }

                if viewModel.denoiseStrength > 0.7 {
                    Text("High values may drop frames in 4K.")
                        .font(.geist(.regular, size: 9))
                        .foregroundColor(OwLensTheme.textSecondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
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
                                .font(.geist(isSelected ? .semiBold : .regular, size: 11))
                        }
                        .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: OwLensTheme.radiusSm, style: .continuous)
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
                                .font(.geist(isSelected ? .semiBold : .regular, size: 12))
                                .foregroundColor(isSelected ? .black : OwLensTheme.textPrimary)
                            Text(lens.name)
                                .font(.geist(.regular, size: 8))
                                .foregroundColor(isSelected ? .black.opacity(0.5) : OwLensTheme.textMuted)
                                .lineLimit(1)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: OwLensTheme.radiusSm, style: .continuous)
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
                .font(.geistMono(.semiBold, size: 9))
                .foregroundColor(OwLensTheme.textMuted)
                .tracking(1.5)

            Spacer()

            Button {
                Haptics.selection()
                viewModel.activePanel = nil
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(OwLensTheme.textMuted)
                    .frame(width: 26, height: 26)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
    }

    private func drawerHeaderRow(title: String, isAutoOn: Binding<Bool>, autoLabel: String) -> some View {
        HStack {
            Text(title)
                .font(.geistMono(.semiBold, size: 9))
                .foregroundColor(OwLensTheme.textMuted)
                .tracking(1.5)

            Spacer()

            Toggle("", isOn: isAutoOn)
                .labelsHidden()
                .toggleStyle(.switch)
                .scaleEffect(0.65)

            Text(autoLabel)
                .font(.geistMono(.semiBold, size: 9))
                .foregroundColor(isAutoOn.wrappedValue ? OwLensTheme.textPrimary : OwLensTheme.textMuted)

            Button {
                Haptics.selection()
                viewModel.activePanel = nil
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(OwLensTheme.textMuted)
                    .frame(width: 26, height: 26)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(.leading, 4)
        }
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
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(OwLensTheme.textPrimary)
                    .frame(width: 28, height: 28)
                    .background(Circle().fill(OwLensTheme.glassBase))
                    .overlay(Circle().strokeBorder(OwLensTheme.glassBorder, lineWidth: 0.5))
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
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
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(OwLensTheme.textPrimary)
                    .frame(width: 28, height: 28)
                    .background(Circle().fill(OwLensTheme.glassBase))
                    .overlay(Circle().strokeBorder(OwLensTheme.glassBorder, lineWidth: 0.5))
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(index.wrappedValue >= maxIndex)

            Text(label)
                .font(.geistMono(.semiBold, size: 13))
                .foregroundColor(OwLensTheme.textPrimary)
                .frame(width: 58, alignment: .trailing)
        }
    }

    // MARK: - Right Record Grip

    private var rightRecordGrip: some View {
        VStack(spacing: 14) {
            // Lock / Unlock Switch
            Button {
                Haptics.impact(.medium)
                if viewModel.controlsLocked {
                    viewModel.unlockControls()
                } else {
                    viewModel.lockControls()
                }
            } label: {
                Image(systemName: viewModel.controlsLocked ? "lock.fill" : "lock.open")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(viewModel.controlsLocked ? OwLensTheme.textPrimary : OwLensTheme.textSecondary)
                    .frame(width: 44, height: 44)
                    .background(
                        Circle()
                            .fill(OwLensTheme.glassBaseHeavy)
                    )
                    .overlay(
                        Circle()
                            .strokeBorder(viewModel.controlsLocked ? OwLensTheme.glassBorderActive : OwLensTheme.glassBorder, lineWidth: 0.5)
                    )
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .disabled(viewModel.isRecording)
            .opacity(viewModel.isRecording ? 0.3 : 1.0)

            // Shutter / Record Trigger
            Button {
                if viewModel.isRecording {
                    Haptics.notification(.success)
                    viewModel.stopRecording()
                } else {
                    if !viewModel.controlsLocked {
                        Haptics.notification(.warning)
                        showLockNoticeToast()
                    } else {
                        Haptics.notification(.success)
                        viewModel.startRecording()
                    }
                }
            } label: {
                ZStack {
                    // Outer Ring
                    Circle()
                        .strokeBorder(OwLensTheme.textPrimary, lineWidth: 2.5)
                        .frame(width: 68, height: 68)
                        .shadow(color: viewModel.isRecording ? OwLensTheme.recordingRed.opacity(0.6) : Color.clear, radius: 8)

                    if viewModel.isRecording {
                        // Red Stop Square
                        RoundedRectangle(cornerRadius: 5, style: .continuous)
                            .fill(OwLensTheme.recordingRed)
                            .frame(width: 24, height: 24)
                    } else {
                        // Inner Record Circle
                        Circle()
                            .fill(
                                viewModel.isDeviceUnsupportedForLog
                                    ? Color.gray.opacity(0.3)
                                    : (viewModel.controlsLocked ? OwLensTheme.recordingRed : OwLensTheme.recordingRed.opacity(0.35))
                            )
                            .frame(width: 52, height: 52)
                    }
                }
                .frame(width: 68, height: 68)
                .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .disabled(viewModel.isDeviceUnsupportedForLog)
        }
        .padding(.trailing, 12)
        .padding(.vertical, 12)
    }

    private func showLockNoticeToast() {
        lockNoticeTimer?.invalidate()
        withAnimation {
            showLockNotice = true
        }
        lockNoticeTimer = Timer.scheduledTimer(withTimeInterval: 2.5, repeats: false) { _ in
            Task { @MainActor in
                withAnimation {
                    showLockNotice = false
                }
            }
        }
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

    private func shortCurveName(_ curve: LogCurveType) -> String {
        switch curve {
        case .linear: return "Linear"
        case .sLog3Approx: return "S-Log3"
        case .appleLog2: return "A-Log2"
        }
    }

    private func curveDescription(_ curve: LogCurveType) -> String {
        switch curve {
        case .linear: return "Linear sensor transform"
        case .sLog3Approx: return "High dynamic range log"
        case .appleLog2: return "Apple Log 2 encoding"
        }
    }
}
