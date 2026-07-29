import SwiftUI
import AVFoundation

/// Landscape cinema-camera controls layered over the preview.
struct ControlsView: View {
    @ObservedObject var viewModel: CameraViewModel

    private let chromeRadius: CGFloat = 8

    var body: some View {
        ZStack {
            statusStrip
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)

            leftToolRail
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)

            VStack(spacing: 8) {
                Spacer(minLength: 0)
                transientStatus
                exposureStrip
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)

            recordGrip
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .trailing)
        }
        .animation(.easeOut(duration: 0.15), value: viewModel.activePanel)
    }

    // MARK: - Top Status

    private var statusStrip: some View {
        ViewThatFits(in: .horizontal) {
            fullStatusStrip
            compactStatusStrip
        }
    }

    private var fullStatusStrip: some View {
        HStack(spacing: 8) {
            if viewModel.isRecording {
                recordingState
            }
            statusItem(icon: "timer", text: viewModel.isRecording ? viewModel.recordingDuration : viewModel.statusText)

            if viewModel.droppedFrames > 0 {
                statusItem(icon: "exclamationmark.triangle.fill", text: "\(viewModel.droppedFrames)", color: .orange)
            }

            Spacer(minLength: 10)

            compactLensSelector

            Spacer(minLength: 10)

            fpsToggleButton(compact: false)
            statusItem(icon: "waveform.path.ecg", text: shortCurveName(viewModel.selectedCurve))
            statusItem(icon: "camera.filters", text: viewModel.cfaLabel)
            statusIconButton(
                systemName: saveDestinationIcon(viewModel.selectedSaveDestination),
                active: viewModel.selectedSaveDestination == .files,
                disabled: viewModel.isRecording,
                accessibilityLabel: "Save destination"
            ) {
                toggleSaveDestination()
            }
        }
        .padding(.leading, 72)
        .padding(.trailing, 98)
        .padding(.top, 10)
    }

    private var compactStatusStrip: some View {
        HStack(spacing: 6) {
            if viewModel.isRecording {
                recordingState
            }
            compactLensSelector
            Spacer(minLength: 6)
            fpsToggleButton(compact: true)
            statusItem(icon: "waveform.path.ecg", text: shortCurveName(viewModel.selectedCurve))
            statusIconButton(
                systemName: saveDestinationIcon(viewModel.selectedSaveDestination),
                active: viewModel.selectedSaveDestination == .files,
                disabled: viewModel.isRecording,
                accessibilityLabel: "Save destination"
            ) {
                toggleSaveDestination()
            }
        }
        .padding(.leading, 60)
        .padding(.trailing, 90)
        .padding(.top, 10)
    }

    private var recordingState: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(Color.red)
                .frame(width: 8, height: 8)
        }
        .foregroundColor(.white.opacity(0.9))
        .padding(.horizontal, 8)
        .frame(height: 32)
        .background(Color.black.opacity(0.54))
        .clipShape(RoundedRectangle(cornerRadius: chromeRadius, style: .continuous))
        .accessibilityLabel(viewModel.isRecording ? "Recording" : "Camera readiness")
    }

    private func statusItem(icon: String, text: String, color: Color = .white) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))
                .frame(width: 12)
            Text(text)
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .lineLimit(1)
                .minimumScaleFactor(0.65)
        }
        .foregroundColor(color.opacity(0.9))
        .padding(.horizontal, 8)
        .frame(height: 32)
        .background(Color.black.opacity(0.48))
        .clipShape(RoundedRectangle(cornerRadius: chromeRadius, style: .continuous))
    }

    private func fpsToggleButton(compact: Bool) -> some View {
        let disabled = viewModel.controlsLocked || viewModel.isRecording || viewModel.isDeviceUnsupportedForLog
        return Button {
            toggleFrameRate()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "gauge.with.dots.needle.33percent")
                    .font(.system(size: 10, weight: .semibold))
                    .frame(width: 12)
                Text(compact ? viewModel.selectedFPS.label : "\(viewModel.selectedFPS.label)fps")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .lineLimit(1)
                    .minimumScaleFactor(0.65)
            }
            .foregroundColor(.white.opacity(disabled ? 0.35 : 0.9))
            .padding(.horizontal, 8)
            .frame(height: 32)
            .background(Color.black.opacity(disabled ? 0.24 : 0.48))
            .clipShape(RoundedRectangle(cornerRadius: chromeRadius, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .accessibilityLabel("Frame rate")
    }

    private var compactLensSelector: some View {
        HStack(spacing: 4) {
            ForEach(viewModel.availableLenses) { lens in
                Button {
                    viewModel.selectedLens = lens
                } label: {
                    Text(lens.shortLabel)
                        .font(.system(size: 11, weight: viewModel.selectedLens?.id == lens.id ? .bold : .semibold, design: .rounded))
                        .foregroundColor(viewModel.selectedLens?.id == lens.id ? .black : .white.opacity(lensControlsDisabled ? 0.38 : 0.82))
                        .lineLimit(1)
                        .minimumScaleFactor(0.75)
                        .frame(width: 38, height: 28)
                        .background(viewModel.selectedLens?.id == lens.id ? Color.white : Color.black.opacity(lensControlsDisabled ? 0.24 : 0.48))
                        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                }
                .buttonStyle(.plain)
                .disabled(lensControlsDisabled || viewModel.isSwitchingLens)
                .accessibilityLabel("Lens \(lens.shortLabel)")
            }
        }
        .padding(2)
        .frame(height: 32)
        .background(Color.black.opacity(0.28))
        .clipShape(RoundedRectangle(cornerRadius: chromeRadius, style: .continuous))
        .accessibilityElement(children: .contain)
    }

    private var lensControlsDisabled: Bool {
        viewModel.controlsLocked || viewModel.isRecording || viewModel.isDeviceUnsupportedForLog
    }

    // MARK: - Left Rail

    private var leftToolRail: some View {
        VStack(spacing: 8) {
            railToggle(
                systemName: viewModel.previewDisplayMode == .log ? "camera.metering.matrix" : "camera.viewfinder",
                active: viewModel.previewDisplayMode == .normalVideo,
                accessibilityLabel: "Preview mode"
            ) {
                viewModel.togglePreviewDisplayMode()
            }
            railToggle(systemName: "grid", active: viewModel.showGrid, accessibilityLabel: "Grid") {
                viewModel.toggleGrid()
            }
            railToggle(systemName: "level", active: viewModel.showLevel, accessibilityLabel: "Level") {
                viewModel.toggleLevel()
            }
            railToggle(systemName: "sun.max.fill", active: viewModel.showClipping, accessibilityLabel: "Clipping") {
                viewModel.toggleClipping()
            }
            railToggle(systemName: "viewfinder", active: viewModel.showFocusPeaking, accessibilityLabel: "Focus peaking") {
                viewModel.toggleFocusPeaking()
            }
            railToggle(systemName: "chart.xyaxis.line", active: viewModel.showScopes, accessibilityLabel: "Scopes") {
                viewModel.toggleScopes()
            }
        }
        .padding(6)
        .background(Color.black.opacity(0.32))
        .clipShape(RoundedRectangle(cornerRadius: chromeRadius, style: .continuous))
        .padding(.leading, 12)
    }

    private func railToggle(
        systemName: String,
        active: Bool,
        accessibilityLabel: String,
        action: @escaping () -> Void
    ) -> some View {
        statusIconButton(
            systemName: systemName,
            active: active,
            disabled: false,
            accessibilityLabel: accessibilityLabel,
            action: action
        )
    }

    private func statusIconButton(
        systemName: String,
        active: Bool,
        disabled: Bool,
        accessibilityLabel: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(active ? .black : .white.opacity(disabled ? 0.35 : 0.84))
                .frame(width: 34, height: 34)
                .background(active ? Color.white : Color.black.opacity(disabled ? 0.22 : 0.52))
                .clipShape(Circle())
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .accessibilityLabel(accessibilityLabel)
        .help(accessibilityLabel)
    }

    // MARK: - Bottom Exposure

    private var transientStatus: some View {
        VStack(spacing: 7) {
            if let err = viewModel.errorMessage {
                messagePill(icon: "xmark.octagon.fill", text: err, color: .red)
            }

            if viewModel.thermalState != .nominal {
                messagePill(icon: "thermometer.medium", text: thermalMessage, color: thermalColor)
            }

            if let panel = viewModel.activePanel, !viewModel.isRecording {
                expandedPanel(panel)
                    .transition(.opacity)
                    .frame(maxWidth: 610)
            }

            if viewModel.showUnverifiedDeviceWarning, !viewModel.isDeviceUnsupportedForLog {
                messagePill(
                    icon: "exclamationmark.triangle.fill",
                    text: "Untested \(viewModel.capabilities?.marketingName ?? "device")",
                    color: .orange
                )
            }
        }
        .padding(.horizontal, 94)
    }

    private var exposureStrip: some View {
        ViewThatFits(in: .horizontal) {
            fullExposureStrip
            compactExposureStrip
        }
        .padding(.vertical, 7)
        .background(
            LinearGradient(
                colors: [.clear, .black.opacity(0.55), .black.opacity(0.88)],
                startPoint: .top,
                endPoint: .bottom
            )
        )
    }

    private var fullExposureStrip: some View {
        HStack(spacing: 6) {
            exposureControl
            wbControl
            focusModeControl
            secondaryControl(systemName: "rectangle.dashed", value: viewModel.selectedFormat.shortLabel, panel: .format)
            secondaryControl(systemName: "speedometer", value: "\(viewModel.selectedBitrate.label)M", panel: .bitrate)
            micControl
        }
        .padding(.leading, 74)
        .padding(.trailing, 88)
    }

    private var compactExposureStrip: some View {
        HStack(spacing: 6) {
            compactControlButton(
                systemName: "camera.aperture",
                value: compactExposureValue,
                selected: viewModel.activePanel == .exposure,
                disabled: exposureControlsDisabled,
                accessibilityLabel: "Exposure"
            ) {
                viewModel.togglePanel(.exposure)
            }
            compactControlButton(
                systemName: "thermometer.sun",
                value: "\(Int(viewModel.wbKelvin))",
                selected: viewModel.activePanel == .wb,
                disabled: exposureControlsDisabled,
                accessibilityLabel: "White balance"
            ) {
                viewModel.togglePanel(.wb)
            }
            compactControlButton(
                systemName: viewModel.isAutoFocus ? "scope" : "dial.low",
                value: viewModel.isAutoFocus ? "AF" : "MF",
                selected: false,
                disabled: exposureControlsDisabled,
                accessibilityLabel: "Focus mode"
            ) {
                viewModel.togglePanel(.focus)
            }
            compactControlButton(
                systemName: "rectangle.dashed",
                value: viewModel.selectedFormat.shortLabel,
                selected: viewModel.activePanel == .format,
                disabled: viewModel.isRecording,
                accessibilityLabel: "Format"
            ) {
                viewModel.togglePanel(.format)
            }
            compactControlButton(
                systemName: micIcon,
                value: micShortName,
                selected: viewModel.activePanel == .mic,
                disabled: micControlsDisabled,
                accessibilityLabel: "Microphone"
            ) {
                viewModel.togglePanel(.mic)
            }
        }
        .padding(.leading, 60)
        .padding(.trailing, 88)
    }

    private var exposureControl: some View {
        controlCell(
            systemName: "camera.aperture",
            value: exposureValue,
            selected: viewModel.activePanel == .exposure,
            disabled: exposureControlsDisabled,
            accessibilityLabel: "Exposure"
        ) {
            viewModel.togglePanel(.exposure)
        }
    }

    private var wbControl: some View {
        controlCell(
            systemName: "thermometer.sun",
            value: viewModel.isAutoWhiteBalanceEnabled ? "A \(Int(viewModel.wbKelvin))K" : "\(Int(viewModel.wbKelvin))K",
            selected: viewModel.activePanel == .wb,
            disabled: exposureControlsDisabled,
            accessibilityLabel: "White balance"
        ) {
            viewModel.togglePanel(.wb)
        }
    }

    private var focusModeControl: some View {
        let panelActive = viewModel.activePanel == .focus
        return Button {
            guard !exposureControlsDisabled else { return }
            viewModel.togglePanel(.focus)
        } label: {
            VStack(spacing: 3) {
                Image(systemName: viewModel.isAutoFocus ? "scope" : "dial.low")
                    .font(.system(size: 14, weight: .semibold))
                Text(viewModel.isAutoFocus ? "AF" : "MF")
                    .font(.system(size: 11, weight: .bold, design: .rounded))
            }
            .foregroundColor(panelActive ? .black : .white.opacity(exposureControlsDisabled ? 0.35 : 0.88))
            .frame(width: 48, height: 40)
            .background(panelActive ? Color.white : Color.black.opacity(exposureControlsDisabled ? 0.22 : 0.52))
            .clipShape(RoundedRectangle(cornerRadius: chromeRadius, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(exposureControlsDisabled)
        .accessibilityLabel("Focus mode")
        .help("Focus mode")
    }

    private func secondaryControl(systemName: String, value: String, panel: CameraViewModel.ControlPanel) -> some View {
        let disabled = viewModel.isRecording || (viewModel.controlsLocked && panel != .format && panel != .log)
        return Button {
            viewModel.togglePanel(panel)
        } label: {
            VStack(spacing: 3) {
                Image(systemName: systemName)
                    .font(.system(size: 13, weight: .semibold))
                Text(value)
                    .font(.system(size: 10, weight: .semibold, design: .rounded))
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
            }
            .foregroundColor(viewModel.activePanel == panel ? .black : .white.opacity(disabled ? 0.35 : 0.84))
            .frame(width: 52, height: 40)
            .background(viewModel.activePanel == panel ? Color.white : Color.black.opacity(disabled ? 0.22 : 0.48))
            .clipShape(RoundedRectangle(cornerRadius: chromeRadius, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .accessibilityLabel(value)
    }

    private var micControl: some View {
        Button {
            viewModel.togglePanel(.mic)
        } label: {
            VStack(spacing: 3) {
                Image(systemName: micIcon)
                    .font(.system(size: 13, weight: .semibold))
                Text(micShortName)
                    .font(.system(size: 10, weight: .semibold, design: .rounded))
                    .lineLimit(1)
                    .minimumScaleFactor(0.62)
            }
            .foregroundColor(viewModel.activePanel == .mic ? .black : .white.opacity(micControlsDisabled ? 0.35 : 0.84))
            .frame(width: 58, height: 40)
            .background(viewModel.activePanel == .mic ? Color.white : Color.black.opacity(micControlsDisabled ? 0.22 : 0.48))
            .clipShape(RoundedRectangle(cornerRadius: chromeRadius, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(micControlsDisabled)
        .accessibilityLabel("Microphone")
    }

    private func controlCell(
        systemName: String,
        value: String,
        selected: Bool,
        disabled: Bool,
        accessibilityLabel: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 3) {
                Image(systemName: systemName)
                    .font(.system(size: 12, weight: .semibold))
                Text(value)
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .lineLimit(1)
                    .minimumScaleFactor(0.58)
            }
            .foregroundColor(selected ? .black : .white.opacity(disabled ? 0.35 : 0.9))
            .frame(width: 66, height: 40)
            .background(selected ? Color.white : Color.black.opacity(disabled ? 0.22 : 0.5))
            .clipShape(RoundedRectangle(cornerRadius: chromeRadius, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .accessibilityLabel(accessibilityLabel)
    }

    private func compactControlButton(
        systemName: String,
        value: String,
        selected: Bool,
        disabled: Bool,
        accessibilityLabel: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 3) {
                Image(systemName: systemName)
                    .font(.system(size: 12, weight: .semibold))
                Text(value)
                    .font(.system(size: 9, weight: .semibold, design: .rounded))
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
            }
            .foregroundColor(selected ? .black : .white.opacity(disabled ? 0.34 : 0.86))
            .frame(width: 46, height: 40)
            .background(selected ? Color.white : Color.black.opacity(disabled ? 0.22 : 0.5))
            .clipShape(RoundedRectangle(cornerRadius: chromeRadius, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .accessibilityLabel(accessibilityLabel)
    }

    private func nudgeButton(systemName: String, disabled: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(.white.opacity(disabled ? 0.25 : 0.78))
                .frame(width: 24, height: 42)
                .background(Color.black.opacity(disabled ? 0.16 : 0.42))
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(disabled)
    }

    private var exposureControlsDisabled: Bool {
        viewModel.controlsLocked || viewModel.isRecording || viewModel.isDeviceUnsupportedForLog
    }

    private var micControlsDisabled: Bool {
        viewModel.controlsLocked || viewModel.isRecording
    }

    private var exposureValue: String {
        let prefix = viewModel.isAutoExposureEnabled ? "A " : ""
        return prefix + "\(Int(viewModel.isoValue))/\(Int(viewModel.shutterValue))°"
    }

    private var compactExposureValue: String {
        let prefix = viewModel.isAutoExposureEnabled ? "A" : ""
        return prefix + "\(Int(viewModel.isoValue))/\(Int(viewModel.shutterValue))"
    }

    private func toggleFrameRate() {
        guard !viewModel.controlsLocked, !viewModel.isRecording else { return }
        viewModel.selectedFPS = viewModel.selectedFPS == .fps24 ? .fps30 : .fps24
    }

    // MARK: - Expanded Panels

    @ViewBuilder
    private func expandedPanel(_ panel: CameraViewModel.ControlPanel) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            switch panel {
            case .exposure, .iso, .shutter:
                panelHeaderRow(
                    title: viewModel.isAutoExposureAdjusting ? "Exposure auto adjusting" : "Exposure",
                    isAutoOn: $viewModel.isAutoExposureEnabled
                )
                VStack(spacing: 10) {
                    HStack(spacing: 10) {
                        panelCaption("ISO")
                        stopSlider(
                            index: $viewModel.isoStopIndex,
                            count: viewModel.isoStops.count,
                            label: String(format: "%.0f", viewModel.isoValue),
                            onNudge: { viewModel.nudgeISO($0) }
                        )
                    }
                    HStack(spacing: 10) {
                        panelCaption("ANG")
                        HStack {
                            Text(String(format: "%.0f°", viewModel.shutterRange.lowerBound))
                                .font(.system(size: 10, weight: .medium))
                                .foregroundColor(.white.opacity(0.6))
                            Slider(value: Binding(get: { viewModel.shutterValue }, set: {
                                viewModel.setShutterAngleWithSnapping($0)
                            }), in: viewModel.shutterRange)
                            .tint(.white)
                            Text(String(format: "%.0f°", viewModel.shutterRange.upperBound))
                                .font(.system(size: 10, weight: .medium))
                                .foregroundColor(.white.opacity(0.6))
                            Text(String(format: "%.0f°", viewModel.shutterValue))
                                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                                .foregroundColor(.white.opacity(0.9))
                                .frame(width: 58, alignment: .trailing)
                        }
                        .padding(.horizontal, 4)
                    }
                }
                .opacity(viewModel.isAutoExposureEnabled ? 0.35 : 1)
                .disabled(viewModel.isAutoExposureEnabled)
            case .wb:
                panelHeaderRow(
                    title: viewModel.isAutoWhiteBalanceAdjusting ? "White balance auto adjusting" : "White balance",
                    isAutoOn: $viewModel.isAutoWhiteBalanceEnabled
                )
                stopSlider(
                    index: $viewModel.wbStopIndex,
                    count: viewModel.wbStops.count,
                    label: String(format: "%.0fK", viewModel.wbKelvin),
                    onNudge: { viewModel.nudgeWB($0) }
                )
                .opacity(viewModel.isAutoWhiteBalanceEnabled ? 0.35 : 1)
                .disabled(viewModel.isAutoWhiteBalanceEnabled)
            case .focus:
                panelHeader("Focus")
                HStack {
                    Text("Macro")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(.white.opacity(0.6))
                    Slider(value: Binding(get: { viewModel.focusLensPosition }, set: {
                        viewModel.isAutoFocus = false
                        viewModel.focusLensPosition = $0
                    }), in: 0.0...1.0)
                    .tint(.white)
                    Text("Infinity")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(.white.opacity(0.6))
                }
                .padding(.horizontal, 4)
            case .fps:
                panelHeader("Frame rate")
                pillRow(items: CaptureFrameRate.allCases.map { ($0.displayName, $0) }) { rate in
                    viewModel.selectedFPS = rate
                } isSelected: { $0 == viewModel.selectedFPS }
            case .format:
                panelHeader("Format")
                pillRow(items: RecordingFormat.allCases.map { ($0.shortLabel, $0) }) { fmt in
                    viewModel.selectedFormat = fmt
                } isSelected: { $0 == viewModel.selectedFormat }
                panelFootnote(viewModel.selectedFormat.displayName + " · " + viewModel.selectedFormat.detailLabel)
            case .bitrate:
                panelHeader("Bitrate")
                pillRow(items: BitratePreset.allCases.filter { $0.rawValue <= viewModel.selectedFormat.maxBitratePreset.rawValue }.map { ($0.displayName, $0) }) { bit in
                    viewModel.selectedBitrate = bit
                } isSelected: { $0 == viewModel.selectedBitrate }
            case .lens:
                panelHeader("Lens")
                lensPanelContent
            case .mic:
                panelHeader("Audio")
                micPanelContent
            case .log:
                panelHeader("Curve")
                HStack(spacing: 6) {
                    ForEach(LogCurveType.uiCases) { curve in
                        pillButton(
                            title: shortCurveName(curve),
                            selected: viewModel.selectedCurve == curve
                        ) {
                            if !viewModel.controlsLocked {
                                viewModel.selectedCurve = curve
                            }
                        }
                    }
                }
            case .save:
                panelHeader("Save")
                HStack(spacing: 6) {
                    ForEach(VideoSaveDestination.allCases) { destination in
                        pillButton(
                            title: destination.label,
                            selected: viewModel.selectedSaveDestination == destination
                        ) {
                            viewModel.chooseSaveDestination(destination)
                        }
                    }
                }
            }
        }
        .padding(10)
        .background(Color.black.opacity(0.72))
        .clipShape(RoundedRectangle(cornerRadius: chromeRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: chromeRadius, style: .continuous)
                .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
        )
    }

    private var lensPanelContent: some View {
        Group {
            if viewModel.isSwitchingLens {
                HStack(spacing: 8) {
                    ProgressView().tint(.white).scaleEffect(0.8)
                    panelFootnote("Switching")
                }
            } else if viewModel.availableLenses.isEmpty {
                panelFootnote("No lenses")
            } else {
                pillRow(items: viewModel.availableLenses.map { ($0.shortLabel, $0) }) { lens in
                    viewModel.selectedLens = lens
                } isSelected: { viewModel.selectedLens?.id == $0.id }
            }
        }
    }

    private var micPanelContent: some View {
        Group {
            if viewModel.isSwitchingMic {
                HStack(spacing: 8) {
                    ProgressView().tint(.white).scaleEffect(0.8)
                    panelFootnote("Switching")
                }
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(viewModel.audioSources) { src in
                            pillButton(
                                title: shortMicLabel(src),
                                selected: viewModel.selectedAudioSource.id == src.id
                            ) {
                                viewModel.selectedAudioSource = src
                            }
                        }
                    }
                }
            }
        }
    }

    private func panelHeader(_ title: String) -> some View {
        Text(title.uppercased())
            .font(.system(size: 9, weight: .bold))
            .foregroundColor(.white.opacity(0.48))
    }

    private func panelCaption(_ title: String) -> some View {
        Text(title)
            .font(.system(size: 10, weight: .bold, design: .rounded))
            .foregroundColor(.white.opacity(0.55))
            .frame(width: 34, alignment: .leading)
    }

    private func panelFootnote(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .medium))
            .foregroundColor(.white.opacity(0.45))
            .lineLimit(1)
            .minimumScaleFactor(0.75)
    }

    private func panelHeaderRow(title: String, isAutoOn: Binding<Bool>) -> some View {
        HStack(spacing: 10) {
            panelHeader(title)
            Spacer(minLength: 0)
            Toggle("", isOn: isAutoOn)
                .labelsHidden()
                .toggleStyle(.switch)
                .scaleEffect(0.72)
            Text("AUTO")
                .font(.system(size: 9, weight: .bold, design: .rounded))
                .foregroundColor(isAutoOn.wrappedValue ? .white : .white.opacity(0.42))
        }
    }

    private func stopSlider(
        index: Binding<Int>,
        count: Int,
        label: String,
        onNudge: @escaping (Int) -> Void
    ) -> some View {
        let maxIndex = max(0, count - 1)
        return HStack(spacing: 8) {
            Button {
                onNudge(-1)
            } label: {
                Image(systemName: "minus")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(.white.opacity(0.85))
                    .frame(width: 32, height: 32)
                    .background(Color.white.opacity(0.12))
                    .clipShape(Circle())
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
            .tint(.white.opacity(0.8))

            Button {
                onNudge(1)
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(.white.opacity(0.85))
                    .frame(width: 32, height: 32)
                    .background(Color.white.opacity(0.12))
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
            .disabled(index.wrappedValue >= maxIndex)

            Text(label)
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                .foregroundColor(.white.opacity(0.9))
                .frame(width: 58, alignment: .trailing)
        }
    }

    private func pillRow<T: Hashable>(
        items: [(String, T)],
        onSelect: @escaping (T) -> Void,
        isSelected: @escaping (T) -> Bool
    ) -> some View {
        HStack(spacing: 6) {
            ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                pillButton(title: item.0, selected: isSelected(item.1)) {
                    onSelect(item.1)
                }
            }
            Spacer(minLength: 0)
        }
    }

    private func pillButton(title: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 11, weight: selected ? .bold : .regular))
                .foregroundColor(selected ? .black : .white.opacity(0.8))
                .lineLimit(1)
                .minimumScaleFactor(0.75)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(selected ? Color.white : Color.white.opacity(0.12))
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Record Grip

    private var recordGrip: some View {
        VStack(spacing: 12) {
            Button {
                if viewModel.controlsLocked {
                    viewModel.unlockControls()
                } else {
                    viewModel.lockControls()
                }
            } label: {
                Image(systemName: viewModel.controlsLocked ? "lock.fill" : "lock.open")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundColor(viewModel.controlsLocked ? .green : .white.opacity(0.78))
                    .frame(width: 50, height: 50)
                    .background(Color.black.opacity(0.5))
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
            .disabled(viewModel.isRecording)
            .opacity(viewModel.isRecording ? 0.35 : 1)
            .accessibilityLabel(viewModel.controlsLocked ? "Unlock controls" : "Lock controls")

            Button {
                if viewModel.isRecording {
                    viewModel.stopRecording()
                } else {
                    viewModel.startRecording()
                }
            } label: {
                ZStack {
                    Circle()
                        .strokeBorder(.white.opacity(0.95), lineWidth: 3)
                        .frame(width: 74, height: 74)
                    if viewModel.isRecording {
                        RoundedRectangle(cornerRadius: 4, style: .continuous)
                            .fill(.red)
                            .frame(width: 25, height: 25)
                    } else {
                        Circle()
                            .fill(
                                viewModel.isDeviceUnsupportedForLog
                                    ? Color.gray.opacity(0.4)
                                    : (viewModel.controlsLocked ? Color.red : Color.red.opacity(0.34))
                            )
                            .frame(width: 57, height: 57)
                    }
                }
                .frame(width: 74, height: 74)
                .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .disabled(
                viewModel.isDeviceUnsupportedForLog
                    || (!viewModel.controlsLocked && !viewModel.isRecording)
            )
            .accessibilityLabel(viewModel.isRecording ? "Stop recording" : "Start recording")
        }
        .padding(.trailing, 12)
        .padding(.vertical, 12)
    }

    // MARK: - Helpers

    private func messagePill(icon: String, text: String, color: Color) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))
            Text(text)
                .font(.system(size: 10, weight: .semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.72)
        }
        .foregroundColor(color)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(color.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: chromeRadius, style: .continuous))
    }

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

    private var micIcon: String {
        viewModel.selectedAudioSource.portUID == nil ? "mic.slash.fill" : "mic.fill"
    }

    private var micShortName: String {
        if viewModel.selectedAudioSource.portUID == nil { return "Off" }
        let name = viewModel.selectedAudioSource.name
        if name == "Built-in Mic" || name == "iPhone" { return "Built-in" }
        if name.count <= 7 { return name }
        return String(name.prefix(6))
    }

    private func shortMicLabel(_ src: AudioSourceOption) -> String {
        if src.portUID == nil { return "Off" }
        let name = src.name
        if name.count <= 16 { return name }
        return String(name.prefix(14)) + "…"
    }

    private var thermalColor: Color {
        switch viewModel.thermalState {
        case .fair: return .yellow
        case .serious: return .orange
        case .critical: return .red
        default: return .green
        }
    }

    private var thermalMessage: String {
        switch viewModel.thermalState {
        case .fair: return "Warming"
        case .serious: return "Thermal limit"
        case .critical: return "Too hot"
        default: return ""
        }
    }

    private func shortCurveName(_ curve: LogCurveType) -> String {
        switch curve {
        case .linear: return "Lin"
        case .sLog3Approx: return "S-Log3"
        }
    }
}
