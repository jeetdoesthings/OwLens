import SwiftUI
import AVFoundation

/// Compact cinema-style controls — chips + one expandable panel at a time.
struct ControlsView: View {
    @ObservedObject var viewModel: CameraViewModel

    var body: some View {
        ZStack {
            topBar
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)

            VStack(spacing: 8) {
                Spacer(minLength: 0)
                lowerStatusStack
                bottomRail
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)

            rightGrip
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .trailing)
        }
        .animation(.easeOut(duration: 0.15), value: viewModel.activePanel)
    }

    // MARK: - Top

    private var topBar: some View {
        HStack(alignment: .top, spacing: 12) {
            topLeftStack

            Spacer(minLength: 12)

            HStack(spacing: 8) {
                previewToggle
                topToggle(systemName: "grid", active: viewModel.showGrid) { viewModel.toggleGrid() }
                topToggle(systemName: "level", active: viewModel.showLevel) { viewModel.toggleLevel() }
                topToggle(systemName: "sun.max.fill", active: viewModel.showClipping) { viewModel.toggleClipping() }
                topToggle(systemName: "viewfinder", active: viewModel.showFocusPeaking) { viewModel.toggleFocusPeaking() }
                topToggle(systemName: "chart.bar", active: viewModel.showScopes) { viewModel.toggleScopes() }
                readoutChip(viewModel.cfaLabel)
            }
        }
        .padding(.leading, 16)
        .padding(.trailing, 92)
        .padding(.top, 12)
    }

    private var topLeftStack: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                if viewModel.isRecording {
                    Circle()
                        .fill(.red)
                        .frame(width: 8, height: 8)
                }
                Text(viewModel.isRecording ? viewModel.recordingDuration : viewModel.statusText)
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .foregroundColor(.white.opacity(0.92))
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)

                if viewModel.isRecording {
                    Text("\(viewModel.frameCount)")
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundColor(.white.opacity(0.45))
                }

                if viewModel.droppedFrames > 0 {
                    Text("DROP \(viewModel.droppedFrames)")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundColor(.orange)
                }
            }
            .padding(.horizontal, 12)
            .frame(height: 34)
            .background(Color.black.opacity(0.48))
            .clipShape(Capsule())

            saveLocationControl
        }
    }

    private var saveLocationControl: some View {
        VStack(alignment: .leading, spacing: 5) {
            Button {
                viewModel.togglePanel(.save)
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: saveDestinationIcon(viewModel.selectedSaveDestination))
                        .font(.system(size: 10, weight: .semibold))
                    Text(viewModel.selectedSaveDestination.label)
                        .font(.system(size: 9, weight: .bold, design: .rounded))
                        .lineLimit(1)
                }
                .foregroundColor(viewModel.activePanel == .save ? .black : .white.opacity(viewModel.isRecording ? 0.35 : 0.85))
                .padding(.horizontal, 10)
                .frame(height: 28)
                .background(viewModel.activePanel == .save ? Color.white : Color.black.opacity(0.42))
                .clipShape(Capsule())
            }
            .buttonStyle(.plain)
            .disabled(viewModel.isRecording)

            if viewModel.activePanel == .save, !viewModel.isRecording {
                HStack(spacing: 5) {
                    ForEach(VideoSaveDestination.allCases) { destination in
                        Button {
                            viewModel.chooseSaveDestination(destination)
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: saveDestinationIcon(destination))
                                    .font(.system(size: 9, weight: .semibold))
                                Text(destination.label)
                                    .font(.system(size: 9, weight: .bold, design: .rounded))
                            }
                            .foregroundColor(viewModel.selectedSaveDestination == destination ? .black : .white.opacity(0.78))
                            .padding(.horizontal, 10)
                            .frame(height: 26)
                            .background(viewModel.selectedSaveDestination == destination ? Color.white : Color.black.opacity(0.55))
                            .clipShape(Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(5)
                .background(Color.black.opacity(0.6))
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
                )
            }
        }
    }

    private func saveDestinationIcon(_ destination: VideoSaveDestination) -> String {
        switch destination {
        case .photos: return "photo.on.rectangle"
        case .files: return "folder"
        }
    }

    private func topToggle(systemName: String, active: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(active ? .black : .white.opacity(0.85))
                .frame(width: 34, height: 34)
                .background(active ? Color.white : Color.black.opacity(0.48))
                .clipShape(Circle())
        }
        .buttonStyle(.plain)
    }

    private var previewToggle: some View {
        Button {
            viewModel.togglePreviewDisplayMode()
        } label: {
            Text(viewModel.previewDisplayMode.label)
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(viewModel.previewDisplayMode == .normalVideo ? .black : .white.opacity(0.9))
                .frame(width: 38, height: 34)
                .background(viewModel.previewDisplayMode == .normalVideo ? Color.white : Color.black.opacity(0.48))
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }

    private func readoutChip(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .bold, design: .monospaced))
            .foregroundColor(.white.opacity(0.55))
            .frame(minWidth: 38)
            .frame(height: 34)
            .padding(.horizontal, 6)
            .background(Color.black.opacity(0.38))
            .clipShape(Capsule())
    }

    // MARK: - Bottom chrome

    private var lowerStatusStack: some View {
        VStack(spacing: 8) {
            if let err = viewModel.errorMessage {
                Text(err)
                    .font(.system(size: 10))
                    .foregroundColor(.red.opacity(0.9))
                    .lineLimit(1)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Color.black.opacity(0.55))
                    .clipShape(Capsule())
            }

            if viewModel.thermalState != .nominal {
                thermalChip
            }

            // Expanded panel — keep layout when recording by using fixed min height zone
            if let panel = viewModel.activePanel, panel != .save, !viewModel.isRecording {
                expandedPanel(panel)
                    .transition(.opacity)
                    .frame(maxWidth: 620)
            }

            if viewModel.showUnverifiedDeviceWarning, !viewModel.isDeviceUnsupportedForLog {
                unverifiedBanner
            }
        }
        .padding(.horizontal, 108)
    }

    private var bottomRail: some View {
        HStack(spacing: 5) {
            chipRow
                .opacity(viewModel.isRecording || viewModel.isDeviceUnsupportedForLog ? 0.35 : 1)
                .allowsHitTesting(!viewModel.isRecording && !viewModel.isDeviceUnsupportedForLog)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .padding(.trailing, 82)
        .background(
            LinearGradient(
                colors: [.clear, .black.opacity(0.5), .black.opacity(0.86)],
                startPoint: .top,
                endPoint: .bottom
            )
        )
    }

    // MARK: - Chips

    private var chipRow: some View {
        HStack(spacing: 5) {
            valueChip(title: "EXP", value: exposureChipValue, panel: .exposure)
            valueChip(title: "FCS", value: viewModel.isAutoFocus ? "AF" : "MF", panel: .focus)
            valueChip(title: "WB", value: viewModel.isAutoWhiteBalanceEnabled ? String(format: "A %.0fK", viewModel.wbKelvin) : String(format: "%.0fK", viewModel.wbKelvin), panel: .wb)
            valueChip(title: "FPS", value: viewModel.selectedFPS.label, panel: .fps)
            valueChip(title: "FMT", value: viewModel.selectedFormat.shortLabel, panel: .format)
            valueChip(title: "BIT", value: viewModel.selectedBitrate.label, panel: .bitrate)
            valueChip(title: "LENS", value: lensShortName, panel: .lens)
            valueChip(title: "MIC", value: micShortName, panel: .mic)
            valueChip(title: "LOG", value: shortCurveName(viewModel.selectedCurve), panel: .log)
        }
    }

    private var lensShortName: String {
        viewModel.selectedLens?.shortLabel ?? "—"
    }

    private var micShortName: String {
        if viewModel.selectedAudioSource.portUID == nil { return "Off" }
        let n = viewModel.selectedAudioSource.name
        if n == "Built-in Mic" || n == "iPhone" { return "Built-in" }
        if n.count <= 6 { return n }
        return String(n.prefix(5))
    }

    private var exposureChipValue: String {
        let prefix = viewModel.isAutoExposureEnabled ? "A " : ""
        return prefix + String(format: "%.0f/%.0f°", viewModel.isoValue, viewModel.shutterValue)
    }

    private func valueChip(title: String, value: String, panel: CameraViewModel.ControlPanel) -> some View {
        let selected = viewModel.activePanel == panel
        let isAllowedWhileLocked = panel == .format || panel == .log
        let lockedOut = viewModel.controlsLocked && !isAllowedWhileLocked
        return Button {
            viewModel.togglePanel(panel)
        } label: {
            VStack(spacing: 1) {
                Text(title)
                    .font(.system(size: 7, weight: .bold))
                    .foregroundColor(selected ? .black.opacity(0.55) : .white.opacity(0.4))
                Text(value)
                    .font(.system(size: 10, weight: .semibold, design: .rounded))
                    .foregroundColor(selected ? .black : .white.opacity(lockedOut ? 0.45 : 0.9))
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 42)
            .background(selected ? Color.white : Color.black.opacity(lockedOut ? 0.22 : 0.48))
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(selected ? Color.clear : Color.white.opacity(lockedOut ? 0.06 : 0.12), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .disabled(lockedOut)
    }

    // MARK: - Expanded panels

    @ViewBuilder
    private func expandedPanel(_ panel: CameraViewModel.ControlPanel) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            switch panel {
            case .exposure, .iso, .shutter:
                panelHeaderRow(
                    title: viewModel.isAutoExposureAdjusting ? "Exposure — auto adjusting" : "Exposure",
                    isAutoOn: $viewModel.isAutoExposureEnabled
                )
                VStack(spacing: 10) {
                    HStack(spacing: 10) {
                        Text("ISO")
                            .font(.system(size: 10, weight: .bold, design: .rounded))
                            .foregroundColor(.white.opacity(0.55))
                            .frame(width: 34, alignment: .leading)
                        stopSlider(
                            index: $viewModel.isoStopIndex,
                            count: viewModel.isoStops.count,
                            label: String(format: "%.0f", viewModel.isoValue),
                            onNudge: { viewModel.nudgeISO($0) }
                        )
                    }
                    HStack(spacing: 10) {
                        Text("ANG")
                            .font(.system(size: 10, weight: .bold, design: .rounded))
                            .foregroundColor(.white.opacity(0.55))
                            .frame(width: 34, alignment: .leading)
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
                    title: viewModel.isAutoWhiteBalanceAdjusting ? "White Balance — auto adjusting" : "White Balance",
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
                HStack(alignment: .center) {
                    panelHeader("Focus")
                    Spacer()
                }
                
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
                panelHeader("Frame Rate (CFR)")
                pillRow(items: CaptureFrameRate.allCases.map { ($0.label, $0) }) { rate in
                    viewModel.selectedFPS = rate
                } isSelected: { $0 == viewModel.selectedFPS }
                Text("File is locked \(viewModel.selectedFPS.label) fps. Gaps held so duration + fps stay correct.")
                    .font(.system(size: 9))
                    .foregroundColor(.white.opacity(0.4))
            case .format:
                panelHeader("Format")
                pillRow(items: RecordingFormat.allCases.map { ($0.shortLabel, $0) }) { fmt in
                    viewModel.selectedFormat = fmt
                } isSelected: { $0 == viewModel.selectedFormat }
                Text(viewModel.selectedFormat.displayName + " · " + viewModel.selectedFormat.detailLabel)
                    .font(.system(size: 9))
                    .foregroundColor(.white.opacity(0.4))
            case .bitrate:
                panelHeader("Bitrate (HEVC)")
                pillRow(items: BitratePreset.allCases.map { ($0.label, $0) }) { bit in
                    viewModel.selectedBitrate = bit
                } isSelected: { $0 == viewModel.selectedBitrate }
                Text("\(viewModel.selectedBitrate.displayName) average")
                    .font(.system(size: 9))
                    .foregroundColor(.white.opacity(0.4))
            case .lens:
                panelHeader("Lens")
                if viewModel.isSwitchingLens {
                    HStack(spacing: 8) {
                        ProgressView().tint(.white).scaleEffect(0.8)
                        Text("Switching lens…")
                            .font(.system(size: 11))
                            .foregroundColor(.white.opacity(0.6))
                    }
                } else if viewModel.availableLenses.isEmpty {
                    Text("No lenses found")
                        .font(.system(size: 11))
                        .foregroundColor(.white.opacity(0.5))
                } else {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            ForEach(viewModel.availableLenses) { lens in
                                pillButton(
                                    title: lens.shortLabel,
                                    selected: viewModel.selectedLens?.id == lens.id
                                ) {
                                    viewModel.selectedLens = lens
                                }
                            }
                        }
                    }
                    Text(viewModel.selectedLens.map { "\($0.name) · \($0.shortLabel)" } ?? "")
                        .font(.system(size: 9))
                        .foregroundColor(.white.opacity(0.4))
                }
            case .mic:
                panelHeader("Audio Source")
                if viewModel.isSwitchingMic {
                    HStack(spacing: 8) {
                        ProgressView()
                            .tint(.white)
                            .scaleEffect(0.8)
                        Text("Switching mic…")
                            .font(.system(size: 11))
                            .foregroundColor(.white.opacity(0.6))
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
                Text(viewModel.selectedAudioSource.name)
                    .font(.system(size: 9))
                    .foregroundColor(.white.opacity(0.4))
                Text("\(viewModel.audioSources.count) inputs · plug mic then reopen MIC")
                    .font(.system(size: 9))
                    .foregroundColor(.white.opacity(0.3))
            case .log:
                panelHeader("Log Curve")
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
                EmptyView()
            }
        }
        .padding(10)
        .background(Color.black.opacity(0.72))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
        )
    }

    private func shortMicLabel(_ src: AudioSourceOption) -> String {
        if src.portUID == nil { return "Off" }
        let n = src.name
        if n.count <= 16 { return n }
        return String(n.prefix(14)) + "…"
    }

    private func panelHeader(_ title: String) -> some View {
        Text(title.uppercased())
            .font(.system(size: 9, weight: .bold))
            .foregroundColor(.white.opacity(0.45))
            .tracking(0.8)
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

    /// Discrete snap slider + − / + for one-stop nudges (ISO 100/200, 1/48, 5600K…).
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
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(selected ? Color.white : Color.white.opacity(0.12))
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Record row

    private var rightGrip: some View {
        VStack(spacing: 14) {
            VStack(spacing: 1) {
                Text(viewModel.selectedFormat.shortLabel)
                    .font(.system(size: 13, weight: .bold, design: .rounded))
                Text("\(viewModel.selectedBitrate.label)M")
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
            }
            .foregroundColor(.white.opacity(0.72))
            .frame(width: 58, height: 42)
            .background(Color.black.opacity(0.42))
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))

            Button {
                if viewModel.controlsLocked {
                    viewModel.unlockControls()
                } else {
                    viewModel.lockControls()
                }
            } label: {
                Image(systemName: viewModel.controlsLocked ? "lock.fill" : "lock.open")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(viewModel.controlsLocked ? .green : .white.opacity(0.75))
                    .frame(width: 50, height: 50)
                    .background(Color.black.opacity(0.46))
                    .clipShape(Circle())
            }
            .disabled(viewModel.isRecording)
            .opacity(viewModel.isRecording ? 0.35 : 1)

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
                        .frame(width: 72, height: 72)
                    if viewModel.isRecording {
                        RoundedRectangle(cornerRadius: 3)
                            .fill(.red)
                            .frame(width: 24, height: 24)
                    } else {
                        Circle()
                            .fill(
                                viewModel.isDeviceUnsupportedForLog
                                    ? Color.gray.opacity(0.4)
                                    : (viewModel.controlsLocked ? Color.red : Color.red.opacity(0.35))
                            )
                            .frame(width: 56, height: 56)
                    }
                }
            }
            .disabled(
                viewModel.isDeviceUnsupportedForLog
                    || (!viewModel.controlsLocked && !viewModel.isRecording)
            )
        }
        .padding(.trailing, 14)
        .padding(.vertical, 14)
        .background(
            LinearGradient(
                colors: [.clear, .black.opacity(0.35), .black.opacity(0.72)],
                startPoint: .leading,
                endPoint: .trailing
            )
        )
    }

    private var unverifiedBanner: some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 10))
            Text("Untested device (\(viewModel.capabilities?.marketingName ?? "?")) — recording may misbehave. Report logs if you try.")
                .font(.system(size: 10, weight: .medium))
                .lineLimit(2)
        }
        .foregroundColor(.orange)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    // MARK: - Thermal

    private var thermalChip: some View {
        HStack(spacing: 6) {
            Image(systemName: "thermometer.medium")
                .font(.system(size: 10))
            Text(thermalMessage)
                .font(.system(size: 10, weight: .semibold))
        }
        .foregroundColor(thermalColor)
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(thermalColor.opacity(0.15))
        .clipShape(Capsule())
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
        case .critical: return "STOP — too hot"
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
