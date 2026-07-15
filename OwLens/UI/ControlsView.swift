import SwiftUI
import AVFoundation

/// Compact cinema-style controls — chips + one expandable panel at a time.
struct ControlsView: View {
    @ObservedObject var viewModel: CameraViewModel

    var body: some View {
        VStack(spacing: 0) {
            topBar
            Spacer(minLength: 0)
            bottomChrome
        }
    }

    // MARK: - Top

    private var topBar: some View {
        HStack(spacing: 10) {
            Text(viewModel.statusText)
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .foregroundColor(.white.opacity(0.9))
                .lineLimit(1)

            Spacer(minLength: 8)

            if viewModel.isRecording {
                HStack(spacing: 4) {
                    Circle().fill(.red).frame(width: 6, height: 6)
                    Text(viewModel.recordingDuration)
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundColor(.white)
                    Text("· \(viewModel.frameCount)")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.white.opacity(0.45))
                }
            }

            if viewModel.droppedFrames > 0 {
                Text("↓\(viewModel.droppedFrames)")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.orange)
            }

            // Grid + level toggles (top-right)
            topToggle(
                systemName: "grid",
                active: viewModel.showGrid,
                action: { viewModel.toggleGrid() }
            )
            topToggle(
                systemName: "level",
                active: viewModel.showLevel,
                action: { viewModel.toggleLevel() }
            )

            Text(viewModel.cfaLabel)
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundColor(.white.opacity(0.4))
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(Color.white.opacity(0.08))
                .clipShape(Capsule())
        }
        .padding(.horizontal, 16)
        .padding(.top, 10)
    }

    private func topToggle(systemName: String, active: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(active ? .black : .white.opacity(0.85))
                .frame(width: 32, height: 32)
                .background(active ? Color.white : Color.white.opacity(0.12))
                .clipShape(Circle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Bottom chrome

    private var bottomChrome: some View {
        VStack(spacing: 8) {
            if let err = viewModel.errorMessage {
                Text(err)
                    .font(.system(size: 10))
                    .foregroundColor(.red.opacity(0.9))
                    .lineLimit(1)
            }

            if viewModel.thermalState != .nominal {
                thermalChip
            }

            // Expanded panel — keep layout when recording by using fixed min height zone
            if let panel = viewModel.activePanel, !viewModel.isRecording {
                expandedPanel(panel)
                    .transition(.opacity)
            }

            if viewModel.showUnverifiedDeviceWarning, !viewModel.isDeviceUnsupportedForLog {
                unverifiedBanner
            }

            // Always reserve chip row space so preview doesn't jump mid-record
            chipRow
                .opacity(viewModel.isRecording || viewModel.isDeviceUnsupportedForLog ? 0.35 : 1)
                .allowsHitTesting(!viewModel.isRecording && !viewModel.isDeviceUnsupportedForLog)

            recordRow
        }
        .padding(.horizontal, 14)
        .padding(.bottom, 18)
        .padding(.top, 10)
        .background(
            LinearGradient(
                colors: [.clear, .black.opacity(0.5), .black.opacity(0.88)],
                startPoint: .top,
                endPoint: .bottom
            )
        )
        .animation(.easeOut(duration: 0.15), value: viewModel.activePanel)
    }

    // MARK: - Chips

    private var chipRow: some View {
        HStack(spacing: 5) {
            valueChip(title: "ISO", value: String(format: "%.0f", viewModel.isoValue), panel: .iso)
            valueChip(title: "SHT", value: String(format: "1/%.0f", viewModel.shutterValue), panel: .shutter)
            valueChip(title: "WB", value: String(format: "%.0fK", viewModel.wbKelvin), panel: .wb)
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
        if n == "iPhone" { return "Phone" }
        if n.count <= 6 { return n }
        return String(n.prefix(5))
    }

    private func valueChip(title: String, value: String, panel: CameraViewModel.ControlPanel) -> some View {
        let selected = viewModel.activePanel == panel
        let lockedOut = viewModel.controlsLocked && panel != .log
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
            .padding(.vertical, 5)
            .background(selected ? Color.white : Color.white.opacity(lockedOut ? 0.05 : 0.1))
            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .strokeBorder(lockedOut ? Color.white.opacity(0.08) : Color.clear, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .disabled(lockedOut && panel != .log)
    }

    // MARK: - Expanded panels

    @ViewBuilder
    private func expandedPanel(_ panel: CameraViewModel.ControlPanel) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            switch panel {
            case .iso:
                panelHeader("ISO — snap stops")
                stopSlider(
                    index: $viewModel.isoStopIndex,
                    count: viewModel.isoStops.count,
                    label: String(format: "%.0f", viewModel.isoValue),
                    onNudge: { viewModel.nudgeISO($0) }
                )
            case .shutter:
                panelHeader("Shutter — snap stops")
                stopSlider(
                    index: $viewModel.shutterStopIndex,
                    count: viewModel.shutterStops.count,
                    label: String(format: "1/%.0f", viewModel.shutterValue),
                    onNudge: { viewModel.nudgeShutter($0) }
                )
            case .wb:
                panelHeader("White Balance — snap stops")
                stopSlider(
                    index: $viewModel.wbStopIndex,
                    count: viewModel.wbStops.count,
                    label: String(format: "%.0fK", viewModel.wbKelvin),
                    onNudge: { viewModel.nudgeWB($0) }
                )
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
                    Spacer(minLength: 0)
                    Button {
                        viewModel.exportLUT()
                    } label: {
                        Image(systemName: "square.and.arrow.up")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundColor(.white.opacity(0.85))
                            .padding(8)
                            .background(Color.white.opacity(0.12))
                            .clipShape(Circle())
                    }
                }
            }
        }
        .padding(10)
        .background(Color.white.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
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
                .background(selected ? Color.white : Color.white.opacity(0.1))
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Record row

    private var recordRow: some View {
        HStack(spacing: 28) {
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
                    .frame(width: 44, height: 44)
                    .background(Color.white.opacity(0.1))
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
                        .strokeBorder(.white, lineWidth: 3)
                        .frame(width: 56, height: 56)
                    if viewModel.isRecording {
                        RoundedRectangle(cornerRadius: 3)
                            .fill(.red)
                            .frame(width: 20, height: 20)
                    } else {
                        Circle()
                            .fill(
                                viewModel.isDeviceUnsupportedForLog
                                    ? Color.gray.opacity(0.4)
                                    : (viewModel.controlsLocked ? Color.red : Color.red.opacity(0.35))
                            )
                            .frame(width: 44, height: 44)
                    }
                }
            }
            .disabled(
                viewModel.isDeviceUnsupportedForLog
                    || (!viewModel.controlsLocked && !viewModel.isRecording)
            )

            VStack(spacing: 1) {
                Text(viewModel.selectedFormat.shortLabel)
                    .font(.system(size: 12, weight: .bold, design: .rounded))
                Text("\(viewModel.selectedBitrate.label)M")
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
            }
            .foregroundColor(.white.opacity(0.55))
            .frame(width: 48, height: 44)
        }
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
