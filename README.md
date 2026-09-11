# OwLens

<img src="owlens-logo.png" width="128" height="128" alt="OwLens Logo" />

🦉 **OwLens** is a professional cinema camera application for iOS designed for filmmakers, colorists, and mobile videographers who demand uncompromised image quality and complete control over video capture.

OwLens bypasses Apple's standard image signal processor (ISP), capturing uncompressed 14-bit RAW Bayer sensor data and processing it in real time through a custom Metal compute pipeline. Footage is encoded directly into 10-bit HEVC with true log transfer functions (**Apple Log 2** or **Sony S-Log3**) and factory-calibrated color matrices.

---

## Key Features

### 🎬 Cinema Log Curves & Calibrated Color Science
- **Dual Log Profiles**:
  - **Apple Log 2**: Fully compliant with Apple's published Log specification (18% middle gray at 0.488 code value, 90% white at 0.682, 1200% dynamic range ceiling).
  - **Sony S-Log3**: Standard Sony S-Log3 curve mapped to Sony S-Gamut3.Cine primaries.
- **Factory DNG Calibration**: Extracts Apple's factory-calibrated `ForwardMatrix2` directly from the camera sensor metadata and applies a Bradford Chromatic Adaptation Transform ($M_{\text{CAT}(D50\to D65)}$) to preserve neutral white balance, prevent oversaturation, and ensure lifelike skin tones.
- **Dynamic Container Color Metadata**: Injects exact NCLC metadata into video containers:
  - **Apple Log 2 / S-Log3**: `9-1-9` (ITU-R BT.2020 primaries, ITU-R BT.709-2 transfer characteristics, ITU-R BT.2020 non-constant matrix).
  - **Linear / Rec.709**: `1-1-1` (ITU-R BT.709 throughout).

### ⚡ High-Performance Metal Compute Pipeline (~2.5 ms Frame Times)
- **Fused Demosaicing Kernel (`debayerFusedLog`)**: Fuses Malvar-He-Cutler directional demosaic, green balancing, 4-term Lens Shading Correction (LSC), white balance gains, calibrated color correction matrix, and Log OETF into a single GPU compute pass. Eliminates 1 full pass and saves ~196 MB/frame of DRAM bandwidth.
- **Single-Pass Resampling & Format Conversion**: Resamples directly into the target `.bgra8Unorm` texture memory in a single step, eliminating redundant Lanczos sinc interpolation passes.
- **Zero-Allocation Bayer Buffer Pooling**: `CaptureController` employs a preallocated `CVPixelBufferPool` for incoming sensor frames, eliminating 720 MB/s of runtime IOSurface kernel allocations and page-fault stalls.
- **Responsive Capture Cadence**: Completion-driven frame capture re-triggering keeps the sensor 100% utilized at true 24 fps / 30 fps CFR without timer drift or hold-frame stutters.

### 🎛️ Minimalist Glass HUD & Intuitive Controls
- **Monochromatic White Aesthetic**: Refined, distraction-free HUD where UI speaks through hierarchy and opacity rather than distracting colors.
- **Manual Camera Controls**: Continuous magnetic sliders for ISO (sensor native range), shutter angle (degrees with cinematic snap targets like $180^\circ$), and White Balance (Kelvin).
- **Dual Viewfinder Monitoring**:
  - **Normal Video (VID)**: Hardware ISP Rec.709 preview via `AVCaptureVideoPreviewLayer` with zero GPU overhead.
  - **Log Preview (LOG)**: Real-time debayered log preview with zebra clipping indicators (sensor saturation) and focus peaking (green Laplacian edge highlight).
- **Professional Framing & Overlays**: Real-time 6-axis gyroscope level monitor, rule-of-thirds grid, and live RGB histogram / luma waveform scopes.
- **Flexible Framing Options**:
  - **Open Gate 4:3** ($1920\times 1440$): Full sensor readout without 16:9 crop, providing maximum vertical framing flexibility in post.
  - **1080p 16:9** ($1920\times 1080$): High-speed standard delivery framing.
  - **4K UHD 16:9** ($3840\times 2160$): Ultra-sharp 4K cinema capture at up to 150 Mbps HEVC.

---

## Post-Production Color Grading

Footage recorded with OwLens is designed to grade cleanly in **DaVinci Resolve**, **Final Cut Pro**, and **Adobe Premiere Pro**.

### DaVinci Resolve: Color Space Transform (CST) Setup

Add a **Color Space Transform** node at the beginning of your node tree:

#### For Apple Log 2 Footage:
| Setting | Value |
|---|---|
| **Input Color Space** | `Rec.2020` |
| **Input Gamma** | `Apple Log` |
| **Output Color Space** | `Rec.709` (or Timeline Color Space) |
| **Output Gamma** | `Rec.709` / `Gamma 2.4` |
| **Tone Mapping** | DaVinci / Tone Mapping (optional) |

#### For Sony S-Log3 Footage:
| Setting | Value |
|---|---|
| **Input Color Space** | `Sony S-Gamut3.Cine` |
| **Input Gamma** | `Sony S-Log3` |
| **Output Color Space** | `Rec.709` (or Timeline Color Space) |
| **Output Gamma** | `Rec.709` / `Gamma 2.4` |

---

## Technical Specifications

| Parameter | Specification |
|---|---|
| **Input Sensor Format** | 14-bit RAW Bayer (RGGB / GRBG / GBRG / BGGR) |
| **Demosaicing** | Malvar-He-Cutler (MHC) 5×5 directional gradient demosaic |
| **Lens Shading Correction** | 4-term per-channel radial ($r^2, r^4$) + azimuth ($\cos 2\theta$) model |
| **Color Spaces** | ITU-R BT.2020 (D65) / Sony S-Gamut3.Cine (D65) |
| **Log Curves** | Apple Log 2 (published white paper spec) / Sony S-Log3 |
| **Container Formats** | QuickTime Movie (`.mov`), HEVC (H.265) 10-bit |
| **Container NCLC Tags** | `9-1-9` (Apple Log 2 / S-Log3) or `1-1-1` (Linear / Rec.709) |
| **Frame Rates** | Constant Frame Rate (CFR) at 24.00 fps or 30.00 fps |
| **Bitrates** | User-selectable: 50, 80, 100, 150 Mbps |
| **Audio** | 48 kHz Linear PCM / AAC, external mic support (USB, Bluetooth, 3.5mm) |

---

## Requirements & Building

- **Mac**: macOS Sonoma or Sequoia with Xcode 15+
- **Device**: iPhone with iOS 17.0 or later (Bayer RAW capture requires physical back camera)
- **Deployment**:
  1. Clone the repository:
     ```bash
     git clone https://github.com/jeetdoesthings/OwLens.git
     cd OwLens
     ```
  2. Open in Xcode:
     ```bash
     open OwLens.xcodeproj
     ```
  3. Under **Signing & Capabilities**, select your development team.
  4. Build and run on your connected iPhone (**Cmd + R**).
  
  That's it, OwLens will launch on your phone! 🦉

---

## License

OwLens is a source-available project. The source code is provided for personal, educational, and evaluation purposes only. Commercial exploitation, distribution, and publishing to any public app store (including the Apple App Store) are strictly prohibited. Refer to the [LICENSE](LICENSE) for full terms.
