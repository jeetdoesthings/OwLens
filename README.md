# OwLens

<img src="owlens-logo.png" width="128" height="128" alt="OwLens Logo" />

🦉 **OwLens** is a professional-grade iOS cinema camera app designed for filmmakers and creators who demand uncompromising image fidelity and true control over video capture. It bypasses Apple's standard image signal processor (ISP), capturing uncompressed 14-bit RAW Bayer sensor data at full sensor resolution ($4032 \times 3024$, 12.2 MP) and encoding it directly into 10-bit HEVC Log in real time.

---

## Key Highlights & Architecture

- **True 14-Bit Bayer RAW Capture:** Bypasses Apple's ISP noise reduction, computational edge sharpening, and adaptive tone mapping. Preserves 100% of native sensor photons directly from camera hardware.
- **Zero Decimation / Full Megapixels:** Full $4032 \times 3024$ native Bayer data is maintained throughout capture and debayering without spatial pixel binning or CFA decimation for 4K video recording.
- **Ultra-Fast Metal Compute Pipeline:**
  - **Fused Compute Kernel:** Bayer demosaicing, optical lens shading correction ($\cos^4\theta$), white balance gains, sensor-to-gamut color correction matrices, and Log OETF encoding are evaluated in a single GPU pass, slashing memory bandwidth by over 130 MB per 4K frame.
  - **Linear Demosaic Factoring:** Malvar-He-Cutler directional demosaicing operates directly on sensor digital numbers (DN) with a single normalization step, eliminating over 300 million arithmetic instructions per frame.
  - **Hardware Bilinear Resampling:** Replaced heavy multi-tap sinc convolutions with single-pass GPU Texture Mapping Unit (TMU) filtering, reducing resample latency from ~24 ms to ~1.5 ms.
  - **Direct 10-Bit YCbCr 4:2:0 Encoding:** Converts linear/log RGB directly to ITU-R BT.2020 10-bit Video Range (`kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange`) with hardware-accelerated single-cycle FMA quantization.
- **Cinema Log Color Science:**
  - **Apple Log 2 & Sony S-Log3:** Standards-compliant implementation of published Apple Log 2 and Sony S-Log3 (S-Gamut3.Cine) opto-electronic transfer functions.
  - **Smooth $C^1$-Continuous Highlight Shoulder:** Hermite cubic highlight roll-off with terminal slope $s_1 = 0.0$ and pure norm-preserving chromaticity scaling, eliminating clipping cliffs, posterization rings, and noise-induced temporal saturation shimmer.
  - **Calibrated Physical $\cos^4\theta$ Lens Shading Correction:** Dynamic concentric radial optical model counteracting lens vignetting across Ultra-Wide, Wide, and Telephoto lenses.
- **Rock-Solid Constant Frame Rate (CFR):** Strict 24.000 fps and 30.000 fps timeline synchronization with pre-warmed buffer pools, sub-millisecond audio sync, and startup stall bridging to guarantee zero dropped or frozen hold frames.
- **Dual Viewfinder Monitoring:**
  - **Real-Time Rec.709 Preview (709):** One-touch toggle between unadulterated Log monitoring and a calibrated Rec.709 display LUT, with high-frequency focus peaking and diagonal clipping zebras.
- **Professional Cinematography Tools:** Real-time 6-axis gyroscope horizon level, framing grids, and live RGB histogram & waveform scopes.
- **Comprehensive Audio Monitoring:** Real-time level meters with automatic detection for built-in, USB, Bluetooth, and 3.5mm external microphones.

---

## Framing & Recording Modes

| Mode | Capture Resolution | Aspect Ratio | Max Bitrate | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Open Gate 4:3** | $1920 \times 1440$ | 4:3 | 100 Mbps | Full sensor vertical capture with maximum reframing flexibility |
| **1080p Full HD** | $1920 \times 1080$ | 16:9 | 100 Mbps | Standard broadcast HD delivery with native 10-bit Log |
| **4K UHD** | $3840 \times 2160$ | 16:9 | 150 Mbps | Pristine 4K capture from native 12.2MP Bayer sensor data |

---

## Color Grading in DaVinci Resolve

OwLens video files contain standard NCLC color primaries, matrix, and transfer function metadata, making them effortless to grade in DaVinci Resolve using Color Space Transform (CST) nodes:

### Apple Log 2
- **Input Color Space:** `Rec.2020`
- **Input Gamma:** `Apple Log`
- **Output Color Space:** `Rec.709` (or timeline working color space)
- **Output Gamma:** `Rec.709` / `Gamma 2.4`

### Sony S-Log3
- **Input Color Space:** `Sony S-Gamut3.Cine`
- **Input Gamma:** `Sony S-Log3`
- **Output Color Space:** `Rec.709` (or timeline working color space)
- **Output Gamma:** `Rec.709` / `Gamma 2.4`

---

## Installation & Setup

### Method 1: Sideloading with Sideloadly (Recommended for Mac & Windows — No Xcode Required!)

This is the easiest way to install OwLens on your iPhone or iPad without needing Xcode or a paid developer account.

1. **Download `OwLens.ipa`:**
   - Grab the latest `OwLens.ipa` from the [Releases](https://github.com/jeetdoesthings/OwLens/releases) page.

2. **Install Sideloadly:**
   - Download and install <a href="https://sideloadly.io" target="_blank" rel="noopener noreferrer">Sideloadly</a> (available for both macOS and Windows).

3. **Connect Your iPhone/iPad:**
   - Connect your device to your computer via USB (tap **Trust This Computer** if prompted).

4. **Sideload the App:**
   - Open Sideloadly.
   - Drag and drop `OwLens.ipa` into the IPA icon box in Sideloadly.
   - Enter your Apple ID email in the **Apple account** field.
   - Click **Start** and enter your password / 2-factor authentication code when prompted. Sideloadly securely communicates with Apple to sign the app for your device.

5. **Trust the Developer Profile on iPhone:**
   - On your device, navigate to **Settings** → **General** → **VPN & Device Management**.
   - Under **Developer App**, tap your Apple ID.
   - Tap **Trust "[Your Apple ID]"** and confirm.
   - Launch **OwLens** from your home screen and enjoy cinema-grade Log recording!

---

### Method 2: Build from Source via Xcode (macOS)

#### Requirements
- Mac running macOS Sonoma or later with Xcode 15+
- Physical iPhone running iOS 17.0+ (RAW capture requires physical camera hardware)
- Apple Developer Account (free personal account works for device provisioning)

#### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jeetdoesthings/OwLens.git
   cd OwLens
   ```

2. **Open the project in Xcode:**
   ```bash
   open OwLens.xcodeproj
   ```

3. **Configure code signing:**
   - In Xcode, select the **OwLens** target.
   - Under **Signing & Capabilities**, check **Automatically manage signing** and select your **Team** (Apple ID).

4. **Build and install on device:**
   - Connect your iPhone and select it as the run target.
   - Press **Cmd + R** to compile and launch.
   - On iOS, navigate to **Settings** → **General** → **VPN & Device Management** and tap **Trust** on your developer certificate (first launch only).

---

## License

OwLens is a source-available project. The source code is provided for personal, educational, and evaluation purposes only. Commercial exploitation, distribution, and publishing to any public app store (including the Apple App Store) are strictly prohibited. Refer to the [LICENSE](LICENSE) for full terms.

