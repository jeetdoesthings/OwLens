# OwLens

<img src="owlens-logo.png" width="128" height="128" alt="OwLens Logo" />

🦉 **OwLens** is a professional-grade iOS cinema camera app designed for filmmakers and creators who want true control over their video capture. It bypasses Apple's standard image signal processor (ISP), capturing uncompressed 14-bit RAW sensor data and encoding it directly into 10-bit HEVC Log in real-time.

## Features

- **True Sensor RAW to Log Pipeline:** Captures unprocessed 14-bit RAW Bayer sensor data straight from the camera hardware. No artificial sharpening, harsh noise reduction, or unwanted tone mapping from the default camera app.
- **Apple Log 2 & Sony S-Log3 Profiles:** Record in standard **Apple Log 2** or **Sony S-Log3** with calibrated color profiles for maximum dynamic range and flexibility in post-production.
- **Minimalist Monochromatic UI:** Clean, distraction-free interface with glass HUD surfaces designed to keep focus on framing and exposure.
- **Dual Viewfinder Modes:**
  - **Normal Video (VID):** Hardware ISP preview for smooth monitoring with zero GPU overhead.
  - **Log Preview (LOG):** Real-time debayered log preview with focus peaking and zebra clipping indicators.
- **Manual Camera Controls:** Continuous sliders for ISO, shutter angle (with cinematic snap points like 180°), and Kelvin white balance.
- **Framing Options:**
  - **Open Gate 4:3 (1920×1440):** Full sensor capture with no crop for maximum reframing freedom.
  - **1080p & 4K UHD 16:9:** Standard delivery resolutions at bitrates up to 150 Mbps.
- **Constant Frame Rate (CFR):** Rock-solid 24 fps and 30 fps capture to ensure seamless audio synchronization in your NLE.
- **Cinematography Tools:** Real-time 6-axis gyroscope level, rule-of-thirds grid, and live RGB histogram & waveform scopes.
- **External Audio Support:** Automatic detection and real-time level meters for USB, Bluetooth, and 3.5mm microphones.

## Color Grading in DaVinci Resolve

OwLens footage grades effortlessly with standard Color Space Transform (CST) nodes in DaVinci Resolve:

### Apple Log 2
- **Input Color Space:** `Rec.2020`
- **Input Gamma:** `Apple Log`
- **Output Color Space:** `Rec.709` (or timeline color space)
- **Output Gamma:** `Rec.709` / `Gamma 2.4`

### Sony S-Log3
- **Input Color Space:** `Sony S-Gamut3.Cine`
- **Input Gamma:** `Sony S-Log3`
- **Output Color Space:** `Rec.709` (or timeline color space)
- **Output Gamma:** `Rec.709` / `Gamma 2.4`

## Installation

You'll need:
- A Mac running Xcode 15 or later
- An iPhone running iOS 17 or later
- An Apple ID (free personal account works)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jeetdoesthings/OwLens.git
   cd OwLens
   ```

2. **Open the project:**
   ```bash
   open OwLens.xcodeproj
   ```

3. **Configure signing:**
   - Select the **OwLens** target in Xcode.
   - Under **Signing & Capabilities**, choose your **Team** (your Apple ID).

4. **Run on your device:**
   - Connect your iPhone and select it as the run destination.
   - Press **Cmd + R** to build and install.
   - On your iPhone, go to **Settings** → **General** → **VPN & Device Management** and tap **Trust** on your developer profile (first install only).

That's it, OwLens will launch on your phone! 🦉

## License

OwLens is a source-available project. The source code is provided for personal, educational, and evaluation purposes only. Commercial exploitation, distribution, and publishing to any public app store (including the Apple App Store) are strictly prohibited. Refer to the [LICENSE](LICENSE) for full terms.
