# OwLens

<img src="owlens-mark.svg" width="128" height="128" alt="OwLens Logo" />

**OwLens** is a professional-grade iOS camera application designed for filmmakers and power users who want maximum control over their video capture. It bypasses Apple's standard image signal processor (ISP) pipeline, capturing uncompressed 12-bit RAW sensor data and encoding it directly into HEVC S-Log3 in real-time using custom Metal shaders.

## Features

- **True RAW to S-Log3 Pipeline:** OwLens captures uncompressed RAW Bayer sensor data (Bayer14) and debayers it on the GPU, applying a true S-Log3 transfer function before saving it as 10-bit HEVC. This produces an unprocessed video straight from the phone sensor, with no artificial sharpening, noise reduction, or local tone mapping from the Apple ISP.
- **Open Gate 4:3 Capture:** Captures the full aspect ratio of the camera sensor without cropping it to 16:9, providing the maximum vertical resolution and total flexibility for reframing in post-production.
- **Constant Frame Rate (CFR):** Guarantees locked 24fps or 30fps files by holding the last valid frame if the camera drops a frame, ensuring perfectly synced audio in NLEs like Premiere Pro and DaVinci Resolve.
- **Manual Controls:** Full manual control over ISO (50–2000), White Balance (Kelvin), and Focus.
- **Cinematic Shutter Angles:** Control motion blur using standard cinematic shutter angles (from 11.25° to 360°) on a continuous magnetic slider, enforcing a true 180° shutter rule when changing frame rates.
- **Focus Peaking & Tap-to-Focus:** High-performance Metal-accelerated Focus Peaking (green edge highlight) for zero-overhead manual focus tracking. Includes a tap-to-focus gesture with a visual reticle and single-shot AF to completely eliminate optical image stabilization (OIS) jitter during pans.
- **Dynamic Lens Switching:** Automatically detects all available single-lens physical cameras on the device (Ultra Wide, Wide, Telephoto) and allows seamless switching.
- **Audio Control:** Automatically detects external microphones (USB, Headset, Bluetooth) and falls back to the high-quality built-in iPhone mic. Includes a real-time audio level monitor.
- **Professional Overlays:** Includes a rule-of-thirds grid, a real-time hardware gyroscope level overlay, and clipping indicators (zebras) to ensure perfectly straight and exposed shots.
- **Metal Accelerated:** Uses a fused Metal compute kernel for simultaneous Debayering, White Balance scaling, and S-Log3 conversion, eliminating CPU bottlenecks and minimizing thermal throttling.

## How it Works

OwLens bypasses standard iOS image processing using a three-stage custom pipeline:

1. **Direct Sensor Access:** The app queries the camera hardware directly to capture uncompressed 12-bit RAW Bayer sensor data, bypassing Apple's built-in noise reduction, sharpening, and tone-mapping.
2. **Metal-Accelerated GPU Pipeline:** A custom Metal compute shader runs directly on the GPU to perform real-time bilinear debayering, apply white balance gains in the linear color space, and apply the Sony S-Log3 transfer function.
3. **Hardware Encoding:** The resulting log-encoded texture is fed directly to the iPhone's H.265 (HEVC) hardware encoder at bitrates up to 200 Mbps, maintaining a constant frame rate (CFR) to ensure audio sync.

## Color Grading in DaVinci Resolve

When importing OwLens footage into DaVinci Resolve, use the following settings in a **Color Space Transform (CST)** node to map the colors and contrast correctly:

- **Input Color Space:** Sony S-Gamut3.Cine
- **Input Gamma:** Sony S-Log3

## Installation

You'll need:

- A Mac running Xcode 15 or later
- An Apple ID (a free account is enough to install on your own device)
- Any iPhone running iOS 17 or later connected via cable, or on the same Wi-Fi network as your Mac

### Steps

1. Download the project:
   ```bash
   git clone https://github.com/[your-username]/owlens.git
   cd owlens
   ```
2. Install XcodeGen, if you don't already have it (used to generate the Xcode project file):
   ```bash
   brew install xcodegen
   ```
3. Generate the project:
   ```bash
   xcodegen generate
   ```
4. Open the project:
   ```bash
   open OwLens.xcodeproj
   ```
5. Sign the app with your Apple ID:
   * In Xcode, click the **OwLens** project in the left sidebar.
   * Under **Signing & Capabilities**, select your name under **Team**. (If you don't see your Apple ID listed, go to **Xcode** → **Settings** → **Accounts** and add it there first.)
6. Connect your iPhone to your Mac, and select it as the run destination from the device dropdown at the top of the Xcode window.
7. Run it: press **Cmd + R**, or click the ▶️ button.
8. Trust the developer certificate on your iPhone (first run only):
   * Go to **Settings** → **General** → **VPN & Device Management** on your iPhone.
   * Tap your Apple ID under "Developer App," then tap **Trust**.

That's it, OwLens will launch on your phone.

## License

OwLens is a source-available project. The source code is provided for personal, educational, and evaluation purposes only. Commercial exploitation, distribution, and publishing to any public app store (including the Apple App Store) are strictly prohibited. Refer to the [LICENSE](LICENSE) for full terms.
