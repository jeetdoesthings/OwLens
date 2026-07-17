# OwLens Landing Page Architecture & Design

This document details the design system, page structure, content strategy, and interactive mechanisms for the **OwLens Landing Page**. 

The webpage source is stored inside the local folder:
📁 [OwLens Landing Page Design/src/app/App.tsx](file:///Users/jeet/Desktop/OwLens/OwLens%20Landing%20Page%20Design/src/app/App.tsx)

---

## 1. Design System & Style Guidelines

The landing page follows a minimalist, premium "hardware-centric" design language matching Apple's own product promotion pages. It uses a high-contrast **Light Mode** to present raw S-Log3 and color-graded image profiles cleanly.

### Typography
- **Primary Font Family:** Helvetica, Helvetica Neue, or the default system sans-serif (San Francisco) on iOS/macOS.
- **Font Weighting:** Standard weights are used to build clean hierarchies:
  - Headers (`h1`, `h2`): `700` (Bold) with slight letter-spacing reduction (`-1px` to `-0.5px`) for a tight, modern aesthetic.
  - Subheadings (`h3`): `600` (Semi-bold).
  - Body & Captions: `400` / `500` (Muted weights).

### Color Palette (CSS Custom Variables)
- **Background (`--bg-color`):** Pure white (`#FFFFFF`) to represent clean, raw light.
- **Base Text (`--text-color`):** Solid off-black charcoal (`#111111`) to prevent the harsh contrast of pure black while retaining sharp readability.
- **Muted Text (`--text-muted`):** Medium grey (`#666666`) for structural sub-headers, captions, and secondary details.
- **Borders (`--border-color`):** Very thin light grey lines (`#E5E5E5`) to separate cards and sections without visual clutter.
- **Accent Elements (`--accent-color`):** Strict pitch black (`#000000`) for primary buttons and call-to-actions, matching the professional cinema camera layout.
- **Card Backgrounds (`--card-bg`):** Soft off-white (`#FAFAFA`) to group specifications and features into distinct containers.

---

## 2. Page Content & Navigation Flow

### Header Navigation
- Displays the **OwLens App Icon** (`AppIcon.png`) aligned alongside the bold title.
- Right-aligned icon navigation list directing visitors to:
  - **GitHub Logo (SVG):** Links directly to the code repository: `https://github.com/jeetdoesthings/OwLens`
  - **LinkedIn Logo (SVG):** Links directly to your professional profile: `https://www.linkedin.com/in/jeet-ghegade/`
  - **Instagram Logo (SVG):** Links directly to your social profile: `https://www.instagram.com/jeetpls/`

### Hero Stage
- Displays a prominent 100x100px version of the `AppIcon.png`.
- A bold title: **12-bit RAW S-Log3 Video on iPhone**.
- Explains the core value proposition: Bypassing Apple's built-in tone-mappers and noise-reducers to give creators control of their raw image data.
- Call-to-Action button linked directly to the GitHub page with a clean SVG GitHub icon embedded.

### Section 1: ISP Bypass Comparison
- Highlights the visual differences between Apple's standard post-processing and OwLens RAW S-Log3 captures.
- **Left Card:** Standard iPhone Video (highlights over-processed sharpening, artificial local tone mapping, and digital smoothing).
- **Right Card:** OwLens RAW S-Log3 (highlights flat dynamic log curves, natural texture detail, and clean highlight roll-off).

### Section 2: Interactive Log vs. Graded Slider
- An interactive container using pointer events for comparing unprocessed **Log** footage against a **Color Graded** version.
- Allows users to drag a split bar left or right to wipe the color correction on and off.
- Labeled overlays clearly identify "LOG" (flat, low-contrast sensor state) and "GRADED" (saturated, cinematic contrast).

### Section 3: Feature Specifications
Six grid cards detail the primary functions built into the app, each styled with an custom vector SVG icon matching the system theme colors:
1. **Manual Exposure (Shutter/ISO Dial SVG):** Locking ISO and shutter speeds to prevent auto-adjustments.
2. **White Balance (Thermometer SVG):** Explicit Kelvin and tint adjustment inside the linear domain.
3. **Constant Frame Rate (Cinema Camera SVG):** Ensuring a locked 24/30fps timeline for perfect audio sync.
4. **Dynamic Lens Switch (Double Lens Ring SVG):** Auto-detecting physical lenses on standard or Pro models.
5. **Audio Monitoring (Microphone SVG):** Live level meters supporting internal and external audio routes.
6. **Composition Aids (3x3 Grid SVG):** Rule-of-thirds grid lines and an integrated gyroscope level sensor.

### Section 4: "How It Works" Pipeline
A numbered vertical steps sequence explaining the engineering path of a frame:
- **Step 01 - Direct Sensor Stream:** Bypassing Apple's ISP filters using `AVCapturePhotoOutput` to stream Bayer14 data.
- **Step 02 - Fused Metal Shader:** bilinear demosaicing, scaling white balance gains, and S-Log3 OETF conversion in a single GPU pass.
- **Step 03 - HEVC Hardware Encode:** Encoding S-Log3 textures as 10-bit HEVC files at up to 200 Mbps with locked CFR.

### Footer Section
- Copyright details: `© 2026-present Jeet. All rights reserved.`
- Centered social navigation group rendering the matching vector SVGs for **GitHub**, **LinkedIn**, and **Instagram** pointing to your verified links.

---

## 3. Interactive Image Slider Mechanism

The slider operates using simple, lightweight vanilla CSS and JavaScript inside `index.html`:
- The container has `relative` positioning and holds two images stacked directly on top of each other.
- The **Graded Image** acts as the top layer (`img-after`) inside a wrapper div configured with `overflow: hidden`.
- The **Log Image** stays stationary underneath as the base layer (`img-before`).
- When a user moves their mouse or drags a finger across the container, the Javascript calculates the pointer's percentage horizontal coordinate relative to the container's width.
- It dynamically updates:
  1. The width of the top wrapper div (`img-after`) to clip it.
  2. The position of the vertical white divider line (`slider-handle`).
- **Image Fallbacks:** If the image files (`normal_screenshot.jpg`, `log_screenshot.jpg`, `log_image.jpg`, or `graded_image.jpg`) are missing from the folder, custom `onerror` events automatically catch the error, hide the broken image icons, and render neat styled light grey placeholder boxes with text descriptors so the site remains looking polished.

---

## 4. Setting Up Your Media Files
To make the site fully live with your own screenshots, place the following images inside the `website/` directory alongside `index.html`:
1. **`normal_screenshot.jpg`**: A screenshot showing standard processed iPhone video (harsh contrast/over-sharpened).
2. **`log_screenshot.jpg`**: A screenshot showing the raw flat S-Log3 video preview interface.
3. **`log_image.jpg`**: The flat, unprocessed log comparison frame for the slider.
4. **`graded_image.jpg`**: The exact same comparison frame after applying a cinematic color grade LUT.
