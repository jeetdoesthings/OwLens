# RAW Format Audit

**Device under test:** iPhone 12 Pro (physical)
**Branch:** `denoise-upgrades`
**Date:** *(to be filled on physical device run)*

## Purpose

Capture the exact Bayer RAW capabilities reported by `AVCapturePhotoOutput` on the reference device. This establishes the ground truth for sensor resolution, available RAW pixel formats, and whether multi-resolution RAW is exposed.

## How to capture

1. Build and run OwLens on a physical iPhone 12 Pro in DEBUG configuration.
2. Look for console logs prefixed with `[DeviceCapabilities]` during app launch.
3. Copy the relevant lines below under **Observed output**.

## Expected findings

- `AVCapturePhotoOutput` exposes exactly one Bayer RAW format.
- The Bayer RAW format is delivered at the sensor's native resolution only.
- No multi-resolution RAW option is available from `photoOutput.availableRawPhotoPixelFormatTypes`.
- iOS does not provide sensor-level binning; any downscale must be done in post (Metal `binBayerCFA`).

## Observed output

```
[DeviceCapabilities] RAW probe camera type=AVCaptureDeviceTypeBuiltInWideAngleCamera virtual=false
[DeviceCapabilities] RAW probe sensor resolution=... dimensions=...
[DeviceCapabilities] RAW probe all=[...] bayer=[...]
[DeviceCapabilities] RAW probe sensor native resolution = ...×... (all bayer formats use this resolution)
[DeviceCapabilities] RAW probe: 1 bayer format(s), single resolution only (sensor native)
```

> **TODO:** Paste the actual console output from a physical iPhone 12 Pro run here.

## Implications for pipeline design

1. The debayer/denoise pipeline must accept the full sensor resolution and downscale via compute, because iOS does not expose a smaller RAW stream.
2. The `binBayerCFA` 2× reduction is safe only when the reduced dimensions still cover the requested encode size, preventing output upscaling artifacts.
3. Temporal history buffers are allocated at either full sensor or half-reduced resolution; no other sensor binning modes need to be supported.

## Sign-off

- [ ] Console output captured on physical iPhone 12 Pro
- [ ] Exactly one Bayer RAW format listed
- [ ] Bayer RAW resolution matches sensor native resolution
- [ ] No multi-resolution RAW option observed
