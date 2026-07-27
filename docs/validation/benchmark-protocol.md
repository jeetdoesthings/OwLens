# Benchmark Protocol

**Branch:** `denoise-upgrades`
**Targets:**
- 1080p path: ≤25 ms/frame
- 4K path: ≤50 ms/frame

## Frame-time benchmark

DEBUG builds already print a per-frame timing line:

```
[MetalPipeline] frame time: 12.34 ms
```

### How to capture

1. Build and run on the target device (iPhone 12 Pro or later) in DEBUG.
2. Record the `[MetalPipeline] frame time:` log lines for 30 seconds in each mode.
3. Compute the 95th percentile frame time.

### Expected results

| Resolution | Target p95 | Notes |
|---|---|---|
| 1080p | ≤25 ms | Local-σ skipped at 4K; active at 1080p |
| 4K | ≤50 ms | `estimateLumaVariance` is skipped when `bayerW >= 3000` |

> **TODO:** Paste observed p95 values after testing on target hardware.

## Visual A/B recording protocol

Record 3-second clips on iPhone 12 Pro:

1. Static low-light wall (ISO 800, 1/30s)
2. Slow pan across high-contrast edge
3. Fast hand-wave

Compare across branches:
- `main` baseline
- Phase 1 (critical bug fixes)
- Phase 4 (local-σ spatial)
- Phase 6 (calibrated)

Inspect at 200% zoom in DaVinci Resolve:

| Metric | How to measure |
|---|---|
| Chroma noise variance | Sample a flat 64×64 region, read RGB parade variance |
| Luma edge width | Measure 10-90% rise on a high-contrast edge |
| Temporal stability | Frame-to-frame pixel difference in static region |

> **TODO:** Add measured values and grading notes after physical-device capture.

## Sign-off

- [ ] 1080p p95 frame time ≤25 ms
- [ ] 4K p95 frame time ≤50 ms
- [ ] Visual A/B clips recorded and reviewed
- [ ] No regressions in color or edge detail relative to `main`
