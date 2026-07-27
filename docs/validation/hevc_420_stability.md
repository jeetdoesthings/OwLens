# HEVC 4:2:0 Stability Validation

This log tracks the Phase 0 validation task from `docs/IMPLEMENTATION_PLAN.md`. OwLens must prove the current HEVC 4:2:0 constant-frame-rate recording path on physical hardware before heavier preview tools are layered on top.

## Goal

Confirm that long recordings remain playable, synchronized, and correctly reported as 24 or 30 fps while the app captures Bayer RAW frames, processes them through Metal, and writes HEVC with audio.

## Device Matrix

| Date | Device | iOS | Lens | Format | FPS | Bitrate | Audio source | Battery | Thermal start | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-19 | iPhone 12 Pro (`iPhone13,3`) | iOS 18+ | Wide 1x | Open Gate | 24 | 100 Mbps | TBD | TBD | TBD | Pass reported |

## Required Test Runs

1. Cold start, 10+ minute recording.
2. Already-warm device, 10+ minute recording.
3. Low battery, 10+ minute recording.
4. Background apps active, 10+ minute recording.
5. 24 fps and 30 fps coverage for the main target format.
6. Built-in mic and at least one external mic, if available.

## During Recording

Capture these observations from Xcode logs and Instruments:

- `AVAssetWriterInput.isReadyForMoreMediaData` false-return frequency.
- `VideoWriter` reported `timeline`, `real`, `holds`, and `drops`.
- `RawFrameBuffer` dropped frame count.
- `ProcessInfo.thermalState` transitions.
- App responsiveness while recording.
- Any preview stalls, GPU warnings, or memory spikes.

## After Recording

For every clip, verify:

- File opens in Photos or Files from the selected save path.
- File imports into DaVinci Resolve or Premiere.
- Container frame rate reports the selected CFR value.
- Duration matches wall-clock recording duration.
- Audio remains in sync at the start, middle, and end.
- No visible frame ordering issues, corrupt frames, or unexpected color transform.
- Log payload remains flat and gradeable; no Rec.709 preview transform is baked in.

## Result Template

### Run TBD

- Date:
- Device:
- iOS:
- Format/FPS/bitrate:
- Lens:
- Audio:
- Starting battery:
- Starting thermal state:
- Recording duration:
- Timeline frames:
- Real frames:
- Held frames:
- Dropped frames:
- Writer readiness issues:
- Ending thermal state:
- Playback result:
- Audio sync result:
- NLE import result:
- Notes:
- Pass/fail:

### Run 2026-07-19 - iPhone 12 Pro

- Date: 2026-07-19
- Device: iPhone 12 Pro (`iPhone13,3`, A14)
- iOS: iOS 18+ based on shutter suppression log
- Format/FPS/bitrate: Open Gate, 24 fps, 100 Mbps
- Lens: Wide 1x selected; Ultra Wide and Telephoto discovered
- Audio: TBD
- Starting battery: TBD
- Starting thermal state: TBD
- Recording duration: user-reported stable
- Timeline frames: TBD
- Real frames: TBD
- Held frames: TBD
- Dropped frames: TBD
- Writer readiness issues: none reported
- Ending thermal state: TBD
- Playback result: pass reported
- Audio sync result: pass reported
- NLE import result: TBD
- Notes: Device exposed Bayer RAW `'rgg4'`, CFA override RGGB, verified device true, recommended Open Gate at 24 fps and 100 Mbps. Logged CoreMotion managed-preference warning and Fig errors appear non-fatal in this run.
- Pass/fail: Pass reported

## Pass Criteria

A configuration passes when:

- The writer completes without `.failed` status.
- The clip plays from beginning to end.
- Audio sync is acceptable at the end of a 10+ minute take.
- Frame rate metadata matches the selected 24 or 30 fps mode.
- Held frames do not create timestamp drift.
- Thermal pressure does not corrupt the recording.

## Failure Triage

If a run fails, record the exact symptom before changing code:

- Writer failure or append failure.
- Audio drift.
- File duration mismatch.
- Excessive held frames.
- App background/GPU error.
- Photos save failure.
- NLE import or metadata issue.

Then map the likely owner:

- `OwLens/Encoding/VideoWriter.swift` for writer, CFR, audio timing, or metadata issues.
- `OwLens/Processing/MetalPipeline.swift` for GPU throughput or hot-path allocation issues.
- `OwLens/UI/CameraViewModel.swift` for recording lifecycle, Photos save, or app active/inactive handling.
- `OwLens/Capture/CaptureController.swift` for device format, RAW capture, lens, or mic routing issues.
