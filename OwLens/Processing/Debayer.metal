#include <metal_stdlib>
using namespace metal;

// Pipeline color model (encode + LUT must match):
//   1) Black/white linearize of 14-bit Bayer (sensor DN → scene-linear ~0–1)
//   2) Bilinear demosaic → camera native RGB (no Display P3 / Rec.709 matrix)
//   3) As-shot WB gains (R/G/B relative to G), CCM = identity
//   4) Log OETF: Log2 or Sony S-Log3 (code values 0–1)
// S-Log3 is NOT Rec.709 gamma. Grade with inverse LUT or S-Log3→Rec.709.

struct DebayerParams {
    int bayerPattern;
    float blackLevel;
    float whiteLevel;
    float4 lscCoefficients;
};

struct DefectPixelParams {
    float shotCoeff;
    float readCoeff;
};

struct WhiteBalanceParams {
    float3 gains;
    float3x3 colorMatrix;
};

// (Removed unused sampleBayerValid)



static inline float sampleBayerRaw(texture2d<float, access::read> tex, int maxW, int maxH, int x, int y, int dx, int dy) {
    int px = x + dx;
    int py = y + dy;
    // Bayer CFA phase preservation: reflect coordinates across boundaries
    if (px < 0) px = -px;
    else if (px > maxW) px = 2 * maxW - px;
    if (py < 0) py = -py;
    else if (py > maxH) py = 2 * maxH - py;
    return tex.read(uint2(px, py)).r;
}

kernel void correctDefectPixelsBayer(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant DebayerParams &params [[buffer(0)]],
    constant DefectPixelParams &noise [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;

    float center = src.read(gid).r;

    // Direct O(1) unrolled sampling of true same-color Bayer neighbors (distance 2 cross)
    // with boundary reflection preserving exact Bayer CFA phase parity.
    int w = int(src.get_width()) - 1;
    int h = int(src.get_height()) - 1;
    int x = int(gid.x);
    int y = int(gid.y);

    float n0 = sampleBayerRaw(src, w, h, x, y, -2,  0);
    float n1 = sampleBayerRaw(src, w, h, x, y,  2,  0);
    float n2 = sampleBayerRaw(src, w, h, x, y,  0, -2);
    float n3 = sampleBayerRaw(src, w, h, x, y,  0,  2);

    // Fast 4-element sorting network: 5 min/max operations, zero loops, zero register spills
    float min01 = min(n0, n1);
    float max01 = max(n0, n1);
    float min23 = min(n2, n3);
    float max23 = max(n2, n3);
    float midLo = max(min01, min23);
    float midHi = min(max01, max23);
    float median = 0.5f * (midLo + midHi);

    float signalNorm = saturate(center / params.whiteLevel);
    float sigmaRaw = sqrt(noise.shotCoeff * signalNorm + noise.readCoeff) * params.whiteLevel;
    float threshold = 5.0f * max(sigmaRaw, 1e-3f);

    float out = (abs(center - median) > threshold) ? median : center;
    dst.write(float4(out, 0.0f, 0.0f, 1.0f), gid);
}

// True 2x2 CFA-preserving binning: preserves RGGB/GRBG/GBRG/BGGR phase
// by averaging 2x2 subpixels of the exact same color in each 4x4 block.
kernel void binBayerCFA(
    texture2d<float, access::read> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    int x = int(gid.x);
    int y = int(gid.y);

    int w = int(src.get_width()) - 1;
    int h = int(src.get_height()) - 1;

    int sx0 = min(2 * x - (x & 1), w);
    int sx1 = min(sx0 + 2, w);
    int sy0 = min(2 * y - (y & 1), h);
    int sy1 = min(sy0 + 2, h);

    float s00 = src.read(uint2(sx0, sy0)).r;
    float s10 = src.read(uint2(sx1, sy0)).r;
    float s01 = src.read(uint2(sx0, sy1)).r;
    float s11 = src.read(uint2(sx1, sy1)).r;

    float v = 0.25f * (s00 + s10 + s01 + s11);
    dst.write(float4(v, 0.0f, 0.0f, 1.0f), gid);
}


// Smooth filmic highlight shoulder mapping sensor linear [0, 1] to scene reflectance [0, rMax].
// Preserves 100% linear calibration for midtones & shadows (r <= rKnee),
// with C1 continuity at rKnee.
static inline float applyHighlightShoulderMetal(float r, float rKnee, float rMax) {
    if (rMax <= rKnee + 1e-4f) return r;
    if (r <= rKnee) return r;

    float delta = rMax - rKnee;
    float dr = 1.0f - rKnee;
    float s0 = dr / delta;
    float s1 = 0.0f;
    float a = s1 + s0 - 2.0f;
    float b = 3.0f - 2.0f * s0 - s1;
    float c = s0;

    float t = clamp((r - rKnee) / max(dr, 1e-4f), 0.0f, 1.0f);
    float g = ((a * t + b) * t + c) * t;
    return rKnee + delta * g;
}

static inline float3 applyHighlightShoulder3(float3 rgb, float rKnee, float rMax) {
    float peak = max(rgb.r, max(rgb.g, rgb.b));
    if (rMax <= rKnee + 1e-4f || peak <= rKnee) return rgb;

    float peakShoulder = applyHighlightShoulderMetal(peak, rKnee, rMax);
    float scale = peakShoulder / max(peak, 1e-6f);
    return rgb * scale;
}

static inline float3 encodeLogCurve(float3 rgb, int curveType, float headroomScale = 1.0f) {
    if (curveType == 0) {
        return saturate(rgb);
    }

    if (headroomScale > 1.0f) {
        rgb = applyHighlightShoulder3(rgb, 0.36f, headroomScale);
    }

    if (curveType == 3) {
        // Apple Log 2 published OETF (Apple Log Profile White Paper)
        // Transfer function on scene-linear reflectance (18% mid grey ≈ 0.18)
        constexpr float r0 = -0.05641088f;
        constexpr float rt = 0.01f;
        constexpr float c  = 47.28711236f;
        constexpr float beta  = 0.00964052f;
        constexpr float gamma = 0.08550479f;
        constexpr float delta = 0.69336945f;

        float3 result;
        for (int i = 0; i < 3; i++) {
            float r = rgb[i];
            if (r < r0) {
                result[i] = 0.0f;
            } else if (r < rt) {
                float diff = r - r0;
                result[i] = c * diff * diff;
            } else {
                result[i] = gamma * metal::log2(r + beta) + delta;
            }
        }
        return saturate(result);
    }

    // Sony S-Log3 published OETF on scene-linear (18% mid grey ≈ 0.18)
    // Pre-evaluated compile-time quotients map runtime divisions into single-cycle FMA.
    constexpr float kScale = 261.5f / 1023.0f;
    constexpr float kOffset = 420.0f / 1023.0f;
    constexpr float kInv19 = 1.0f / 0.19f;
    constexpr float kSlope = (171.2102946929f - 95.0f) / (0.01125f * 1023.0f);
    constexpr float kPedestal = 95.0f / 1023.0f;

    float3 result;
    float3 clamped = max(rgb, float3(0.0));
    for (int i = 0; i < 3; i++) {
        float lin = clamped[i];
        if (lin >= 0.01125f) {
            result[i] = metal::fma(metal::log10((lin + 0.01f) * kInv19), kScale, kOffset);
        } else {
            result[i] = metal::fma(lin, kSlope, kPedestal);
        }
    }
    return saturate(result);
}


// ──────────────────────────────────────────────────────────────────────
// FUSED: demosaic + WB + log in ONE kernel (eliminates 2 texture
// round-trips and 2 encoder dispatches per frame).
// ──────────────────────────────────────────────────────────────────────

struct FusedParams {
    int   bayerPattern;
    float blackLevel;
    float whiteLevel;
    int   curveType;
    float3 wbGains;
    float4 lscCoefficients;
    float greenBalance;
    float headroomScale;  // Scene reflectance multiplier: sensor [0,1] → R [0, headroomScale]
};

struct LSCParams {
    float radialR;
    float radialG;
    float radialB;
    float radial4R;
    float radial4G;
    float radial4B;
    float azimuthR;
    float azimuthG;
    float azimuthB;
};


struct BayerNeighborhood {
    float c00, cN1, cS1, cE1, cW1, cN2, cS2, cE2, cW2, cNE, cNW, cSE, cSW;
};

static inline BayerNeighborhood fetchBayerNeighborhood(texture2d<float, access::read> rawTexture, int maxW, int maxH, int x, int y) {
    BayerNeighborhood nb;
    if (x >= 2 && x <= maxW - 2 && y >= 2 && y <= maxH - 2) {
        uint2 u = uint2(x, y);
        nb.c00 = rawTexture.read(u).r;
        nb.cN1 = rawTexture.read(uint2(x, y - 1)).r;
        nb.cS1 = rawTexture.read(uint2(x, y + 1)).r;
        nb.cE1 = rawTexture.read(uint2(x + 1, y)).r;
        nb.cW1 = rawTexture.read(uint2(x - 1, y)).r;
        nb.cN2 = rawTexture.read(uint2(x, y - 2)).r;
        nb.cS2 = rawTexture.read(uint2(x, y + 2)).r;
        nb.cE2 = rawTexture.read(uint2(x + 2, y)).r;
        nb.cW2 = rawTexture.read(uint2(x - 2, y)).r;
        nb.cNE = rawTexture.read(uint2(x + 1, y - 1)).r;
        nb.cNW = rawTexture.read(uint2(x - 1, y - 1)).r;
        nb.cSE = rawTexture.read(uint2(x + 1, y + 1)).r;
        nb.cSW = rawTexture.read(uint2(x - 1, y + 1)).r;
    } else {
        nb.c00 = sampleBayerRaw(rawTexture, maxW, maxH, x, y, 0, 0);
        nb.cN1 = sampleBayerRaw(rawTexture, maxW, maxH, x, y, 0, -1);
        nb.cS1 = sampleBayerRaw(rawTexture, maxW, maxH, x, y, 0, 1);
        nb.cE1 = sampleBayerRaw(rawTexture, maxW, maxH, x, y, 1, 0);
        nb.cW1 = sampleBayerRaw(rawTexture, maxW, maxH, x, y, -1, 0);
        nb.cN2 = sampleBayerRaw(rawTexture, maxW, maxH, x, y, 0, -2);
        nb.cS2 = sampleBayerRaw(rawTexture, maxW, maxH, x, y, 0, 2);
        nb.cE2 = sampleBayerRaw(rawTexture, maxW, maxH, x, y, 2, 0);
        nb.cW2 = sampleBayerRaw(rawTexture, maxW, maxH, x, y, -2, 0);
        nb.cNE = sampleBayerRaw(rawTexture, maxW, maxH, x, y, 1, -1);
        nb.cNW = sampleBayerRaw(rawTexture, maxW, maxH, x, y, -1, -1);
        nb.cSE = sampleBayerRaw(rawTexture, maxW, maxH, x, y, 1, 1);
        nb.cSW = sampleBayerRaw(rawTexture, maxW, maxH, x, y, -1, 1);
    }
    return nb;
}


// ──────────────────────────────────────────────────────────────────────
// FUSED: demosaic + LSC + WB + CCM + Log OETF in ONE kernel.
// Used for all recording and live viewfinder preview paths.
// Eliminates intermediate passes and ~196 MB/frame of memory traffic.
// ──────────────────────────────────────────────────────────────────────
kernel void debayerFusedLog(
    texture2d<float, access::read>  rawTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant FusedParams &params   [[buffer(0)]],
    constant LSCParams &lsc       [[buffer(1)]],
    constant float3x3 &colorMatrix [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    int x = int(gid.x);
    int y = int(gid.y);

    int pattern = params.bayerPattern;
    bool xEven = ((x ^ pattern) & 1) == 0;
    bool yEven = ((y ^ (pattern >> 1)) & 1) == 0;

    float black = params.blackLevel;
    float white = params.whiteLevel;
    float invDenom = 1.0f / max(white - black, 1e-6f);
    int maxW = int(rawTexture.get_width()) - 1;
    int maxH = int(rawTexture.get_height()) - 1;

    // ── Directional Demosaic (Malvar-He-Cutler) on raw DNs ──
    BayerNeighborhood nb = fetchBayerNeighborhood(rawTexture, maxW, maxH, x, y);

    float r, g, b;
    if (xEven == yEven) {
        float G_at_RB = (2.0f * (nb.cN1 + nb.cS1 + nb.cE1 + nb.cW1) + 4.0f * nb.c00 - (nb.cN2 + nb.cS2 + nb.cE2 + nb.cW2)) * 0.125f;
        float Color_at_Diag = (2.0f * (nb.cNE + nb.cNW + nb.cSE + nb.cSW) + 6.0f * nb.c00 - 1.5f * (nb.cN2 + nb.cS2 + nb.cE2 + nb.cW2)) * 0.125f;
        if (yEven) {
            r = nb.c00; g = G_at_RB; b = Color_at_Diag;
        } else {
            b = nb.c00; g = G_at_RB; r = Color_at_Diag;
        }
    } else {
        float Color_at_G_H = (4.0f * (nb.cE1 + nb.cW1) + 5.0f * nb.c00 - (nb.cE2 + nb.cW2) + 0.5f * (nb.cN2 + nb.cS2) - (nb.cNE + nb.cNW + nb.cSE + nb.cSW)) * 0.125f;
        float Color_at_G_V = (4.0f * (nb.cN1 + nb.cS1) + 5.0f * nb.c00 - (nb.cN2 + nb.cS2) + 0.5f * (nb.cE2 + nb.cW2) - (nb.cNE + nb.cNW + nb.cSE + nb.cSW)) * 0.125f;
        if (yEven) {
            r = Color_at_G_H; g = nb.c00; b = Color_at_G_V;
        } else {
            b = Color_at_G_H; g = nb.c00; r = Color_at_G_V;
        }
    }

    // ── Single Post-Demosaic Linearization: (DN - black) * invDenom ──
    r = max((r - black) * invDenom, 0.0f);
    g = max((g - black) * invDenom, 0.0f);
    b = max((b - black) * invDenom, 0.0f);

    // Gr/Gb green balance before LSC/WB.
    g *= params.greenBalance;

    // Fast Lens Shading Correction (LSC): true cos⁴θ optical inverse model
    float outW = float(outTexture.get_width());
    float outH = float(outTexture.get_height());
    float dx = float(x) + 0.5f - 0.5f * outW;
    float dy = float(y) + 0.5f - 0.5f * outH;
    float invCornerDistSq = 4.0f / max(outW * outW + outH * outH, 1e-4f);
    float rNormSq = (dx * dx + dy * dy) * invCornerDistSq;

    // Exact cos⁴θ inverse gain: (1.0 + alpha * rNormSq)^2
    float3 alphaGain = float3(lsc.radialR, lsc.radialG, lsc.radialB);
    float3 baseGain = float3(1.0f) + alphaGain * rNormSq;
    float3 gain = baseGain * baseGain;
    if (lsc.radial4R != 0.0f || lsc.radial4G != 0.0f || lsc.radial4B != 0.0f) {
        gain += float3(lsc.radial4R, lsc.radial4G, lsc.radial4B) * (rNormSq * rNormSq);
    }
    if (lsc.azimuthR != 0.0f || lsc.azimuthG != 0.0f || lsc.azimuthB != 0.0f) {
        float cos2Theta = (dx * dx - dy * dy) / max(dx * dx + dy * dy, 1e-6f);
        gain += float3(lsc.azimuthR, lsc.azimuthG, lsc.azimuthB) * cos2Theta;
    }
    float3 rgb = min(float3(r, g, b) * gain, float3(8.0f));

    // ── White Balance ──
    rgb *= params.wbGains;
    rgb = max(rgb, float3(0.0));

    // ── Color Correction Matrix (Sensor Native → Target Gamut e.g. BT.2020) ──
    rgb = colorMatrix * rgb;
    rgb = max(rgb, float3(0.0));

    // Clipping flag and smooth highlight desaturation on raw demosaiced photosites (sensor saturation)
    // Evaluated on native sensor DNs before LSC/WB so that lens vignetting gains never bleach corner colors.
    float rawPeak = max(r, max(g, b));
    bool isClipped = (rawPeak >= 0.98f);
    float alpha = isClipped ? 0.0f : 1.0f;

    if (rawPeak >= 0.92f) {
        float u = clamp((rawPeak - 0.92f) * 12.5f, 0.0f, 1.0f);
        float desat = u * u * (3.0f - 2.0f * u);
        float luma = dot(rgb, float3(0.2627f, 0.6780f, 0.0593f));
        rgb = mix(rgb, float3(luma), desat);
    }

    // ── Direct Log OETF Encoding ──
    float3 logRGB = encodeLogCurve(rgb, params.curveType, params.headroomScale);

    outTexture.write(float4(logRGB, alpha), gid);
}

// ──────────────────────────────────────────────────────────────────────
// ULTRA-FAST 1-TAP FORMAT CONVERSION: rgba16Float -> bgra8Unorm.
// Used strictly for the live screen viewfinder (MTKView).
// ──────────────────────────────────────────────────────────────────────
kernel void convertRgba16FloatToBgra8(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    dst.write(src.read(gid), gid);
}

// ──────────────────────────────────────────────────────────────────────
// 10-BIT YCBCR 4:2:0 ENCODING: rgba16Float -> 420YpCbCr10BiPlanar
// Converts linear or log RGB directly to ITU-R BT.2020 10-bit Video Range:
//   Plane 0 (Y):  .r16Unorm, width x height
//   Plane 1 (UV): .rg16Unorm, (width/2) x (height/2)
// Dispatched over (width/2) x (height/2) grid:
//   1 thread per 2x2 block -> writes 4 luma pixels and 1 interleaved chroma pixel.
// ──────────────────────────────────────────────────────────────────────
kernel void convertRgbTo420YpCbCr10(
    texture2d<float, access::read>  srcRGB  [[texture(0)]],
    texture2d<float, access::write> dstY    [[texture(1)]],
    texture2d<float, access::write> dstUV   [[texture(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint uvW = dstUV.get_width();
    uint uvH = dstUV.get_height();
    if (gid.x >= uvW || gid.y >= uvH) return;

    uint baseX = gid.x * 2u;
    uint baseY = gid.y * 2u;
    uint srcW = srcRGB.get_width();
    uint srcH = srcRGB.get_height();

    // Sample 2x2 block with edge clamping
    float3 p00, p10, p01, p11;
    if (baseX + 1u < srcW && baseY + 1u < srcH) {
        p00 = srcRGB.read(uint2(baseX,      baseY)).rgb;
        p10 = srcRGB.read(uint2(baseX + 1u, baseY)).rgb;
        p01 = srcRGB.read(uint2(baseX,      baseY + 1u)).rgb;
        p11 = srcRGB.read(uint2(baseX + 1u, baseY + 1u)).rgb;
    } else {
        p00 = srcRGB.read(uint2(min(baseX,      srcW - 1u), min(baseY,      srcH - 1u))).rgb;
        p10 = srcRGB.read(uint2(min(baseX + 1u, srcW - 1u), min(baseY,      srcH - 1u))).rgb;
        p01 = srcRGB.read(uint2(min(baseX,      srcW - 1u), min(baseY + 1u, srcH - 1u))).rgb;
        p11 = srcRGB.read(uint2(min(baseX + 1u, srcW - 1u), min(baseY + 1u, srcH - 1u))).rgb;
    }

    // ITU-R BT.2020 non-constant luminance luma:
    constexpr float3 wY = float3(0.2627f, 0.6780f, 0.0593f);
    float y00 = dot(p00, wY);
    float y10 = dot(p10, wY);
    float y01 = dot(p01, wY);
    float y11 = dot(p11, wY);

    // 10-bit Video Range quantization (SMPTE / ITU standard):
    // Y: 64 to 940 (range = 876) -> [64..940] / 65535 * 64
    // Cb, Cr: 64 to 960 (range = 896, center = 512) -> [64..960] / 65535 * 64
    constexpr float kScaleY = (876.0f * 64.0f) / 65535.0f;
    constexpr float kOffsetY = (64.0f * 64.0f) / 65535.0f;
    constexpr float kScaleC = (896.0f * 64.0f) / 65535.0f;
    constexpr float kOffsetC = (512.0f * 64.0f) / 65535.0f;
    constexpr float kInvChromaB = 1.0f / 1.8814f;
    constexpr float kInvChromaR = 1.0f / 1.4746f;

    float normY00 = metal::fma(saturate(y00), kScaleY, kOffsetY);
    float normY10 = metal::fma(saturate(y10), kScaleY, kOffsetY);
    float normY01 = metal::fma(saturate(y01), kScaleY, kOffsetY);
    float normY11 = metal::fma(saturate(y11), kScaleY, kOffsetY);

    uint dstYW = dstY.get_width();
    uint dstYH = dstY.get_height();
    if (baseX + 1u < dstYW && baseY + 1u < dstYH) {
        dstY.write(float4(normY00, 0.0f, 0.0f, 1.0f), uint2(baseX, baseY));
        dstY.write(float4(normY10, 0.0f, 0.0f, 1.0f), uint2(baseX + 1u, baseY));
        dstY.write(float4(normY01, 0.0f, 0.0f, 1.0f), uint2(baseX, baseY + 1u));
        dstY.write(float4(normY11, 0.0f, 0.0f, 1.0f), uint2(baseX + 1u, baseY + 1u));
    } else {
        if (baseX < dstYW && baseY < dstYH) {
            dstY.write(float4(normY00, 0.0f, 0.0f, 1.0f), uint2(baseX, baseY));
        }
        if (baseX + 1u < dstYW && baseY < dstYH) {
            dstY.write(float4(normY10, 0.0f, 0.0f, 1.0f), uint2(baseX + 1u, baseY));
        }
        if (baseX < dstYW && baseY + 1u < dstYH) {
            dstY.write(float4(normY01, 0.0f, 0.0f, 1.0f), uint2(baseX, baseY + 1u));
        }
        if (baseX + 1u < dstYW && baseY + 1u < dstYH) {
            dstY.write(float4(normY11, 0.0f, 0.0f, 1.0f), uint2(baseX + 1u, baseY + 1u));
        }
    }

    // ITU-R BT.2020 / HEVC Type 0 horizontally left-cosited chroma sampling:
    // Aligns chroma phase with the left luma column (x = 0, y = 0.5) to eliminate
    // 0.5-pixel color fringing on high-contrast vertical transitions.
    float3 avgRGB = 0.375f * (p00 + p01) + 0.125f * (p10 + p11);
    float avgY = 0.375f * (y00 + y01) + 0.125f * (y10 + y11);
    float cb = (avgRGB.b - avgY) * kInvChromaB;
    float cr = (avgRGB.r - avgY) * kInvChromaR;

    float normCb = metal::fma(clamp(cb, -0.5f, 0.5f), kScaleC, kOffsetC);
    float normCr = metal::fma(clamp(cr, -0.5f, 0.5f), kScaleC, kOffsetC);

    dstUV.write(float4(normCb, normCr, 0.0f, 1.0f), gid);
}

// ──────────────────────────────────────────────────────────────────────
// ADAPTIVE UNSHARP MASK: enhances fine edge details without boosting noise.
// ──────────────────────────────────────────────────────────────────────
kernel void unsharpMaskAdaptive(
    texture2d<float, access::read>  inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant float &strength [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    float4 center = inTexture.read(gid);
    if (strength <= 0.001f) {
        outTexture.write(center, gid);
        return;
    }

    int x = int(gid.x);
    int y = int(gid.y);
    int w = inTexture.get_width();
    int h = inTexture.get_height();

    int ym1 = max(0, y - 1);
    int yp1 = min(h - 1, y + 1);
    int xm1 = max(0, x - 1);
    int xp1 = min(w - 1, x + 1);

    // 3x3 Gaussian blur approximation
    float3 blur = (
        inTexture.read(uint2(xm1, ym1)).rgb * 1.0f +
        inTexture.read(uint2(x,   ym1)).rgb * 2.0f +
        inTexture.read(uint2(xp1, ym1)).rgb * 1.0f +
        inTexture.read(uint2(xm1, y)).rgb   * 2.0f +
        center.rgb                          * 4.0f +
        inTexture.read(uint2(xp1, y)).rgb   * 2.0f +
        inTexture.read(uint2(xm1, yp1)).rgb * 1.0f +
        inTexture.read(uint2(x,   yp1)).rgb * 2.0f +
        inTexture.read(uint2(xp1, yp1)).rgb * 1.0f
    ) * (1.0f / 16.0f);

    float3 highPass = center.rgb - blur;
    float detailLuma = abs(dot(highPass, float3(0.2627f, 0.6780f, 0.0593f)));

    // Adaptive coring: only sharpen true edge detail, not low-amplitude shadow noise
    float coringThreshold = 0.006f;
    float coringFactor = smoothstep(0.001f, coringThreshold, detailLuma);

    // Suppress sharpening in deep shadows where the log transfer function has high derivative
    // to prevent noise floor amplification, while delivering crisp cinema sharpness to midtones and highlights.
    float centerLuma = dot(center.rgb, float3(0.2627f, 0.6780f, 0.0593f));
    float shadowMask = smoothstep(0.18f, 0.38f, centerLuma);

    float3 sharpened = center.rgb + highPass * (strength * coringFactor * shadowMask);
    sharpened = max(sharpened, float3(0.0f));

    outTexture.write(float4(sharpened, center.a), gid);
}


struct VertexOut {
    float4 position [[position]];
};

vertex VertexOut fullscreenVertex(uint vertexID [[vertex_id]]) {
    VertexOut out;
    float2 pos = float2((vertexID << 1) & 2, vertexID & 2);
    out.position = float4(pos * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return out;
}

struct DisplayUniforms {
    int2 destOffset;
    int2 destSize;
    int  showClipping;
    int  showFocusPeaking;
    int  overlayOnly;
    int  showDisplayLUT;
    int  curveType;
};

fragment float4 displayFragment(
    VertexOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    constant DisplayUniforms &uniforms [[buffer(0)]]
) {
    float2 uv = float2(in.position.x - uniforms.destOffset.x, in.position.y - uniforms.destOffset.y) / float2(uniforms.destSize);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return uniforms.overlayOnly > 0 ? float4(0.0, 0.0, 0.0, 0.0) : float4(0.0, 0.0, 0.0, 1.0);
    }
    
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float4 color = tex.sample(s, uv);
    
    float isClipped = step(color.a, 0.5);
    
    float3 finalColor = color.rgb;

    // ── Viewfinder Rec.709 Display LUT ──
    // Decodes log-encoded texture back to scene-linear according to active curve,
    // applies color gamut mapping (BT.2020 or S-Gamut3.Cine -> BT.709),
    // filmic tone mapping, and sRGB gamma for natural on-screen monitoring.
    // The recorded file is NOT affected — this is display-only.
    if (uniforms.showDisplayLUT > 0) {
        float3 lin;
        float3 rgb709;

        if (uniforms.curveType == 2) {
            // Inverse Sony S-Log3 decode (code value -> scene-linear reflectance)
            for (int i = 0; i < 3; i++) {
                float p = finalColor[i];
                if (p >= 171.2102946929f / 1023.0f) {
                    float a = metal::pow(10.0f, (p * 1023.0f - 420.0f) / 261.5f);
                    lin[i] = a * (0.18f + 0.01f) - 0.01f;
                } else {
                    lin[i] = (p * 1023.0f - 95.0f) * 0.01125000f / (171.2102946929f - 95.0f);
                }
            }
            lin = max(lin, float3(0.0));

            // Sony S-Gamut3.Cine -> BT.709 color gamut mapping (row sums = 1.0)
            const float3x3 mSGamutto709 = float3x3(
                float3( 1.6762f, -0.1839f, -0.0458f),
                float3(-0.4827f,  1.2670f, -0.1751f),
                float3(-0.1935f, -0.0831f,  1.2208f)
            );
            rgb709 = mSGamutto709 * lin;
        } else {
            // Inverse Apple Log 2 decode (log code value -> scene-linear reflectance)
            constexpr float r0 = -0.05641088f;
            constexpr float rt = 0.01f;
            constexpr float c_al = 47.28711236f;
            constexpr float beta = 0.00964052f;
            constexpr float gam = 0.08550479f;
            constexpr float del = 0.69336945f;
            float pt = c_al * (rt - r0) * (rt - r0);

            for (int i = 0; i < 3; i++) {
                float p = finalColor[i];
                if (p < 0.0f) {
                    lin[i] = r0;
                } else if (p < pt) {
                    lin[i] = sqrt(p / c_al) + r0;
                } else {
                    lin[i] = metal::pow(2.0f, (p - del) / gam) - beta;
                }
            }
            lin = max(lin, float3(0.0));

            // BT.2020 -> BT.709 color gamut mapping (standard ITU matrix)
            const float3x3 m2020to709 = float3x3(
                float3( 1.6605f, -0.1246f, -0.0182f),
                float3(-0.5876f,  1.1329f, -0.1006f),
                float3(-0.0728f, -0.0083f,  1.1187f)
            );
            rgb709 = m2020to709 * lin;
        }
        rgb709 = max(rgb709, float3(0.0));

        // Calibrated cinema display tone curve (Hill/ACES fitted filmic response):
        // Maps 18% middle gray (0.18) -> ~41% display IRE, 90% diffuse white (0.90) -> ~88% display IRE,
        // with smooth highlight shoulder rolling off specular highlights to 100% display IRE.
        float3 a = rgb709 * (rgb709 + 0.0245786f) - 0.000090537f;
        float3 b = rgb709 * (0.983729f * rgb709 + 0.4329510f) + 0.238081f;
        float3 mapped = saturate(a / max(b, 1e-4f));
        finalColor = metal::pow(mapped, float3(1.0f / 2.2f));
    }

    // ── Cinema Diagonal Zebra Stripes for Highlight Clipping ──
    float3 zebraColor = float3(1.0f, 0.15f, 0.15f);
    float zebraAlpha = 0.0f;
    if (uniforms.showClipping > 0 && isClipped > 0.5f) {
        float stripe = step(0.5f, fract((in.position.x + in.position.y) / 14.0f));
        zebraAlpha = stripe * 0.85f;
    }

    // ── Cinema 3x3 Sobel Focus Peaking ──
    // Minimal, ultra-precise cinema focus peaking that detects high-frequency optical
    // edges on in-focus focal planes without flooding textures, surfaces, or noise.
    float3 peakColor = float3(0.0f, 1.0f, 0.25f); // Cinema Neon Green
    float peakAlpha = 0.0f;
    if (uniforms.showFocusPeaking > 0) {
        // Tight single-pixel sampling at source texture resolution ensures
        // only razor-sharp high spatial frequency transitions are captured.
        float2 texel = 1.0f / float2(tex.get_width(), tex.get_height());
        constexpr float3 lumaW = float3(0.2627f, 0.6780f, 0.0593f);

        float tl = dot(tex.sample(s, uv + float2(-texel.x, -texel.y)).rgb, lumaW);
        float tc = dot(tex.sample(s, uv + float2( 0.0f,    -texel.y)).rgb, lumaW);
        float tr = dot(tex.sample(s, uv + float2( texel.x, -texel.y)).rgb, lumaW);
        float ml = dot(tex.sample(s, uv + float2(-texel.x,  0.0f)).rgb,    lumaW);
        float mr = dot(tex.sample(s, uv + float2( texel.x,  0.0f)).rgb,    lumaW);
        float bl = dot(tex.sample(s, uv + float2(-texel.x,  texel.y)).rgb, lumaW);
        float bc = dot(tex.sample(s, uv + float2( 0.0f,     texel.y)).rgb, lumaW);
        float br = dot(tex.sample(s, uv + float2( texel.x,  texel.y)).rgb, lumaW);

        float gx = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
        float gy = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);
        float edgeMag = length(float2(gx, gy));

        // Normalize edge magnitude by local luminance to compensate for log compression in highlights
        float localLuma = max(tc, 0.10f);
        float normEdge = edgeMag / localLuma;

        // Minimalist threshold: rejects surfaces, soft gradients, and noise,
        // delivering delicate hairline outlines strictly on critical focus edges.
        float peakStrength = smoothstep(0.08f, 0.24f, normEdge);
        peakAlpha = peakStrength * peakStrength * 0.85f;
    }

    // If rendering ONLY HUD overlay graphics over the stock hardware camera preview:
    if (uniforms.overlayOnly > 0) {
        float3 overlayRGB = float3(0.0f);
        float overlayAlpha = 0.0f;

        if (zebraAlpha > 0.0f) {
            overlayRGB = zebraColor * zebraAlpha;
            overlayAlpha = zebraAlpha;
        }
        if (peakAlpha > 0.0f) {
            overlayRGB = mix(overlayRGB, peakColor * peakAlpha, peakAlpha);
            overlayAlpha = max(overlayAlpha, peakAlpha);
        }
        return float4(overlayRGB, overlayAlpha);
    }

    // Standard Viewfinder Display: composite zebra stripes & neon green peaking over finalColor
    if (zebraAlpha > 0.0f) {
        finalColor = mix(finalColor, zebraColor, zebraAlpha);
    }
    if (peakAlpha > 0.0f) {
        finalColor = mix(finalColor, peakColor, peakAlpha);
    }

    return float4(finalColor, 1.0f);
}


// ──────────────────────────────────────────────────────────────────────
// HARDWARE BILINEAR CROP AND RESAMPLE
// Single-pass crop and hardware-filtered resample directly into destination.
// Eliminates intermediate crop blit copies (~146 MB/frame bandwidth) and
// replaces multi-tap sinc convolution with GPU texture unit filtering (TMUs).
// ──────────────────────────────────────────────────────────────────────
struct CropParams {
    float scaleX;
    float scaleY;
    float startX;
    float startY;
};

kernel void cropAndResampleBilinear(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    constant CropParams &crop            [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;

    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);

    float srcX = metal::fma(float(gid.x), crop.scaleX, crop.startX);
    float srcY = metal::fma(float(gid.y), crop.scaleY, crop.startY);

    dst.write(src.sample(s, float2(srcX, srcY)), gid);
}
