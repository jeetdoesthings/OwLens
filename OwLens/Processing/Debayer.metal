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
    float denoiseStrength;
};



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
    float signalNorm = saturate((center - params.blackLevel) / max(params.whiteLevel - params.blackLevel, 1e-6f));
    // Fade the correction out in highlights so a specular cannot toggle
    // between its actual value and the neighborhood median on adjacent frames.
    if (noise.denoiseStrength <= 0.001f || signalNorm >= 0.80f) {
        dst.write(float4(center, 0.0f, 0.0f, 1.0f), gid);
        return;
    }

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

    float sigmaRaw = sqrt(noise.shotCoeff * signalNorm + noise.readCoeff) * params.whiteLevel;
    float threshold = (5.0f / max(noise.denoiseStrength, 0.2f)) * max(sigmaRaw, 1e-3f);

    float correction = smoothstep(threshold, 1.5f * threshold, abs(center - median));
    correction *= 1.0f - smoothstep(0.45f, 0.80f, signalNorm);
    float out = mix(center, median, correction);
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


// Official manufacturer log transfer functions.
//
// IMPORTANT: the input is scene-linear reflectance, not a gain-adjusted sensor value.
// 18% middle gray must arrive as exactly 0.18 here.  Apple Log 2 then maps it to
// 0.488272 (48.8% IRE) and S-Log3 maps it to 420/1023 (41.1% IRE), exactly as the
// published white papers specify.  Do not apply a sensor-to-scene pre-gain and do
// not add a rational highlight shoulder: Apple Log 2 already carries 12.0x scene
// reflectance (12.0 -> 1.0), and S-Log3 already carries its own published range.
static inline float3 encodeLogCurve(float3 rgb, int curveType) {
    if (curveType == 0) {
        return saturate(rgb);
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
    float headroomScale;  // Retained for layout compatibility; curves are now anchored to official OETFs.
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
    float radialRolloff;
    float outputScale;
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
// FUSED: demosaic + LSC + WB + CCM in ONE kernel, outputting scene-linear RGB.
// Used for all recording and live viewfinder preview paths.
// Eliminates intermediate passes and ~196 MB/frame of memory traffic.
// Log encoding happens later, after crop/scale and scene-linear sharpening.
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

    // ── Pre-WB Bayer Neighborhood Fetch ──
    BayerNeighborhood nb = fetchBayerNeighborhood(rawTexture, maxW, maxH, x, y);

    // Measure true physical sensor saturation before WB and LSC
    float rawMaxDN = max(nb.c00, max(max(nb.cN1, nb.cS1), max(nb.cE1, nb.cW1)));
    float rawPeak = saturate((rawMaxDN - black) * invDenom);

    // ── Pre-WB Malvar-He-Cutler Channel Scaling ──
    // Determine CFA channel gains for the 4 neighborhood parity groups
    float3 channelScale = float3(params.wbGains.r, params.greenBalance * params.wbGains.g, params.wbGains.b) * invDenom;
    float cGain, hGain, vGain, dGain;
    if (xEven == yEven) {
        cGain = yEven ? channelScale.r : channelScale.b;
        hGain = channelScale.g;
        vGain = channelScale.g;
        dGain = yEven ? channelScale.b : channelScale.r;
    } else {
        cGain = channelScale.g;
        hGain = yEven ? channelScale.r : channelScale.b;
        vGain = yEven ? channelScale.b : channelScale.r;
        dGain = channelScale.g;
    }

    nb.c00 = max((nb.c00 - black) * cGain, 0.0f);
    nb.cN2 = max((nb.cN2 - black) * cGain, 0.0f);
    nb.cS2 = max((nb.cS2 - black) * cGain, 0.0f);
    nb.cE2 = max((nb.cE2 - black) * cGain, 0.0f);
    nb.cW2 = max((nb.cW2 - black) * cGain, 0.0f);

    nb.cE1 = max((nb.cE1 - black) * hGain, 0.0f);
    nb.cW1 = max((nb.cW1 - black) * hGain, 0.0f);

    nb.cN1 = max((nb.cN1 - black) * vGain, 0.0f);
    nb.cS1 = max((nb.cS1 - black) * vGain, 0.0f);

    nb.cNE = max((nb.cNE - black) * dGain, 0.0f);
    nb.cNW = max((nb.cNW - black) * dGain, 0.0f);
    nb.cSE = max((nb.cSE - black) * dGain, 0.0f);
    nb.cSW = max((nb.cSW - black) * dGain, 0.0f);

    // ── Directional Demosaic (Malvar-He-Cutler) on WB-equalized gradients ──
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

    r = max(r, 0.0f);
    g = max(g, 0.0f);
    b = max(b, 0.0f);

    // Lens shading correction in uncropped sensor coordinates.
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
    if (lsc.radialRolloff > 0.0f) {
        gain = 1.0f + (gain - 1.0f) / (1.0f + lsc.radialRolloff * rNormSq);
    }
    if (lsc.azimuthR != 0.0f || lsc.azimuthG != 0.0f || lsc.azimuthB != 0.0f) {
        float cos2Theta = (dx * dx - dy * dy) / max(dx * dx + dy * dy, 1e-6f);
        gain += float3(lsc.azimuthR, lsc.azimuthG, lsc.azimuthB) * cos2Theta;
    }
    // Cap at 16x reflectance so Apple Log 2's full 12x highlight range survives the CCM;
    // the old 8x clamp silently clipped the top ~0.6 stop of speculars.
    float3 rgb = min(float3(r, g, b) * gain, float3(16.0f));

    // ── Color Correction Matrix (Sensor Native → Target Gamut e.g. BT.2020) ──
    rgb = colorMatrix * rgb;
    rgb = max(rgb, float3(0.0));

    // ── Highlight White-Point Neutralization (Anti-Magenta / Anti-Cyan) ──
    // When physical sensor photosites approach saturation, Green clips
    // before Red or Blue, while digital White Balance multipliers (up to 4.5x on Blue, 2.5x on Red)
    // boost the unclipped photosites. Post-CCM, this imbalance produces severe magenta or cyan
    // tinting in specular highlights.
    // We smoothly roll off chrominance as photosites saturate, converging blown-out light
    // sources, reflections, lamps, and sun to pure neutral white (R = G = B).
    if (rawPeak > 0.78f) {
        float desat = smoothstep(0.78f, 0.998f, rawPeak);
        // Use luminance in the output gamut. Retaining luminance prevents
        // a changing largest RGB channel from pumping highlight brightness.
        float3 weights = params.curveType == 3
            ? float3(0.270410f, 0.816097f, -0.086507f)
            : (params.curveType == 2 ? float3(0.2627f, 0.6780f, 0.0593f) : float3(0.2126f, 0.7152f, 0.0722f));
        float lum = max(0.0f, dot(rgb, weights));
        rgb = mix(rgb, float3(lum), desat);
    }

    // Fade the display-only zebra mask across the clipping threshold.
    float alpha = 1.0f - smoothstep(0.975f, 0.995f, rawPeak);

    // ── Output scene-linear RGB ──
    // Log encoding is performed after crop/scale in the dedicated encodeLogCurve kernel.
    // Keeping this pass linear means the OETF always receives true scene reflectance
    // (18% gray = 0.18), with no artificial tinting or sharpening.
    outTexture.write(float4(rgb * lsc.outputScale, alpha), gid);
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
    texture2d<float, access::read>  srcRGB    [[texture(0)]],
    texture2d<float, access::write> dstY      [[texture(1)]],
    texture2d<float, access::write> dstUV     [[texture(2)]],
    constant int                   &curveType [[buffer(0)]],
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

    // Log shadow chroma noise gating:
    // When log encoding is active, residual sensor read noise around the log black pedestal
    // causes high-frequency temporal chroma fluctuations. Gate only the first few percent above
    // the pedestal so true colours (including deep blues and warm highlights) are never touched.
    float logPedestal = 0.0f;
    if (curveType == 3) {
        logPedestal = 0.1505f; // Apple Log
    } else if (curveType == 2) {
        logPedestal = 0.0929f; // S-Log3
    }
    if (logPedestal > 0.0f && avgY < (logPedestal + 0.006f)) {
        float chromaGate = smoothstep(logPedestal + 0.0005f, logPedestal + 0.006f, avgY);
        cb *= chromaGate;
        cr *= chromaGate;
    }

    // NOTE: the former >78% IRE chroma attenuation has been removed. It was a workaround for
    // HEVC 4:2:0 inter-frame quantization and visibly desaturated warm highlights. ProRes 422 HQ
    // (and the corrected 10-bit HEVC path) preserve the full highlight chroma instead.

    float normCb = metal::fma(clamp(cb, -0.5f, 0.5f), kScaleC, kOffsetC);
    float normCr = metal::fma(clamp(cr, -0.5f, 0.5f), kScaleC, kOffsetC);

    dstUV.write(float4(normCb, normCr, 0.0f, 1.0f), gid);
}

// ──────────────────────────────────────────────────────────────────────
// 10-BIT YCBCR 4:2:2 ENCODING: rgba16Float -> 422YpCbCr10BiPlanar
// Required for Apple ProRes 422 / 422 HQ (10-bit intra-frame, 4:2:2).
//   Plane 0 (Y):  .r16Unorm, width x height
//   Plane 1 (UV): .rg16Unorm, (width/2) x height
// Dispatched over (width/2) x height grid: one thread writes two luma samples
// and one horizontally co-sited chroma pair.  Vertical chroma is full resolution,
// so no 4:2:0 macroblock chroma quantisation can occur.
// ──────────────────────────────────────────────────────────────────────
kernel void convertRgbTo422YpCbCr10(
    texture2d<float, access::read>  srcRGB    [[texture(0)]],
    texture2d<float, access::write> dstY      [[texture(1)]],
    texture2d<float, access::write> dstUV     [[texture(2)]],
    constant int                   &curveType [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint uvW = dstUV.get_width();
    uint uvH = dstUV.get_height();
    if (gid.x >= uvW || gid.y >= uvH) return;

    uint srcW = srcRGB.get_width();
    uint srcH = srcRGB.get_height();
    if (srcW == 0 || srcH == 0) return;

    uint y = min(gid.y, srcH - 1u);
    uint x0 = min(gid.x * 2u, srcW - 1u);
    uint x1 = min(x0 + 1u, srcW - 1u);

    float3 p0 = srcRGB.read(uint2(x0, y)).rgb;
    float3 p1 = srcRGB.read(uint2(x1, y)).rgb;

    // BT.2020 for log masters; Rec.709 for the linear debug path (matching its metadata).
    bool is709 = (curveType == 0);
    float3 wY = is709 ? float3(0.2126f, 0.7152f, 0.0722f) : float3(0.2627f, 0.6780f, 0.0593f);
    float y0 = dot(p0, wY);
    float y1 = dot(p1, wY);

    constexpr float kScaleY = (876.0f * 64.0f) / 65535.0f;
    constexpr float kOffsetY = (64.0f * 64.0f) / 65535.0f;
    constexpr float kScaleC = (896.0f * 64.0f) / 65535.0f;
    constexpr float kOffsetC = (512.0f * 64.0f) / 65535.0f;
    float kInvChromaB = is709 ? (1.0f / 1.8556f) : (1.0f / 1.8814f);
    float kInvChromaR = is709 ? (1.0f / 1.5748f) : (1.0f / 1.4746f);

    uint dstYW = dstY.get_width();
    uint dstYH = dstY.get_height();
    if (x0 < dstYW && y < dstYH) {
        dstY.write(float4(metal::fma(saturate(y0), kScaleY, kOffsetY), 0.0f, 0.0f, 1.0f), uint2(x0, y));
    }
    if (x1 < dstYW && y < dstYH && x1 != x0) {
        dstY.write(float4(metal::fma(saturate(y1), kScaleY, kOffsetY), 0.0f, 0.0f, 1.0f), uint2(x1, y));
    }

    // Horizontally co-sited chroma: average the two luma columns covered by this sample.
    float3 avgRGB = 0.5f * (p0 + p1);
    float avgY = 0.5f * (y0 + y1);
    float cb = (avgRGB.b - avgY) * kInvChromaB;
    float cr = (avgRGB.r - avgY) * kInvChromaR;

    // Minimal shadow gate only; ProRes is intra-frame so highlight chroma needs no attenuation.
    float logPedestal = 0.0f;
    if (curveType == 3) {
        logPedestal = 0.1505f;
    } else if (curveType == 2) {
        logPedestal = 0.0929f;
    }
    if (logPedestal > 0.0f && avgY < (logPedestal + 0.006f)) {
        float chromaGate = smoothstep(logPedestal + 0.0005f, logPedestal + 0.006f, avgY);
        cb *= chromaGate;
        cr *= chromaGate;
    }

    float normCb = metal::fma(clamp(cb, -0.5f, 0.5f), kScaleC, kOffsetC);
    float normCr = metal::fma(clamp(cr, -0.5f, 0.5f), kScaleC, kOffsetC);
    dstUV.write(float4(normCb, normCr, 0.0f, 1.0f), gid);
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
    int  flip180;
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
    if (uniforms.flip180 != 0) {
        uv = 1.0f - uv;
    }
    
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float4 color = tex.sample(s, uv);
    
    float clipStrength = saturate(1.0f - color.a);
    
    float3 finalColor = color.rgb;

    // ── Viewfinder Rec.709 Display Transform ──
    // This is deliberately a color transform, not a contrast-only display LUT:
    // log code -> linear BT.2020 -> linear BT.709 -> Rec.709 OETF. It is exact
    // for values that fit SDR. Values outside the Rec.709 display gamut/range are
    // clipped here; an aesthetic highlight roll-off belongs in a separate monitor
    // LUT and must never be presented as an accurate Rec.709 conversion.
    // The recording texture is never modified.
    if (uniforms.showDisplayLUT > 0) {
        float3 linear;
        if (uniforms.curveType == 2) {
            // Inverse S-Log3 OETF. This app stores that transfer curve in a
            // BT.2020 container, not Sony S-Gamut3.Cine.
            constexpr float log2_10 = 3.32192809489f;
            for (int i = 0; i < 3; ++i) {
                float p = finalColor[i];
                linear[i] = p >= (171.2102946929f / 1023.0f)
                    ? metal::exp2(((p * 1023.0f - 420.0f) / 261.5f) * log2_10) * 0.19f - 0.01f
                    : (p * 1023.0f - 95.0f) * 0.01125f / (171.2102946929f - 95.0f);
            }
        } else if (uniforms.curveType == 3) {
            // Inverse Apple Log 2 OETF.
            constexpr float r0 = -0.05641088f;
            constexpr float c = 47.28711236f;
            constexpr float beta = 0.00964052f;
            constexpr float invGamma = 1.0f / 0.08550479f;
            constexpr float delta = 0.69336945f;
            constexpr float breakpoint = c * (0.01f - r0) * (0.01f - r0);
            for (int i = 0; i < 3; ++i) {
                float p = finalColor[i];
                linear[i] = p < breakpoint
                    ? metal::sqrt(max(p, 0.0f) / c) + r0
                    : metal::exp2((p - delta) * invGamma) - beta;
            }
        } else {
            linear = finalColor;
        }
        linear = max(linear, float3(0.0f));

        // Linear BT.2020 (D65) -> linear BT.709 (D65), column-major.
        const float3x3 bt2020To709 = float3x3(
            float3( 1.660491f, -0.124550f, -0.018151f),
            float3(-0.587641f,  1.132900f, -0.100579f),
            float3(-0.072850f, -0.008349f,  1.118730f)
        );
        // Apple Log 2 uses Apple Wide Gamut, not BT.2020. This is the
        // Apple Wide Gamut (D65) -> BT.709 matrix derived from its published
        // RGB-to-XYZ primaries.
        const float3x3 appleWideGamutTo709 = float3x3(
            float3( 1.707280f, -0.125011f, -0.043624f),
            float3(-0.519020f,  1.314662f, -0.191214f),
            float3(-0.188261f, -0.189652f,  1.234839f)
        );
        float3 rgb709;
        if (uniforms.curveType == 3) {
            rgb709 = appleWideGamutTo709 * linear;
        } else if (uniforms.curveType == 0) {
            rgb709 = linear;
        } else {
            rgb709 = bt2020To709 * linear;
        }
        rgb709 = saturate(rgb709);

        // ITU-R BT.709 OETF (not the similar-but-different 2.2 display gamma).
        for (int i = 0; i < 3; ++i) {
            float v = rgb709[i];
            finalColor[i] = v < 0.018f ? 4.5f * v : 1.099f * metal::pow(v, 0.45f) - 0.099f;
        }
    }

    // ── Cinema Diagonal Zebra Stripes for Highlight Clipping ──
    float3 zebraColor = float3(1.0f, 0.15f, 0.15f);
    float zebraAlpha = 0.0f;
    if (uniforms.showClipping > 0 && clipStrength > 0.0f) {
        float stripe = step(0.5f, fract((in.position.x + in.position.y) / 14.0f));
        zebraAlpha = stripe * 0.85f * clipStrength;
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
// AREA-INTEGRATING ANTI-ALIASED CROP AND RESAMPLE
//
// Clean 4-tap footprint integration using hardware bilinear filtering.
// Eliminates:
//   1. Negative-lobe ringing & moiré patterns (the "dotted overlay")
//   2. Optical axis phase interference (the "colorless shape in centre")
//   3. False color fringing and haloing on high-contrast edges
// ──────────────────────────────────────────────────────────────────────
struct CropParams {
    float scaleX;
    float scaleY;
    float startX;
    float startY;
    int   flip180;
};

static inline float4 sampleCrop(
    texture2d<float, access::sample> src, constant CropParams &crop, uint2 size, uint2 gid)
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);

    float effectiveX = crop.flip180 ? float(size.x - 1 - gid.x) : float(gid.x);
    float effectiveY = crop.flip180 ? float(size.y - 1 - gid.y) : float(gid.y);

    float srcX = metal::fma(effectiveX, crop.scaleX, crop.startX);
    float srcY = metal::fma(effectiveY, crop.scaleY, crop.startY);

    float4 result;
    if (crop.scaleX > 1.05f || crop.scaleY > 1.05f) {
        // Anti-aliased area integration over 4 quadrants with positive weights (+0.25 each):
        // Each bilinear tap averages a 2x2 texel block; 4 symmetrically spaced taps cover
        // the entire footprint smoothly without negative lobes or checkerboard moiré.
        float offX = 0.25f * crop.scaleX;
        float offY = 0.25f * crop.scaleY;
        float4 s00 = src.sample(s, float2(srcX - offX, srcY - offY));
        float4 s10 = src.sample(s, float2(srcX + offX, srcY - offY));
        float4 s01 = src.sample(s, float2(srcX - offX, srcY + offY));
        float4 s11 = src.sample(s, float2(srcX + offX, srcY + offY));
        result = 0.25f * (s00 + s10 + s01 + s11);
    } else {
        // 1:1 sensor-crop or zoom: single phase-correct hardware-filtered sample
        result = src.sample(s, float2(srcX, srcY));
    }

    result.rgb = max(result.rgb, float3(0.0f));
    result.a = saturate(result.a);
    return result;
}

kernel void cropAndResampleCatmullRom(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant CropParams &crop [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint2 size(dst.get_width(), dst.get_height());
    if (any(gid >= size)) return;
    dst.write(sampleCrop(src, crop, size, gid), gid);
}

kernel void cropAndEncodeLog(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant CropParams &crop [[buffer(0)]],
    constant int &curveType [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint2 size(dst.get_width(), dst.get_height());
    if (any(gid >= size)) return;
    // Match the previous RGBA16Float crop intermediate's rounding in registers.
    float4 linear = float4(half4(sampleCrop(src, crop, size, gid)));
    dst.write(float4(encodeLogCurve(linear.rgb, curveType), linear.a), gid);
}



