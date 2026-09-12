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

static inline float linearize(float raw, float black, float white) {
    float denom = max(white - black, 1e-6);
    return (raw - black) / denom; // Do NOT clamp negative noise here, let it average to zero during demosaic!
}

static inline float sampleBayerClamp(texture2d<float, access::read> tex, int x, int y, int dx, int dy, float black, float white) {
    int nx = clamp(x + dx, 0, int(tex.get_width()) - 1);
    int ny = clamp(y + dy, 0, int(tex.get_height()) - 1);
    float v = tex.read(uint2(nx, ny)).r;
    return linearize(v, black, white);
}

static inline float sampleBayerFast(texture2d<float, access::read> tex, int x, int y, int dx, int dy, float black, float invDenom) {
    int nx = clamp(x + dx, 0, int(tex.get_width()) - 1);
    int ny = clamp(y + dy, 0, int(tex.get_height()) - 1);
    float v = tex.read(uint2(nx, ny)).r;
    return (v - black) * invDenom;
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

    // Direct O(1) unrolled sampling of same-color Bayer neighbors (4 diagonals)
    int w = int(src.get_width()) - 1;
    int h = int(src.get_height()) - 1;
    int x = int(gid.x);
    int y = int(gid.y);

    float n0 = src.read(uint2(clamp(x - 1, 0, w), clamp(y - 1, 0, h))).r;
    float n1 = src.read(uint2(clamp(x + 1, 0, w), clamp(y - 1, 0, h))).r;
    float n2 = src.read(uint2(clamp(x - 1, 0, w), clamp(y + 1, 0, h))).r;
    float n3 = src.read(uint2(clamp(x + 1, 0, w), clamp(y + 1, 0, h))).r;

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

// CFA-preserving half-res bin: out(x,y) = in(2x+(x&1), 2y+(y&1))
// Keeps RGGB/GRBG/… phase. DO NOT use out=in(2x,2y) — that is all one color (pink).
kernel void binBayerCFA(
    texture2d<float, access::read> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    int x = int(gid.x);
    int y = int(gid.y);
    int sx = 2 * x + (x & 1);
    int sy = 2 * y + (y & 1);
    sx = min(sx, int(src.get_width()) - 1);
    sy = min(sy, int(src.get_height()) - 1);
    float v = src.read(uint2(sx, sy)).r;
    dst.write(float4(v, 0.0, 0.0, 1.0), gid);
}



static inline float3 applyHighlightShoulder(float3 r, float rKnee, float rMax) {
    if (rMax <= rKnee + 1e-4f) return r;
    float delta = rMax - rKnee;
    float dr = 1.0f - rKnee;
    float s0 = dr / delta;
    float s1 = 2.0f;
    float a = s1 + s0 - 2.0f;
    float b = 3.0f - 2.0f * s0 - s1;
    float c = s0;

    float3 out;
    for (int i = 0; i < 3; i++) {
        float val = r[i];
        if (val <= rKnee) {
            out[i] = val;
        } else {
            float t = saturate((val - rKnee) / max(dr, 1e-4f));
            float g = ((a * t + b) * t + c) * t;
            out[i] = rKnee + delta * g;
        }
    }
    return out;
}

static inline float3 encodeLogCurve(float3 rgb, int curveType, float headroomScale = 1.0f) {
    if (curveType == 0) {
        return saturate(rgb);
    }

    // Apply filmic highlight shoulder when headroom expansion is active (e.g. headroomScale = 10.0–12.0)
    // Preserves 100% linear calibration for midtones & shadows (r <= 0.36, 18% gray at 0.18)
    // while smoothly rolling off highlights up to the container ceiling.
    if (headroomScale > 1.0f) {
        rgb = applyHighlightShoulder(rgb, 0.36f, headroomScale);
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
    // Output code values roughly 0–1 (10-bit /1023).
    float3 result;
    float3 clamped = max(rgb, float3(0.0));
    for (int i = 0; i < 3; i++) {
        float lin = clamped[i];
        if (lin >= 0.01125) {
            result[i] = (420.0 + log10((lin + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0;
        } else {
            result[i] = (lin * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0;
        }
    }
    return saturate(result);
}

struct LogOnlyParams {
    int   curveType;
    float headroomScale;
};

kernel void applyLogOnly(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant LogOnlyParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    float4 pixel = inTexture.read(gid);
    float3 result = encodeLogCurve(float3(pixel.r, pixel.g, pixel.b), params.curveType, params.headroomScale);
    outTexture.write(float4(result, pixel.a), gid);
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


// ──────────────────────────────────────────────────────────────────────
// LINEAR OUTPUT: demosaic + LSC + WB — NO log curve.
// Used by the linear denoise pipeline before luma/chroma split and log encoding.
// ──────────────────────────────────────────────────────────────────────

kernel void debayerWBLinear(
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

    bool xEven = (x % 2 == 0);
    bool yEven = (y % 2 == 0);

    int pattern = params.bayerPattern;
    if (pattern == 1) { xEven = !xEven; }
    else if (pattern == 2) { yEven = !yEven; }
    else if (pattern == 3) { xEven = !xEven; yEven = !yEven; }

    float black = params.blackLevel;
    float white = params.whiteLevel;
    float invDenom = 1.0f / max(white - black, 1e-6f);

    // ── Directional Demosaic (Malvar-He-Cutler) with FMA linearize ──
    float c00 = sampleBayerFast(rawTexture, x, y, 0, 0, black, invDenom);
    float cN1 = sampleBayerFast(rawTexture, x, y, 0, -1, black, invDenom);
    float cS1 = sampleBayerFast(rawTexture, x, y, 0, 1, black, invDenom);
    float cE1 = sampleBayerFast(rawTexture, x, y, 1, 0, black, invDenom);
    float cW1 = sampleBayerFast(rawTexture, x, y, -1, 0, black, invDenom);
    
    float cN2 = sampleBayerFast(rawTexture, x, y, 0, -2, black, invDenom);
    float cS2 = sampleBayerFast(rawTexture, x, y, 0, 2, black, invDenom);
    float cE2 = sampleBayerFast(rawTexture, x, y, 2, 0, black, invDenom);
    float cW2 = sampleBayerFast(rawTexture, x, y, -2, 0, black, invDenom);
    
    float cNE = sampleBayerFast(rawTexture, x, y, 1, -1, black, invDenom);
    float cNW = sampleBayerFast(rawTexture, x, y, -1, -1, black, invDenom);
    float cSE = sampleBayerFast(rawTexture, x, y, 1, 1, black, invDenom);
    float cSW = sampleBayerFast(rawTexture, x, y, -1, 1, black, invDenom);

    float G_at_RB = (2*(cN1 + cS1 + cE1 + cW1) + 4*c00 - (cN2 + cS2 + cE2 + cW2)) * 0.125f;
    float Color_at_G_H = (4*(cE1 + cW1) + 5*c00 - (cE2 + cW2) - 0.5f*(cN2 + cS2) - (cNE + cNW + cSE + cSW)) * 0.125f;
    float Color_at_G_V = (4*(cN1 + cS1) + 5*c00 - (cN2 + cS2) - 0.5f*(cE2 + cW2) - (cNE + cNW + cSE + cSW)) * 0.125f;
    float Color_at_Diag = (2*(cNE + cNW + cSE + cSW) + 6*c00 - 1.5f*(cN2 + cS2 + cE2 + cW2)) * 0.125f;

    float r, g, b;
    if (yEven && xEven) {
        r = c00; g = G_at_RB; b = Color_at_Diag;
    } else if (yEven && !xEven) {
        r = Color_at_G_H; g = c00; b = Color_at_G_V;
    } else if (!yEven && xEven) {
        b = Color_at_G_H; g = c00; r = Color_at_G_V;
    } else {
        b = c00; g = G_at_RB; r = Color_at_Diag;
    }
    r = max(r, 0.0f);
    g = max(g, 0.0f);
    b = max(b, 0.0f);

    // Gr/Gb green balance before LSC/WB.
    g *= params.greenBalance;

    // Fast SIMD Lens Shading Correction (LSC): radial polynomial + algebraic azimuth
    float outW = float(outTexture.get_width());
    float outH = float(outTexture.get_height());
    float2 uv = (float2(float(x) + 0.5f, float(y) + 0.5f) / float2(outW, outH)) - 0.5f;
    float r2 = dot(uv, uv);
    float r4 = r2 * r2;

    float3 gain = float3(1.0f) + float3(lsc.radialR, lsc.radialG, lsc.radialB) * r2 + float3(lsc.radial4R, lsc.radial4G, lsc.radial4B) * r4;
    if (lsc.azimuthR != 0.0f || lsc.azimuthG != 0.0f || lsc.azimuthB != 0.0f) {
        float cos2Theta = (uv.x * uv.x - uv.y * uv.y) / max(r2, 1e-6f);
        gain += float3(lsc.azimuthR, lsc.azimuthG, lsc.azimuthB) * cos2Theta;
    }
    float3 rgb = min(float3(r, g, b) * gain, float3(8.0f));

    // ── White Balance ──
    rgb *= params.wbGains;
    rgb = max(rgb, float3(0.0));

    // ── Highlight Desaturation & Reconstruction ──
    // When raw sensor channels clip (typically green first on Bayer sensors),
    // WB gains multiply red/blue channels to ~2x while green stays pinned at 1.0,
    // causing severe magenta/pink highlights. Smoothly desaturate chroma towards
    // peak highlight luminance as raw levels approach clipping (> 0.88), rolling off
    // into clean, neutral white highlights.
    float maxRaw = max(r, max(g, b));
    if (maxRaw > 0.88f) {
        float desat = smoothstep(0.88f, 0.98f, maxRaw);
        float peakVal = max(rgb.r, max(rgb.g, rgb.b));
        rgb = mix(rgb, float3(peakVal), desat);
    }

    // ── Color Correction Matrix (Sensor Native → Target Gamut e.g. BT.2020) ──
    rgb = colorMatrix * rgb;
    rgb = max(rgb, float3(0.0));

    // Clipping flag on raw demosaiced values (sensor saturation)
    bool isClipped = (r >= 0.98 || g >= 0.98 || b >= 0.98);
    float alpha = isClipped ? 0.0 : 1.0;

    // Output scene-linear RGB (NO log curve)
    outTexture.write(float4(rgb, alpha), gid);
}

// ──────────────────────────────────────────────────────────────────────
// FUSED: demosaic + LSC + WB + CCM + Log OETF in ONE kernel.
// Used for 4K recording, OpenGate fast path, and live viewfinder preview
// when spatial/chroma denoise is bypassed. Eliminates 1 full GPU pass
// and ~196 MB/frame of intermediate memory traffic.
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

    bool xEven = (x % 2 == 0);
    bool yEven = (y % 2 == 0);

    int pattern = params.bayerPattern;
    if (pattern == 1) { xEven = !xEven; }
    else if (pattern == 2) { yEven = !yEven; }
    else if (pattern == 3) { xEven = !xEven; yEven = !yEven; }

    float black = params.blackLevel;
    float white = params.whiteLevel;
    float invDenom = 1.0f / max(white - black, 1e-6f);

    // ── Directional Demosaic (Malvar-He-Cutler) with FMA linearize ──
    float c00 = sampleBayerFast(rawTexture, x, y, 0, 0, black, invDenom);
    float cN1 = sampleBayerFast(rawTexture, x, y, 0, -1, black, invDenom);
    float cS1 = sampleBayerFast(rawTexture, x, y, 0, 1, black, invDenom);
    float cE1 = sampleBayerFast(rawTexture, x, y, 1, 0, black, invDenom);
    float cW1 = sampleBayerFast(rawTexture, x, y, -1, 0, black, invDenom);
    
    float cN2 = sampleBayerFast(rawTexture, x, y, 0, -2, black, invDenom);
    float cS2 = sampleBayerFast(rawTexture, x, y, 0, 2, black, invDenom);
    float cE2 = sampleBayerFast(rawTexture, x, y, 2, 0, black, invDenom);
    float cW2 = sampleBayerFast(rawTexture, x, y, -2, 0, black, invDenom);
    
    float cNE = sampleBayerFast(rawTexture, x, y, 1, -1, black, invDenom);
    float cNW = sampleBayerFast(rawTexture, x, y, -1, -1, black, invDenom);
    float cSE = sampleBayerFast(rawTexture, x, y, 1, 1, black, invDenom);
    float cSW = sampleBayerFast(rawTexture, x, y, -1, 1, black, invDenom);

    float G_at_RB = (2*(cN1 + cS1 + cE1 + cW1) + 4*c00 - (cN2 + cS2 + cE2 + cW2)) * 0.125f;
    float Color_at_G_H = (4*(cE1 + cW1) + 5*c00 - (cE2 + cW2) - 0.5f*(cN2 + cS2) - (cNE + cNW + cSE + cSW)) * 0.125f;
    float Color_at_G_V = (4*(cN1 + cS1) + 5*c00 - (cN2 + cS2) - 0.5f*(cE2 + cW2) - (cNE + cNW + cSE + cSW)) * 0.125f;
    float Color_at_Diag = (2*(cNE + cNW + cSE + cSW) + 6*c00 - 1.5f*(cN2 + cS2 + cE2 + cW2)) * 0.125f;

    float r, g, b;
    if (yEven && xEven) {
        r = c00; g = G_at_RB; b = Color_at_Diag;
    } else if (yEven && !xEven) {
        r = Color_at_G_H; g = c00; b = Color_at_G_V;
    } else if (!yEven && xEven) {
        b = Color_at_G_H; g = c00; r = Color_at_G_V;
    } else {
        b = c00; g = G_at_RB; r = Color_at_Diag;
    }
    r = max(r, 0.0f);
    g = max(g, 0.0f);
    b = max(b, 0.0f);

    // Gr/Gb green balance before LSC/WB.
    g *= params.greenBalance;

    // Fast SIMD Lens Shading Correction (LSC): radial polynomial + algebraic azimuth
    float outW = float(outTexture.get_width());
    float outH = float(outTexture.get_height());
    float2 uv = (float2(float(x) + 0.5f, float(y) + 0.5f) / float2(outW, outH)) - 0.5f;
    float r2 = dot(uv, uv);
    float r4 = r2 * r2;

    float3 gain = float3(1.0f) + float3(lsc.radialR, lsc.radialG, lsc.radialB) * r2 + float3(lsc.radial4R, lsc.radial4G, lsc.radial4B) * r4;
    if (lsc.azimuthR != 0.0f || lsc.azimuthG != 0.0f || lsc.azimuthB != 0.0f) {
        float cos2Theta = (uv.x * uv.x - uv.y * uv.y) / max(r2, 1e-6f);
        gain += float3(lsc.azimuthR, lsc.azimuthG, lsc.azimuthB) * cos2Theta;
    }
    float3 rgb = min(float3(r, g, b) * gain, float3(8.0f));

    // ── White Balance ──
    rgb *= params.wbGains;
    rgb = max(rgb, float3(0.0));

    // ── Highlight Desaturation & Reconstruction ──
    // Smoothly desaturate chroma towards peak highlight luminance as raw levels approach
    // clipping (> 0.88), preventing pink/magenta cast on clipped highlights.
    float maxRaw = max(r, max(g, b));
    if (maxRaw > 0.88f) {
        float desat = smoothstep(0.88f, 0.98f, maxRaw);
        float peakVal = max(rgb.r, max(rgb.g, rgb.b));
        rgb = mix(rgb, float3(peakVal), desat);
    }

    // ── Color Correction Matrix (Sensor Native → Target Gamut e.g. BT.2020) ──
    rgb = colorMatrix * rgb;
    rgb = max(rgb, float3(0.0));

    // Clipping flag on raw demosaiced values (sensor saturation)
    bool isClipped = (r >= 0.98 || g >= 0.98 || b >= 0.98);
    float alpha = isClipped ? 0.0 : 1.0;

    // ── Direct Log OETF Encoding with Filmic Highlight Shoulder ──
    float3 logRGB = encodeLogCurve(rgb, params.curveType, params.headroomScale);

    outTexture.write(float4(logRGB, alpha), gid);
}

// ──────────────────────────────────────────────────────────────────────
// ULTRA-FAST 1-TAP FORMAT CONVERSION: rgba16Float -> bgra8Unorm.
// 1 vectorized load + 1 vectorized store per pixel (<1ms at 4K).
// Used instead of expensive multi-tap Lanczos sinc filtering when
// resolution already matches target framing.
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
    float detailLuma = abs(dot(highPass, float3(0.2126f, 0.7152f, 0.0722f)));

    // Adaptive coring: only sharpen true edge detail, not low-amplitude shadow noise
    float coringThreshold = 0.006f;
    float coringFactor = smoothstep(0.001f, coringThreshold, detailLuma);

    float3 sharpened = center.rgb + highPass * (strength * coringFactor);
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

fragment float4 displayFragment(
    VertexOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    constant int2 &destOffset [[buffer(0)]],
    constant int2 &destSize [[buffer(1)]],
    constant int &showClipping [[buffer(2)]],
    constant int &showFocusPeaking [[buffer(3)]],
    constant int &overlayOnly [[buffer(4)]],
    constant int &showDisplayLUT [[buffer(5)]],
    constant int &curveType [[buffer(6)]]
) {
    float2 uv = float2(in.position.x - destOffset.x, in.position.y - destOffset.y) / float2(destSize);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return overlayOnly > 0 ? float4(0.0, 0.0, 0.0, 0.0) : float4(0.0, 0.0, 0.0, 1.0);
    }
    
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float4 color = tex.sample(s, uv);
    
    float isClipped = step(color.a, 0.5);
    float applyRed = (showClipping > 0) ? isClipped : 0.0;
    
    float3 finalColor = color.rgb;

    // ── Viewfinder Rec.709 Display LUT ──
    // Decodes log-encoded texture back to scene-linear according to active curve,
    // applies color gamut mapping (BT.2020 or S-Gamut3.Cine -> BT.709),
    // filmic tone mapping, and sRGB gamma for natural on-screen monitoring.
    // The recorded file is NOT affected — this is display-only.
    if (showDisplayLUT > 0 && applyRed < 0.5) {
        float3 lin;
        float3 rgb709;

        if (curveType == 2) {
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

            // S-Gamut3.Cine -> BT.709 color gamut mapping (row sums = 1.0)
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

        // Filmic tone mapping (Reinhard with highlight shoulder)
        rgb709 = rgb709 / (rgb709 + 1.0f) * 1.15f;

        // sRGB gamma approximation for display
        finalColor = metal::pow(saturate(rgb709), float3(1.0f / 2.2f));
    }

    // ── Cinema Diagonal Zebra Stripes for Highlight Clipping ──
    // Replaces solid opaque red blob with 45-degree diagonal zebra stripes
    // so camera operators can monitor clipping while seeing scene details underneath.
    float overlayAlpha = 0.0f;
    if (showClipping > 0 && isClipped > 0.5f) {
        float stripe = step(0.5f, fract((in.position.x + in.position.y) / 14.0f));
        finalColor = mix(finalColor, float3(1.0f, 0.15f, 0.15f), stripe * 0.85f);
        overlayAlpha = 0.75f;
    }
    
    // ── Cinema 3x3 Sobel Focus Peaking ──
    // Uses horizontal and vertical luminance gradients to detect sharp optical edges
    // with smooth thresholding, delivering clean neon green outlines on in-focus subjects.
    if (showFocusPeaking > 0) {
        float2 texel = 1.0f / float2(tex.get_width(), tex.get_height());
        constexpr float3 lumaW = float3(0.2126f, 0.7152f, 0.0722f);
        
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

        float peakStrength = smoothstep(0.06f, 0.14f, edgeMag);
        if (peakStrength > 0.0f) {
            float3 peakColor = float3(0.0f, 1.0f, 0.2f); // Cinema Neon Green
            finalColor = mix(finalColor, peakColor, peakStrength * 0.90f);
            overlayAlpha = max(overlayAlpha, peakStrength);
        }
    }
    
    if (overlayOnly > 0) {
        if (overlayAlpha <= 0.0f) {
            return float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        return float4(finalColor, overlayAlpha);
    }
    return float4(finalColor, 1.0f);
}

// ──────────────────────────────────────────────────────────────────────
// PHASE 3: CHROMA BILATERAL DENOISING
// ──────────────────────────────────────────────────────────────────────

static inline float3 rgb2yuv(float3 rgb) {
    float y  = dot(rgb, float3(0.2126, 0.7152, 0.0722));
    float u  = dot(rgb, float3(-0.1146, -0.3854, 0.5)) + 0.5;
    float v  = dot(rgb, float3(0.5, -0.4542, -0.0458)) + 0.5;
    return float3(y, u, v);
}

static inline float3 yuv2rgb(float3 yuv) {
    float y  = yuv.x;
    float u  = yuv.y - 0.5;
    float v  = yuv.z - 0.5;
    float r  = y + 1.5748 * v;
    float g  = y - 0.1873 * u - 0.4681 * v;
    float b  = y + 1.8556 * u;
    return float3(r, g, b);
}

struct BilateralParams {
    float iso;
};


// ──────────────────────────────────────────────────────────────────────
// SPATIAL DENOISING (Linear Space)
// Bilateral filter on luma with ISO-adaptive strength.
// Operates before log curve for better noise statistics.
// ──────────────────────────────────────────────────────────────────────

struct DenoiseParams {
    float iso;
    int   radius;
    float shotCoeff;
    float readCoeff;
    float strength;  // 0.0–1.0 adaptive boost from frame-time budget
};

struct TemporalParams {
    float iso;
    float maxBlend;
    float shotCoeff;
    float readCoeff;
};

struct RingTemporalParams {
    float iso;
    float maxBlend;
    int   slotCount;
    int   validSlots;
    int   chromaW;
    int   chromaH;
    int   cursor;
    float lambda;
    float shotCoeff;
    float readCoeff;
};

kernel void spatialDenoise(
    texture2d<float, access::read>  inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    texture2d<float, access::read>  statsTexture [[texture(2)]],
    constant DenoiseParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;
    
    float4 centerPx = inTexture.read(gid);
    float3 centerRGB = centerPx.rgb;
    float3 centerYUV = rgb2yuv(centerRGB);
    
    float iso = max(params.iso, 33.0);
    int radius = params.radius;
    float maxDist2 = float(radius * radius);
    
    // ISO-adaptive sigma values (tuned for linear-space data)
    // Calibrated path: use measured shot/read coefficients when available.
    // Fallback: sqrt(iso/33) heuristic.
    float isoScale = sqrt(iso / 33.0);
    float sigmaRef;  // noise std-dev at mid-gray reference signal (0.5)
    if (params.shotCoeff > 0.0) {
        sigmaRef = sqrt(params.shotCoeff * 0.5 + params.readCoeff);
    } else {
        sigmaRef = 0.012 * isoScale;  // legacy hardcoded proxy
    }
    float luma01 = saturate(centerYUV.x);

    // Per-pixel local-sigma guide: stronger denoise where local variance is low,
    // lighter denoise on edges/high-variance regions. Falls back to signal-based
    // shadowBoost at 4K where the stats pass is skipped.
    float shadowBoost;
    if (statsTexture.get_width() > 1) {
        float localSigma = max(statsTexture.read(gid).y, 1e-4);
        shadowBoost = clamp(localSigma / sigmaRef, 0.5, 2.0);
    } else {
        shadowBoost = mix(1.45, 0.8, luma01);
    }

    // Luma bilateral radius: sigmaRef * shadowBoost, scaled by adaptive strength.
    // strength=0 → normal sigma; strength=1 → 2× sigma (heavier denoise when
    // frame time is well under budget). Clamped to prevent pathological values.
    float adaptiveScale = 1.0 + min(max(params.strength, 0.0), 1.0);
    float lumaRS = sigmaRef * shadowBoost * adaptiveScale;
    float lumaRS2 = lumaRS * lumaRS;
    
    // Spatial sigma adapts to kernel radius
    float spatialS2 = float(radius) * float(radius) * 0.5;
    
    int w = inTexture.get_width();
    int h = inTexture.get_height();
    int cx = int(gid.x);
    int cy = int(gid.y);
    
    float sumLuma = 0.0;
    float sumLumaW = 0.0;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            float dist2 = float(dx*dx + dy*dy);
            if (dist2 > maxDist2) continue; // Diamond pattern
            
            uint2 pid = uint2(clamp(cx + dx, 0, w - 1), clamp(cy + dy, 0, h - 1));
            float3 sYUV = rgb2yuv(inTexture.read(pid).rgb);
            
            float spatialW = exp(-dist2 / (2.0 * spatialS2));
            float lumaDiff = sYUV.x - centerYUV.x;
            
            // Luma bilateral: edge-stopped by luma difference (tight threshold)
            float lumaW = spatialW * exp(-(lumaDiff * lumaDiff) / (2.0 * lumaRS2));
            sumLumaW += lumaW;
            sumLuma += sYUV.x * lumaW;
            // NOTE: Chroma is NOT smoothed here. The dedicated half-res chroma
            // pipeline (extractHalfResChroma -> denoiseHalfResChroma -> recombine)
            // handles all chroma denoising. Any chroma work in this pass would
            // be discarded by the recombine kernel which reads UV exclusively
            // from chromaDenoisedOut.
        }
    }
    
    float finalY = (sumLumaW > 1e-4) ? (sumLuma / sumLumaW) : centerYUV.x;
    float3 finalRGB = max(yuv2rgb(float3(finalY, centerYUV.y, centerYUV.z)), float3(0.0));
    outTexture.write(float4(finalRGB, centerPx.a), gid);
}

kernel void extractHalfResChroma(
    texture2d<float, access::read>  inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    int w = int(inTexture.get_width());
    int h = int(inTexture.get_height());
    int baseX = int(gid.x) * 2;
    int baseY = int(gid.y) * 2;

    float2 sumUV = float2(0.0);
    float count = 0.0;
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int sx = baseX + dx;
            int sy = baseY + dy;
            if (sx >= w || sy >= h) continue;
            sumUV += rgb2yuv(inTexture.read(uint2(sx, sy)).rgb).yz;
            count += 1.0;
        }
    }

    float2 uv = (count > 0.0) ? (sumUV / count) : float2(0.5);
    outTexture.write(float4(uv.x, uv.y, 0.0, 1.0), gid);
}

kernel void denoiseHalfResChroma(
    texture2d<float, access::read>  chromaTexture [[texture(0)]],
    texture2d<float, access::read>  lumaGuideTexture [[texture(1)]],
    texture2d<float, access::write> outTexture [[texture(2)]],
    texture2d<float, access::read>  statsTexture [[texture(3)]],
    constant DenoiseParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    float iso = max(params.iso, 33.0);
    float isoScale = sqrt(iso / 33.0);
    // Use a slightly wider chroma radius because chroma noise is coarser than luma.
    int radius = params.radius + 1;
    float maxDist2 = float(radius * radius);
    float spatialS2 = float(radius) * float(radius) * 0.5;

    int chromaW = int(chromaTexture.get_width());
    int chromaH = int(chromaTexture.get_height());
    int guideW = int(lumaGuideTexture.get_width());
    int guideH = int(lumaGuideTexture.get_height());
    int cx = int(gid.x);
    int cy = int(gid.y);

    int centerGuideX = clamp(cx * 2 + 1, 0, guideW - 1);
    int centerGuideY = clamp(cy * 2 + 1, 0, guideH - 1);
    float centerY = rgb2yuv(lumaGuideTexture.read(uint2(centerGuideX, centerGuideY)).rgb).x;
    float luma01 = saturate(centerY);
    // Calibrated chroma sigma: use measured coefficients when available.
    float chromaSigmaRef;
    if (params.shotCoeff > 0.0) {
        chromaSigmaRef = sqrt(params.shotCoeff * 0.5 + params.readCoeff);
    } else {
        chromaSigmaRef = 0.045 * isoScale;
    }
    float shadowBoost;
    if (statsTexture.get_width() > 1) {
        uint2 statsCoord = uint2(
            min(uint(centerGuideX), uint(statsTexture.get_width())  - 1),
            min(uint(centerGuideY), uint(statsTexture.get_height()) - 1));
        float localSigma = max(statsTexture.read(statsCoord).y, 1e-4);
        shadowBoost = clamp(localSigma / chromaSigmaRef, 0.5, 2.0);
    } else {
        shadowBoost = mix(1.55, 0.9, luma01);
    }
    float chromaRS = chromaSigmaRef * shadowBoost * (1.0 + min(max(params.strength, 0.0), 1.0));
    float chromaRS2 = chromaRS * chromaRS;

    float2 sumUV = float2(0.0);
    float sumW = 0.0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            float dist2 = float(dx * dx + dy * dy);
            if (dist2 > maxDist2) continue;

            int px = clamp(cx + dx, 0, chromaW - 1);
            int py = clamp(cy + dy, 0, chromaH - 1);
            int guideX = clamp(px * 2 + 1, 0, guideW - 1);
            int guideY = clamp(py * 2 + 1, 0, guideH - 1);
            float sampleY = rgb2yuv(lumaGuideTexture.read(uint2(guideX, guideY)).rgb).x;
            float lumaDiff = sampleY - centerY;
            float spatialW = exp(-dist2 / (2.0 * spatialS2));
            float chromaWgt = spatialW * exp(-(lumaDiff * lumaDiff) / (2.0 * chromaRS2));

            sumUV += chromaTexture.read(uint2(px, py)).rg * chromaWgt;
            sumW += chromaWgt;
        }
    }

    float2 centerUV = chromaTexture.read(gid).rg;
    float2 finalUV = (sumW > 1e-4) ? (sumUV / sumW) : centerUV;
    outTexture.write(float4(finalUV.x, finalUV.y, 0.0, 1.0), gid);
}

kernel void estimateLumaVariance(
    texture2d<float, access::read>  lumaIn   [[texture(0)]],
    texture2d<float, access::write> statsOut [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= statsOut.get_width() || gid.y >= statsOut.get_height()) return;

    float sum = 0.0, sum2 = 0.0, count = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int sx = clamp(int(gid.x) + dx, 0, int(lumaIn.get_width())  - 1);
            int sy = clamp(int(gid.y) + dy, 0, int(lumaIn.get_height()) - 1);
            float y = rgb2yuv(lumaIn.read(uint2(sx, sy)).rgb).x;
            sum  += y;
            sum2 += y * y;
            count += 1.0;
        }
    }
    float mean = sum / count;
    float variance = max(sum2 / count - mean * mean, 0.0);
    float sigma = sqrt(variance);
    statsOut.write(float4(mean, sigma, 0.0, 1.0), gid);
}

static inline float2 readChromaClamped(texture2d<float, access::read> chromaTexture, int x, int y) {
    int w = int(chromaTexture.get_width());
    int h = int(chromaTexture.get_height());
    return chromaTexture.read(uint2(clamp(x, 0, w - 1), clamp(y, 0, h - 1))).rg;
}

kernel void recombineLumaWithHalfResChroma(
    texture2d<float, access::read>  lumaTexture [[texture(0)]],
    texture2d<float, access::read>  chromaTexture [[texture(1)]],
    texture2d<float, access::write> outTexture [[texture(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    float4 lumaPx = lumaTexture.read(gid);
    float y = rgb2yuv(lumaPx.rgb).x;

    float2 chromaCoord = (float2(gid) + 0.5) * 0.5 - 0.5;
    int2 p0 = int2(floor(chromaCoord));
    float2 f = fract(chromaCoord);

    float2 uv00 = readChromaClamped(chromaTexture, p0.x,     p0.y);
    float2 uv10 = readChromaClamped(chromaTexture, p0.x + 1, p0.y);
    float2 uv01 = readChromaClamped(chromaTexture, p0.x,     p0.y + 1);
    float2 uv11 = readChromaClamped(chromaTexture, p0.x + 1, p0.y + 1);

    float2 uv0 = mix(uv00, uv10, f.x);
    float2 uv1 = mix(uv01, uv11, f.x);
    float2 uv = mix(uv0, uv1, f.y);

    float3 rgb = max(yuv2rgb(float3(y, uv.x, uv.y)), float3(0.0));
    outTexture.write(float4(rgb, lumaPx.a), gid);
}


// ──────────────────────────────────────────────────────────────────────
// TEMPORAL DENOISE — RING BUFFER (N-slot weighted average)
// texture(0) = current (full-res RGBA)
// texture(1) = output   (full-res RGBA)
// texture(2) = lumaHistory   (full-res 2D-array RGBA)
// texture(3) = chromaHistory  (half-res 2D-array RG16Float)
// buffer(0) = RingTemporalParams
// Dispatched at full luma resolution; chroma history is read at half-res coordinates.
// ──────────────────────────────────────────────────────────────────────

kernel void temporalDenoiseRing(
    texture2d<float, access::read>  currentTexture [[texture(0)]],
    texture2d<float, access::write> outTexture     [[texture(1)]],
    texture2d_array<float, access::read> lumaHistory   [[texture(2)]],
    texture2d_array<float, access::read> chromaHistory  [[texture(3)]],
    constant RingTemporalParams &params [[buffer(0)]],
    device const float* globalMotionMetric [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    float4 currentPx = currentTexture.read(gid);

    // If global motion is high, skip temporal blending entirely for this frame.
    // Threshold 0.02 catches moderate camera pans and partial-frame motion.
    if (globalMotionMetric != nullptr && *globalMotionMetric > 0.02) {
        outTexture.write(currentPx, gid);
        return;
    }
    float3 currentYUV = rgb2yuv(currentPx.rgb);

    float iso = max(params.iso, 33.0);
    float isoScale = sqrt(iso / 33.0);
    float luma01 = saturate(currentYUV.x);
    // Tighten the shadow/highlight boost so the motion threshold is smaller; small
    // frame-to-frame differences then register as motion and do not get averaged.
    float shadowBoost = mix(1.0, 0.65, luma01);

    float sigmaRef = (params.shotCoeff > 0.0)
        ? sqrt(params.shotCoeff * 0.5 + params.readCoeff)
        : 0.01 * isoScale;
    // Tighten motion thresholds so small frame-to-frame differences register as motion
    // and do not get averaged into ghost trails.
    float lumaThreshold   = max(sigmaRef * shadowBoost, 1e-5);
    float chromaThreshold = max(sigmaRef * 1.5 * shadowBoost, 1e-5);

    float totalWeight = 1.0;
    float3 weightedRGB = currentPx.rgb;

    int lumaW = int(lumaHistory.get_width());
    int lumaH = int(lumaHistory.get_height());

    for (int i = 0; i < params.slotCount; i++) {
        if (i >= params.validSlots) break;

        // Newest slot is (cursor - 1) mod slotCount; oldest is cursor.
        int slot = (int(params.cursor) - 1 - i + params.slotCount) % params.slotCount;

        // Luma history at full resolution
        uint2 lumaCoord = uint2(
            clamp(int(gid.x), 0, lumaW - 1),
            clamp(int(gid.y), 0, lumaH - 1));
        float histY = lumaHistory.read(lumaCoord, slot).r;

        // Chroma history at half resolution — map full-res thread coords to half-res.
        // The chroma ring stores half-resolution UV produced by averaging 2x2 full-res blocks.
        uint2 chromaCoord = uint2(
            min(uint(gid.x) / 2u, uint(params.chromaW - 1)),
            min(uint(gid.y) / 2u, uint(params.chromaH - 1)));
        float2 chromaUV = chromaHistory.read(chromaCoord, slot).rg;
        float3 histYUV = float3(histY, chromaUV.x, chromaUV.y);

        float lumaDiff   = abs(currentYUV.x - histYUV.x);
        float chromaDiff = length(currentYUV.yz - histYUV.yz);

        float lumaMotion   = saturate(lumaDiff / lumaThreshold);
        float chromaMotion = saturate(chromaDiff / chromaThreshold);
        float motion = max(lumaMotion, chromaMotion);

        // i=0 newest gets λ^0 = 1; older slots decay (lambda^i).
        // slotCount == 3 (validSlots <= 3), so i ∈ {0,1,2}: compute powers
        // directly instead of pow() — GPU pow() expands to a costly
        // fexp(flog(x)*y) per thread. This is numerically identical.
        float recency;
        if (i == 1) { recency = params.lambda; }
        else if (i == 2) { recency = params.lambda * params.lambda; }
        else { recency = 1.0f; }
        float slotWeight = params.maxBlend * (1.0 - motion) * recency;
        // Soft motion gate: smooth taper replaces hard binary cutoff to eliminate
        // ghosting from threshold-edge content. At motion=0.15 the gate starts
        // reducing weight, reaching zero by motion=0.35. This replaces the old
        // motion > 0.25 hard gate which created visible ghosting on pixels that
        // straddled the threshold between adjacent frames.
        float motionGate = 1.0 - smoothstep(0.15, 0.35, motion);
        slotWeight *= motionGate;
        slotWeight = clamp(slotWeight, 0.0, 0.95);

        // Reconstruct RGB from luma history Y and chroma history UV
        float3 histRGB = max(yuv2rgb(histYUV), float3(0.0));
        weightedRGB += histRGB * slotWeight;
        totalWeight += slotWeight;
    }

    float3 result = weightedRGB / max(totalWeight, 1e-6);
    outTexture.write(float4(max(result, float3(0.0)), currentPx.a), gid);
}

// ──────────────────────────────────────────────────────────────────────
// GLOBAL MOTION ESTIMATE — coarse frame-to-frame luma difference metric.
// texture(0) = current pre-temporal RGB
// texture(1) = luma history array (newest slot is (cursor-1) mod slotCount)
// buffer(0) = device float* where the metric is written
// buffer(1) = cursor int
// buffer(2) = slotCount int
// Single-threaded 16x12 grid sampling to avoid atomics; metric is ~O(200) reads.
// ──────────────────────────────────────────────────────────────────────

kernel void estimateGlobalMotion(
    texture2d<float, access::read> currentRGB [[texture(0)]],
    texture2d_array<float, access::read> lumaHistory [[texture(1)]],
    device float* motionMetric [[buffer(0)]],
    constant int &cursor [[buffer(1)]],
    constant int &slotCount [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Single thread computes the metric to avoid atomics.
    if (gid.x != 0 || gid.y != 0) return;

    int w = int(currentRGB.get_width());
    int h = int(currentRGB.get_height());
    if (w <= 0 || h <= 0 || slotCount <= 0) {
        *motionMetric = 0.0;
        return;
    }

    int newestSlot = (cursor - 1 + slotCount) % slotCount;

    const int gridW = 32;
    const int gridH = 24;
    float sumDiff = 0.0;
    int count = 0;

    for (int gy = 0; gy < gridH; gy++) {
        for (int gx = 0; gx < gridW; gx++) {
            int x = (w * gx) / gridW;
            int y = (h * gy) / gridH;
            uint2 coord = uint2(clamp(x, 0, w - 1), clamp(y, 0, h - 1));
            float curY = rgb2yuv(currentRGB.read(coord).rgb).x;
            float histY = lumaHistory.read(coord, newestSlot).r;
            sumDiff += abs(curY - histY);
            count++;
        }
    }

    // Mean across all tiles. With a properly tuned threshold this catches
    // both full-frame and partial motion adequately without expensive sorting.
    *motionMetric = (count > 0) ? (sumDiff / float(count)) : 0.0;
}

// ──────────────────────────────────────────────────────────────────────
// STORE CHROMA HISTORY — Extract half-res UV from full-res denoised RGB
// for the temporal chroma ring buffer.
// texture(0) = full-res denoised RGB input
// texture(1) = half-res 2D-array UV output (chroma ring slot)
// buffer(0) = StoreChromaParams { int slice; }
// ──────────────────────────────────────────────────────────────────────

struct StoreChromaParams {
    int slice;
};

kernel void storeChromaHistory(
    texture2d<float, access::read>  fullResRGB [[texture(0)]],
    texture2d_array<float, access::write> halfResUVArray [[texture(1)]],
    constant StoreChromaParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= halfResUVArray.get_width() || gid.y >= halfResUVArray.get_height()) return;

    int w = int(fullResRGB.get_width());
    int h = int(fullResRGB.get_height());
    int baseX = int(gid.x) * 2;
    int baseY = int(gid.y) * 2;

    float2 sumUV = float2(0.0);
    float count = 0.0;
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int sx = baseX + dx;
            int sy = baseY + dy;
            if (sx >= w || sy >= h) continue;
            sumUV += rgb2yuv(fullResRGB.read(uint2(sx, sy)).rgb).yz;
            count += 1.0;
        }
    }

    float2 uv = (count > 0.0) ? (sumUV / count) : float2(0.5);
    halfResUVArray.write(float4(uv.x, uv.y, 0.0, 1.0), gid, params.slice);
}

kernel void storeLumaHistory(
    texture2d<float, access::read>  fullResRGB [[texture(0)]],
    texture2d_array<float, access::write> lumaArray [[texture(1)]],
    constant StoreChromaParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= lumaArray.get_width() || gid.y >= lumaArray.get_height()) return;
    float4 px = fullResRGB.read(gid);
    float y = dot(px.rgb, float3(0.2126, 0.7152, 0.0722));
    lumaArray.write(float4(y, 0.0, 0.0, 1.0), gid, params.slice);
}
