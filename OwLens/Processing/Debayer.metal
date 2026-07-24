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

kernel void debayerBilinear(
    texture2d<float, access::read> rawTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant DebayerParams &params [[buffer(0)]],
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

    // ── Directional Demosaic (Malvar-He-Cutler) ──
    float c00 = sampleBayerClamp(rawTexture, x, y, 0, 0, black, white);
    float cN1 = sampleBayerClamp(rawTexture, x, y, 0, -1, black, white);
    float cS1 = sampleBayerClamp(rawTexture, x, y, 0, 1, black, white);
    float cE1 = sampleBayerClamp(rawTexture, x, y, 1, 0, black, white);
    float cW1 = sampleBayerClamp(rawTexture, x, y, -1, 0, black, white);
    
    float cN2 = sampleBayerClamp(rawTexture, x, y, 0, -2, black, white);
    float cS2 = sampleBayerClamp(rawTexture, x, y, 0, 2, black, white);
    float cE2 = sampleBayerClamp(rawTexture, x, y, 2, 0, black, white);
    float cW2 = sampleBayerClamp(rawTexture, x, y, -2, 0, black, white);
    
    float cNE = sampleBayerClamp(rawTexture, x, y, 1, -1, black, white);
    float cNW = sampleBayerClamp(rawTexture, x, y, -1, -1, black, white);
    float cSE = sampleBayerClamp(rawTexture, x, y, 1, 1, black, white);
    float cSW = sampleBayerClamp(rawTexture, x, y, -1, 1, black, white);

    float G_at_RB = (2*(cN1 + cS1 + cE1 + cW1) + 4*c00 - (cN2 + cS2 + cE2 + cW2)) / 8.0;
    float Color_at_G_H = (4*(cE1 + cW1) + 5*c00 - (cE2 + cW2) - 0.5*(cN2 + cS2) - (cNE + cNW + cSE + cSW)) / 8.0;
    float Color_at_G_V = (4*(cN1 + cS1) + 5*c00 - (cN2 + cS2) - 0.5*(cE2 + cW2) - (cNE + cNW + cSE + cSW)) / 8.0;
    float Color_at_Diag = (2*(cNE + cNW + cSE + cSW) + 6*c00 - 1.5*(cN2 + cS2 + cE2 + cW2)) / 8.0;

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
    r = max(r, 0.0);
    g = max(g, 0.0);
    b = max(b, 0.0);

    // Per-channel Lens Shading Correction (LSC)
    float2 uv = float2(float(x) + 0.5, float(y) + 0.5) / float2(float(outTexture.get_width()), float(outTexture.get_height()));
    float2 d = uv - float2(0.5, 0.5);
    float r2 = dot(d, d);
    float gainR = 1.0 + params.lscCoefficients[0] * r2;
    float gainG = 1.0 + ((params.lscCoefficients[1] + params.lscCoefficients[2]) * 0.5) * r2;
    float gainB = 1.0 + params.lscCoefficients[3] * r2;
    float3 rgb = float3(r * gainR, g * gainG, b * gainB);
    rgb = min(rgb, float3(8.0)); // allow headroom pre-log

    outTexture.write(float4(rgb, 1.0), gid);
}

kernel void applyWhiteBalanceAndColorMatrix(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant WhiteBalanceParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    float4 pixel = inTexture.read(gid);
    float3 rgb = float3(pixel.r, pixel.g, pixel.b);

    rgb *= params.gains;
    rgb = params.colorMatrix * rgb;
    rgb = max(rgb, float3(0.0));

    outTexture.write(float4(rgb, 1.0), gid);
}

static inline float3 encodeLogCurve(float3 rgb, int curveType) {
    if (curveType == 0) {
        return saturate(rgb);
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

kernel void applyLogCurve(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant int &curveType [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    float4 pixel = inTexture.read(gid);
    float3 result = encodeLogCurve(float3(pixel.r, pixel.g, pixel.b), curveType);
    outTexture.write(float4(result, pixel.a), gid);
}

kernel void applyLogOnly(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant int &curveType [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    float4 pixel = inTexture.read(gid);
    float3 result = encodeLogCurve(float3(pixel.r, pixel.g, pixel.b), curveType);
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
};

kernel void debayerWBLog(
    texture2d<float, access::read>  rawTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant FusedParams &params   [[buffer(0)]],
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

    // ── Directional Demosaic (Malvar-He-Cutler) ──
    float c00 = sampleBayerClamp(rawTexture, x, y, 0, 0, black, white);
    float cN1 = sampleBayerClamp(rawTexture, x, y, 0, -1, black, white);
    float cS1 = sampleBayerClamp(rawTexture, x, y, 0, 1, black, white);
    float cE1 = sampleBayerClamp(rawTexture, x, y, 1, 0, black, white);
    float cW1 = sampleBayerClamp(rawTexture, x, y, -1, 0, black, white);
    
    float cN2 = sampleBayerClamp(rawTexture, x, y, 0, -2, black, white);
    float cS2 = sampleBayerClamp(rawTexture, x, y, 0, 2, black, white);
    float cE2 = sampleBayerClamp(rawTexture, x, y, 2, 0, black, white);
    float cW2 = sampleBayerClamp(rawTexture, x, y, -2, 0, black, white);
    
    float cNE = sampleBayerClamp(rawTexture, x, y, 1, -1, black, white);
    float cNW = sampleBayerClamp(rawTexture, x, y, -1, -1, black, white);
    float cSE = sampleBayerClamp(rawTexture, x, y, 1, 1, black, white);
    float cSW = sampleBayerClamp(rawTexture, x, y, -1, 1, black, white);

    float G_at_RB = (2*(cN1 + cS1 + cE1 + cW1) + 4*c00 - (cN2 + cS2 + cE2 + cW2)) / 8.0;
    float Color_at_G_H = (4*(cE1 + cW1) + 5*c00 - (cE2 + cW2) - 0.5*(cN2 + cS2) - (cNE + cNW + cSE + cSW)) / 8.0;
    float Color_at_G_V = (4*(cN1 + cS1) + 5*c00 - (cN2 + cS2) - 0.5*(cE2 + cW2) - (cNE + cNW + cSE + cSW)) / 8.0;
    float Color_at_Diag = (2*(cNE + cNW + cSE + cSW) + 6*c00 - 1.5*(cN2 + cS2 + cE2 + cW2)) / 8.0;

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
    
    r = max(r, 0.0);
    g = max(g, 0.0);
    b = max(b, 0.0);

    // Per-channel Lens Shading Correction (LSC)
    float2 uv = float2(float(x) + 0.5, float(y) + 0.5) / float2(float(outTexture.get_width()), float(outTexture.get_height()));
    float2 d = uv - float2(0.5, 0.5);
    float r2 = dot(d, d);
    float gainR = 1.0 + params.lscCoefficients[0] * r2;
    float gainG = 1.0 + ((params.lscCoefficients[1] + params.lscCoefficients[2]) * 0.5) * r2;
    float gainB = 1.0 + params.lscCoefficients[3] * r2;
    float3 rgb = float3(r * gainR, g * gainG, b * gainB);
    rgb = min(rgb, float3(8.0));

    // ── White Balance ──
    rgb *= params.wbGains;
    rgb = max(rgb, float3(0.0));

    // ── Log curve ──
    float3 result;
    int curveType = params.curveType;

    if (curveType == 0) {
        result = saturate(rgb);
    } else {
        // Sony S-Log3
        float3 clamped = max(rgb, float3(0.0));
        for (int i = 0; i < 3; i++) {
            float lin = clamped[i];
            if (lin >= 0.01125) {
                result[i] = (420.0 + log10((lin + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0;
            } else {
                result[i] = (lin * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0;
            }
        }
        result = saturate(result);
    }

    bool isClipped = (r >= 0.99 || g >= 0.99 || b >= 0.99);
    float alpha = isClipped ? 0.0 : 1.0;
    outTexture.write(float4(result, alpha), gid);
}

// ──────────────────────────────────────────────────────────────────────
// LINEAR OUTPUT: demosaic + LSC + WB — NO log curve.
// Used with spatialDenoise → applyLogOnly 3-pass pipeline.
// ──────────────────────────────────────────────────────────────────────

kernel void debayerWBLinear(
    texture2d<float, access::read>  rawTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant FusedParams &params   [[buffer(0)]],
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

    // ── Directional Demosaic (Malvar-He-Cutler) ──
    float c00 = sampleBayerClamp(rawTexture, x, y, 0, 0, black, white);
    float cN1 = sampleBayerClamp(rawTexture, x, y, 0, -1, black, white);
    float cS1 = sampleBayerClamp(rawTexture, x, y, 0, 1, black, white);
    float cE1 = sampleBayerClamp(rawTexture, x, y, 1, 0, black, white);
    float cW1 = sampleBayerClamp(rawTexture, x, y, -1, 0, black, white);
    
    float cN2 = sampleBayerClamp(rawTexture, x, y, 0, -2, black, white);
    float cS2 = sampleBayerClamp(rawTexture, x, y, 0, 2, black, white);
    float cE2 = sampleBayerClamp(rawTexture, x, y, 2, 0, black, white);
    float cW2 = sampleBayerClamp(rawTexture, x, y, -2, 0, black, white);
    
    float cNE = sampleBayerClamp(rawTexture, x, y, 1, -1, black, white);
    float cNW = sampleBayerClamp(rawTexture, x, y, -1, -1, black, white);
    float cSE = sampleBayerClamp(rawTexture, x, y, 1, 1, black, white);
    float cSW = sampleBayerClamp(rawTexture, x, y, -1, 1, black, white);

    float G_at_RB = (2*(cN1 + cS1 + cE1 + cW1) + 4*c00 - (cN2 + cS2 + cE2 + cW2)) / 8.0;
    float Color_at_G_H = (4*(cE1 + cW1) + 5*c00 - (cE2 + cW2) - 0.5*(cN2 + cS2) - (cNE + cNW + cSE + cSW)) / 8.0;
    float Color_at_G_V = (4*(cN1 + cS1) + 5*c00 - (cN2 + cS2) - 0.5*(cE2 + cW2) - (cNE + cNW + cSE + cSW)) / 8.0;
    float Color_at_Diag = (2*(cNE + cNW + cSE + cSW) + 6*c00 - 1.5*(cN2 + cS2 + cE2 + cW2)) / 8.0;

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
    r = max(r, 0.0);
    g = max(g, 0.0);
    b = max(b, 0.0);

    // Per-channel Lens Shading Correction (LSC)
    float2 uv = float2(float(x) + 0.5, float(y) + 0.5) / float2(float(outTexture.get_width()), float(outTexture.get_height()));
    float2 d = uv - float2(0.5, 0.5);
    float r2 = dot(d, d);
    float gainR = 1.0 + params.lscCoefficients[0] * r2;
    float gainG = 1.0 + ((params.lscCoefficients[1] + params.lscCoefficients[2]) * 0.5) * r2;
    float gainB = 1.0 + params.lscCoefficients[3] * r2;
    float3 rgb = float3(r * gainR, g * gainG, b * gainB);
    rgb = min(rgb, float3(8.0));

    // ── White Balance ──
    rgb *= params.wbGains;
    rgb = max(rgb, float3(0.0));

    // Clipping flag on raw demosaiced values (sensor saturation)
    bool isClipped = (r >= 0.99 || g >= 0.99 || b >= 0.99);
    float alpha = isClipped ? 0.0 : 1.0;

    // Output scene-linear RGB (NO log curve)
    outTexture.write(float4(rgb, alpha), gid);
}

kernel void applyClippingOverlay(
    texture2d<float, access::read_write> tex [[texture(0)]],
    constant int &showClipping [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
    if (showClipping == 0) return;
    
    float4 color = tex.read(gid);
    float isClipped = step(color.a, 0.5);
    float3 finalColor = mix(color.rgb, float3(1.0, 0.0, 0.0), isClipped);
    tex.write(float4(finalColor, 1.0), gid);
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
    constant int &overlayOnly [[buffer(4)]]
) {
    float2 uv = float2(in.position.x - destOffset.x, in.position.y - destOffset.y) / float2(destSize);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return overlayOnly > 0 ? float4(0.0, 0.0, 0.0, 0.0) : float4(0.0, 0.0, 0.0, 1.0);
    }
    
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float4 color = tex.sample(s, uv);
    
    float isClipped = step(color.a, 0.5);
    float applyRed = (showClipping > 0) ? isClipped : 0.0;
    
    float3 finalColor = mix(color.rgb, float3(1.0, 0.0, 0.0), applyRed);
    float overlayAlpha = (overlayOnly > 0 && applyRed > 0.0) ? 0.75 : 0.0;
    
    if (showFocusPeaking > 0) {
        // Lightweight edge detection (Laplacian approximation)
        float2 texel = 1.0 / float2(tex.get_width(), tex.get_height());
        
        float c = (color.r + color.g + color.b) / 3.0;
        float top = (tex.sample(s, uv + float2(0, -texel.y)).r + tex.sample(s, uv + float2(0, -texel.y)).g + tex.sample(s, uv + float2(0, -texel.y)).b) / 3.0;
        float bottom = (tex.sample(s, uv + float2(0, texel.y)).r + tex.sample(s, uv + float2(0, texel.y)).g + tex.sample(s, uv + float2(0, texel.y)).b) / 3.0;
        float left = (tex.sample(s, uv + float2(-texel.x, 0)).r + tex.sample(s, uv + float2(-texel.x, 0)).g + tex.sample(s, uv + float2(-texel.x, 0)).b) / 3.0;
        float right = (tex.sample(s, uv + float2(texel.x, 0)).r + tex.sample(s, uv + float2(texel.x, 0)).g + tex.sample(s, uv + float2(texel.x, 0)).b) / 3.0;
        
        float edge = abs(top + bottom + left + right - 4.0 * c);
        
        // Threshold for edge detection
        if (edge > 0.05) {
            finalColor = float3(0.0, 1.0, 0.0); // Bright Green
            overlayAlpha = 1.0;
        }
    }
    
    if (overlayOnly > 0) {
        if (overlayAlpha <= 0.0) {
            return float4(0.0, 0.0, 0.0, 0.0);
        }
        return float4(finalColor, overlayAlpha);
    }
    return float4(finalColor, 1.0);
}

// ──────────────────────────────────────────────────────────────────────
// PHASE 3: CHROMA BILATERAL DENOISING
// ──────────────────────────────────────────────────────────────────────

static inline float3 rgb2yuv(float3 rgb) {
    float y = dot(rgb, float3(0.299, 0.587, 0.114));
    float u = dot(rgb, float3(-0.1687, -0.3313, 0.5)) + 0.5;
    float v = dot(rgb, float3(0.5, -0.4187, -0.0813)) + 0.5;
    return float3(y, u, v);
}

static inline float3 yuv2rgb(float3 yuv) {
    float y = yuv.x;
    float u = yuv.y - 0.5;
    float v = yuv.z - 0.5;
    float r = y + 1.402 * v;
    float g = y - 0.3441 * u - 0.7141 * v;
    float b = y + 1.772 * u;
    return float3(r, g, b);
}

struct BilateralParams {
    float iso;
};

kernel void chromaBilateral(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant BilateralParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;
    
    float3 centerRGB = inTexture.read(gid).rgb;
    float3 centerYUV = rgb2yuv(centerRGB);
    
    float sumLuma = 0.0;
    float sumLumaWeights = 0.0;
    float sumChromaWeights = 0.0;
    float2 sumUV = float2(0.0);
    
    // Dynamic ISO-scaled Luma edge threshold
    float iso = max(params.iso, 33.0);
    float isoScale = sqrt(iso / 33.0);
    float chromaRangeSigma = 0.02 * isoScale;
    float chromaRangeSigma2 = chromaRangeSigma * chromaRangeSigma;
    float lumaRangeSigma = chromaRangeSigma * 0.45; // tighter than chroma to preserve detail
    float lumaRangeSigma2 = lumaRangeSigma * lumaRangeSigma;
    
    int radius = (iso > 200.0) ? 3 : 2;
    float maxDist2 = float(radius * radius);
    float spatialSigma2 = float(radius * radius) * 0.5;
    
    int w = inTexture.get_width();
    int h = inTexture.get_height();
    int cx = int(gid.x);
    int cy = int(gid.y);
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            float dist2 = dx*dx + dy*dy;
            if (dist2 > maxDist2) continue; // Skip corners -> makes it a diamond
            
            uint2 pid = uint2(clamp(cx + dx, 0, w - 1), clamp(cy + dy, 0, h - 1));
            float3 yuv = rgb2yuv(inTexture.read(pid).rgb);
            
            float spatialWeight = exp(-dist2 / (2.0 * spatialSigma2));
            
            float lumaDiff = yuv.x - centerYUV.x;
            
            // CROSS-BILATERAL: Use ONLY Luma difference as the edge-stopping function.
            // If we include chroma difference, the filter refuses to blur massive color noise!
            float lumaWeight = spatialWeight * exp(-(lumaDiff * lumaDiff) / (2.0 * lumaRangeSigma2));
            float chromaWeight = spatialWeight * exp(-(lumaDiff * lumaDiff) / (2.0 * chromaRangeSigma2));
            
            sumLumaWeights += lumaWeight;
            sumLuma += yuv.x * lumaWeight;
            sumChromaWeights += chromaWeight;
            sumUV += yuv.yz * chromaWeight;
        }
    }
    
    float finalY = (sumLumaWeights > 0.0001) ? (sumLuma / sumLumaWeights) : centerYUV.x;
    float2 finalUV = (sumChromaWeights > 0.0001) ? (sumUV / sumChromaWeights) : centerYUV.yz;
    float3 finalYUV = float3(finalY, finalUV.x, finalUV.y);
    float3 finalRGB = yuv2rgb(finalYUV); // Spatially denoised current frame
    
    
    float alpha = inTexture.read(gid).a;
    outTexture.write(float4(finalRGB, alpha), gid);
}

// ──────────────────────────────────────────────────────────────────────
// SPATIAL DENOISING (Linear Space)
// Bilateral filter on BOTH luma and chroma with ISO-adaptive strength.
// Operates before log curve for better noise statistics.
// ──────────────────────────────────────────────────────────────────────

struct DenoiseParams {
    float iso;
    int   radius;
};

struct TemporalParams {
    float iso;
    float maxBlend;
};

kernel void spatialDenoise(
    texture2d<float, access::read>  inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
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
    float isoScale = sqrt(iso / 33.0);
    float luma01 = saturate(centerYUV.x);
    // Shot-noise-aware proxy: push stronger denoise in shadows, lighter in highlights.
    float shadowBoost = mix(1.45, 0.8, luma01);
    
    // Luma: tight sigma — smooth flat-area grain, preserve edges
    float lumaRS = 0.012 * isoScale * shadowBoost;
    float lumaRS2 = lumaRS * lumaRS;
    
    // Chroma: wider sigma — stronger color noise reduction
    float chromaRS = 0.035 * isoScale * shadowBoost;
    float chromaRS2 = chromaRS * chromaRS;
    
    // Spatial sigma adapts to kernel radius
    float spatialS2 = float(radius) * float(radius) * 0.5;
    
    int w = inTexture.get_width();
    int h = inTexture.get_height();
    int cx = int(gid.x);
    int cy = int(gid.y);
    
    float sumLuma = 0.0;
    float sumLumaW = 0.0;
    float2 sumUV = float2(0.0);
    float sumChromaW = 0.0;
    
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
            
            // Chroma bilateral: edge-stopped by luma difference (wider threshold)
            float chromaW = spatialW * exp(-(lumaDiff * lumaDiff) / (2.0 * chromaRS2));
            sumChromaW += chromaW;
            sumUV += sYUV.yz * chromaW;
        }
    }
    
    float finalY = (sumLumaW > 1e-4) ? (sumLuma / sumLumaW) : centerYUV.x;
    float2 finalUV = (sumChromaW > 1e-4) ? (sumUV / sumChromaW) : centerYUV.yz;
    
    float3 finalRGB = max(yuv2rgb(float3(finalY, finalUV.x, finalUV.y)), float3(0.0));
    outTexture.write(float4(finalRGB, centerPx.a), gid);
}

kernel void temporalDenoise(
    texture2d<float, access::read>  currentTexture [[texture(0)]],
    texture2d<float, access::read>  historyTexture [[texture(1)]],
    texture2d<float, access::write> outTexture [[texture(2)]],
    constant TemporalParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    float4 currentPx = currentTexture.read(gid);
    float4 historyPx = historyTexture.read(gid);
    float3 currentYUV = rgb2yuv(currentPx.rgb);
    float3 historyYUV = rgb2yuv(historyPx.rgb);

    float iso = max(params.iso, 33.0);
    float isoScale = sqrt(iso / 33.0);

    float luma01 = saturate(currentYUV.x);
    float shadowBoost = mix(1.4, 0.85, luma01);
    float lumaThreshold = max(0.006 * isoScale * shadowBoost, 1e-5);
    float chromaThreshold = max(0.010 * isoScale * shadowBoost, 1e-5);

    float lumaDiff = abs(currentYUV.x - historyYUV.x);
    float chromaDiff = length(currentYUV.yz - historyYUV.yz);

    float lumaMotion = saturate(lumaDiff / lumaThreshold);
    float chromaMotion = saturate(chromaDiff / chromaThreshold);
    float motion = max(lumaMotion, chromaMotion);

    float historyBlend = clamp(params.maxBlend * (1.0 - motion), 0.0, 0.95);
    float3 mixedRGB = mix(currentPx.rgb, historyPx.rgb, historyBlend);
    outTexture.write(float4(max(mixedRGB, float3(0.0)), currentPx.a), gid);
}
