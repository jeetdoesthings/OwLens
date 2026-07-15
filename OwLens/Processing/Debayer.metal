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
    float _pad;
};

struct WhiteBalanceParams {
    float3 gains;
    float3x3 colorMatrix;
};

// Sample with bounds check; returns whether sample is inside (for edge averages).
static inline bool sampleBayerValid(texture2d<float, access::read> tex, int x, int y, int dx, int dy, thread float &outV) {
    int nx = x + dx;
    int ny = y + dy;
    int w = int(tex.get_width());
    int h = int(tex.get_height());
    if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
        outV = 0.0;
        return false;
    }
    outV = tex.read(uint2(nx, ny)).r;
    return true;
}

static inline float linearize(float raw, float black, float white) {
    float denom = max(white - black, 1e-6);
    return saturate((raw - black) / denom);
}

// Average only in-bounds neighbors — avoids dark vignette from clamp-duplicating edges.
static inline float avgLinNeighbors4(texture2d<float, access::read> tex, int x, int y,
                                     int dx0, int dy0, int dx1, int dy1, int dx2, int dy2, int dx3, int dy3,
                                     float black, float white) {
    float s = 0.0;
    float n = 0.0;
    float v;
    if (sampleBayerValid(tex, x, y, dx0, dy0, v)) { s += linearize(v, black, white); n += 1.0; }
    if (sampleBayerValid(tex, x, y, dx1, dy1, v)) { s += linearize(v, black, white); n += 1.0; }
    if (sampleBayerValid(tex, x, y, dx2, dy2, v)) { s += linearize(v, black, white); n += 1.0; }
    if (sampleBayerValid(tex, x, y, dx3, dy3, v)) { s += linearize(v, black, white); n += 1.0; }
    return n > 0.0 ? s / n : 0.0;
}

static inline float avgLinNeighbors2(texture2d<float, access::read> tex, int x, int y,
                                     int dx0, int dy0, int dx1, int dy1,
                                     float black, float white) {
    float s = 0.0;
    float n = 0.0;
    float v;
    if (sampleBayerValid(tex, x, y, dx0, dy0, v)) { s += linearize(v, black, white); n += 1.0; }
    if (sampleBayerValid(tex, x, y, dx1, dy1, v)) { s += linearize(v, black, white); n += 1.0; }
    return n > 0.0 ? s / n : 0.0;
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

    float r, g, b;
    float center;
    sampleBayerValid(rawTexture, x, y, 0, 0, center);
    float cLin = linearize(center, black, white);

    if (yEven && xEven) {
        r = cLin;
        g = avgLinNeighbors4(rawTexture, x, y, -1, 0, 1, 0, 0, -1, 0, 1, black, white);
        b = avgLinNeighbors4(rawTexture, x, y, -1, -1, 1, -1, -1, 1, 1, 1, black, white);
    } else if (yEven && !xEven) {
        g = cLin;
        r = avgLinNeighbors2(rawTexture, x, y, -1, 0, 1, 0, black, white);
        b = avgLinNeighbors2(rawTexture, x, y, 0, -1, 0, 1, black, white);
    } else if (!yEven && xEven) {
        g = cLin;
        b = avgLinNeighbors2(rawTexture, x, y, -1, 0, 1, 0, black, white);
        r = avgLinNeighbors2(rawTexture, x, y, 0, -1, 0, 1, black, white);
    } else {
        b = cLin;
        g = avgLinNeighbors4(rawTexture, x, y, -1, 0, 1, 0, 0, -1, 0, 1, black, white);
        r = avgLinNeighbors4(rawTexture, x, y, -1, -1, 1, -1, -1, 1, 1, 1, black, white);
    }

    // Mild optical falloff compensation (sensor vignette without shading map)
    // Strength kept low so center stays natural.
    float2 uv = float2(float(x) + 0.5, float(y) + 0.5) / float2(float(outTexture.get_width()), float(outTexture.get_height()));
    float2 d = uv - float2(0.5, 0.5);
    float r2 = dot(d, d); // 0 center → ~0.5 corners
    float vignetteGain = 1.0 + 0.35 * r2; // lift corners slightly
    float3 rgb = float3(r, g, b) * vignetteGain;
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

kernel void applyLogCurve(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant int &curveType [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    float4 pixel = inTexture.read(gid);
    float3 rgb = float3(pixel.r, pixel.g, pixel.b);
    float3 result;

    if (curveType == 0) {
        result = saturate(rgb);
    } else {
        // Sony S-Log3 published OETF on scene-linear (18% mid grey ≈ 0.18)
        // Output code values roughly 0–1 (10-bit /1023).
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

    outTexture.write(float4(result, 1.0), gid);
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
    float  _pad;
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

    // ── Demosaic (bilinear) ──
    float r, g, b;
    float center;
    sampleBayerValid(rawTexture, x, y, 0, 0, center);
    float cLin = linearize(center, black, white);

    if (yEven && xEven) {
        r = cLin;
        g = avgLinNeighbors4(rawTexture, x, y, -1, 0, 1, 0, 0, -1, 0, 1, black, white);
        b = avgLinNeighbors4(rawTexture, x, y, -1, -1, 1, -1, -1, 1, 1, 1, black, white);
    } else if (yEven && !xEven) {
        g = cLin;
        r = avgLinNeighbors2(rawTexture, x, y, -1, 0, 1, 0, black, white);
        b = avgLinNeighbors2(rawTexture, x, y, 0, -1, 0, 1, black, white);
    } else if (!yEven && xEven) {
        g = cLin;
        b = avgLinNeighbors2(rawTexture, x, y, -1, 0, 1, 0, black, white);
        r = avgLinNeighbors2(rawTexture, x, y, 0, -1, 0, 1, black, white);
    } else {
        b = cLin;
        g = avgLinNeighbors4(rawTexture, x, y, -1, 0, 1, 0, 0, -1, 0, 1, black, white);
        r = avgLinNeighbors4(rawTexture, x, y, -1, -1, 1, -1, -1, 1, 1, 1, black, white);
    }

    // Mild vignette compensation
    float2 uv = float2(float(x) + 0.5, float(y) + 0.5) / float2(float(outTexture.get_width()), float(outTexture.get_height()));
    float2 d = uv - float2(0.5, 0.5);
    float r2 = dot(d, d);
    float vignetteGain = 1.0 + 0.35 * r2;
    float3 rgb = float3(r, g, b) * vignetteGain;
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
    constant int &showClipping [[buffer(2)]]
) {
    float2 uv = float2(in.position.x - destOffset.x, in.position.y - destOffset.y) / float2(destSize);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float4 color = tex.sample(s, uv);
    
    float isClipped = step(color.a, 0.5);
    float applyRed = (showClipping > 0) ? isClipped : 0.0;
    
    return float4(mix(color.rgb, float3(1.0, 0.0, 0.0), applyRed), 1.0);
}
