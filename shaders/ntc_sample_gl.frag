#version 450
#extension GL_GOOGLE_include_directive : require
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;
#define NTC_GL 1
#define NTC_NO_OUT 1
#include "ntc_decode.glsl"

void main() {
    uvec4 h0 = g_desc[0];
    uint mip = g_pc.x;
    uint W = max(h0.x >> mip, 1u);
    uint H = max(h0.y >> mip, 1u);
    // gl_FragCoord origin is the lower left. glReadPixels is the same origin,
    // and decode_mip row 0 is v = 0.5/H, so no flip is applied.
    vec2 uv = gl_FragCoord.xy / vec2(float(W), float(H));
    uint filt = g_pc.y;
    if (filt == 0u) {
        ntc_eval(uv, int(mip));
    } else {
        vec2 dudx = dFdx(v_uv);
        vec2 dudy = dFdy(v_uv);
        ntc_eval_filtered(uv, dudx, dudy, float(mip), g_desc[2].w, filt, g_desc[3].y);
    }
    float r = g_y[0];
    float gch = h0.z > 1u ? g_y[1] : 0.0;
    float b = h0.z > 2u ? g_y[2] : 0.0;
    float a = h0.z > 3u ? g_y[3] : 1.0;
    out_color = vec4(r, gch, b, a);
}
