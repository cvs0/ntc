#version 450
#extension GL_GOOGLE_include_directive : require
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;
#define NTC_NO_OUT 1
#include "ntc_decode.glsl"

// On-sample path. Filter mode 3 picks one bilinear corner. The mean of many
// frames is the color bilinear. Frame is descriptor word 2, component w.
void main() {
    vec2 uv = gl_FragCoord.xy / vec2(float(g_pc.z), float(g_pc.w));
    vec2 d = vec2(1.0) / vec2(float(g_pc.z), float(g_pc.w));
    ntc_eval_filtered(uv, vec2(d.x, 0.0), vec2(0.0, d.y), 0.0, g_desc[2].w, 3u, 1u);
    float r = g_y[0];
    float g = g_desc[0].z > 1u ? g_y[1] : 0.0;
    float b = g_desc[0].z > 2u ? g_y[2] : 0.0;
    out_color = vec4(r, g, b, 1.0);
}
