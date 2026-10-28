#version 450
#extension GL_GOOGLE_include_directive : require
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;
#define NTC_NO_OUT 1
#include "ntc_decode.glsl"

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(float(g_pc.z), float(g_pc.w));
    ntc_eval(uv, 0);
    float r = g_y[0];
    float g = g_desc[0].z > 1u ? g_y[1] : 0.0;
    float b = g_desc[0].z > 2u ? g_y[2] : 0.0;
    out_color = vec4(r, g, b, 1.0);
}
