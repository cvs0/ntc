#version 450
#extension GL_GOOGLE_include_directive : require
layout(location = 0) in vec3 v_world;
layout(location = 1) in vec3 v_nrm;
layout(location = 2) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

#define NTC_NO_OUT 1
#include "ntc_decode.glsl"

layout(set = 0, binding = 2) uniform sampler2DArray u_ref;
layout(set = 0, binding = 3) uniform sampler2D u_bc7;
layout(set = 0, binding = 4) uniform sampler2D u_bc5;
layout(set = 0, binding = 5) uniform sampler2D u_bc7_orm;
layout(set = 0, binding = 6) uniform sampler2D u_bc1;
layout(set = 0, binding = 7) uniform sampler2D u_bc1_orm;
layout(set = 0, binding = 9) uniform sampler2D u_astc;
layout(set = 0, binding = 10) uniform sampler2D u_overlay;
layout(std140, set = 0, binding = 8) uniform Frame {
    mat4 u_mvp;
    mat4 u_model;
    vec4 u_eye;
    vec4 u_light;
    uvec4 u_sw;
    uvec4 u_sem0;
    uvec4 u_sem1;
    uvec4 u_sem2;
};

struct Mat {
    vec3 alb;
    vec3 nrm;
    float rough;
    float metal;
    float ao;
};

float ch(uint i) {
    if (i >= 16u) return 0.0;
    return g_y[i];
}

Mat eval_ntc() {
    vec2 dx = dFdx(v_uv);
    vec2 dy = dFdy(v_uv);
    float tex = max(float(g_desc[0].x), float(g_desc[0].y));
    float rho = max(length(dx), length(dy));
    float lod = log2(max(rho * tex, 1e-6));
    ntc_eval_filtered(v_uv, dx, dy, lod, g_desc[2].w, 4u, g_desc[3].y);
    Mat m;
    m.alb = vec3(ch(u_sem0.x), ch(u_sem0.y), ch(u_sem0.z));
    m.nrm = vec3(ch(u_sem0.w), ch(u_sem1.x), ch(u_sem1.y));
    m.rough = u_sem1.z < 16u ? ch(u_sem1.z) : 0.5;
    m.metal = u_sem1.w < 16u ? ch(u_sem1.w) : 0.0;
    m.ao = u_sem2.x < 16u ? ch(u_sem2.x) : 1.0;
    return m;
}

Mat eval_ref() {
    Mat m;
    m.alb = texture(u_ref, vec3(v_uv, 0.0)).rgb;
    m.nrm = texture(u_ref, vec3(v_uv, 1.0)).rgb;
    vec3 orm = texture(u_ref, vec3(v_uv, 2.0)).rgb;
    m.ao = orm.r;
    m.rough = orm.g;
    m.metal = orm.b;
    return m;
}

Mat eval_bc7() {
    Mat m;
    m.alb = texture(u_bc7, v_uv).rgb;
    vec2 xy = texture(u_bc5, v_uv).rg * 2.0 - 1.0;
    float z2 = max(1.0 - dot(xy, xy), 0.0);
    m.nrm = vec3(xy, sqrt(z2)) * 0.5 + 0.5;
    vec3 orm = texture(u_bc7_orm, v_uv).rgb;
    m.ao = orm.r;
    m.rough = orm.g;
    m.metal = orm.b;
    return m;
}

Mat eval_bc1() {
    Mat m;
    m.alb = texture(u_bc1, v_uv).rgb;
    vec2 xy = texture(u_bc5, v_uv).rg * 2.0 - 1.0;
    float z2 = max(1.0 - dot(xy, xy), 0.0);
    m.nrm = vec3(xy, sqrt(z2)) * 0.5 + 0.5;
    vec3 orm = texture(u_bc1_orm, v_uv).rgb;
    m.ao = orm.r;
    m.rough = orm.g;
    m.metal = orm.b;
    return m;
}

Mat eval_astc() {
    Mat m;
    m.alb = texture(u_astc, v_uv).rgb;
    m.nrm = vec3(0.5, 0.5, 1.0);
    m.rough = 0.5;
    m.metal = 0.0;
    m.ao = 1.0;
    return m;
}

Mat eval_mode(uint mode) {
    if (mode == 1u) return eval_bc7();
    if (mode == 2u) return eval_bc1();
    if (mode == 3u) return eval_astc();
    if (mode == 4u) return eval_ref();
    return eval_ntc();
}

vec3 shade(Mat m) {
    vec3 ntex = m.nrm * 2.0 - 1.0;
    vec3 Ngeom = normalize(v_nrm);
    vec3 T = dFdx(v_world);
    vec3 B = dFdy(v_world);
    if (dot(cross(T, B), Ngeom) < 0.0) B = -B;
    T = normalize(T);
    B = normalize(B);
    vec3 Nw = normalize(mat3(T, B, Ngeom) * ntex);
    vec3 V = normalize(u_eye.xyz - v_world);
    vec3 L = normalize(u_light.xyz);
    float ndl = max(dot(Nw, L), 0.0);
    float ndv = max(dot(Nw, V), 0.0);
    vec3 H = normalize(L + V);
    float ndh = max(dot(Nw, H), 0.0);
    float rough = clamp(m.rough, 0.045, 1.0);
    float a = rough * rough;
    float a2 = a * a;
    float den = ndh * ndh * (a2 - 1.0) + 1.0;
    float D = a2 / max(3.14159265 * den * den, 1e-4);
    float k = (rough + 1.0) * (rough + 1.0) / 8.0;
    float Gv = ndv / max(ndv * (1.0 - k) + k, 1e-4);
    float Gl = ndl / max(ndl * (1.0 - k) + k, 1e-4);
    vec3 F0 = mix(vec3(0.04), m.alb, clamp(m.metal, 0.0, 1.0));
    float sch = pow(1.0 - max(dot(H, V), 0.0), 5.0);
    vec3 F = F0 + (1.0 - F0) * sch;
    vec3 spec = D * Gv * Gl * F / max(4.0 * ndl * ndv, 1e-4);
    vec3 diff = (1.0 - F) * (1.0 - clamp(m.metal, 0.0, 1.0)) * m.alb / 3.14159265;
    vec3 amb = m.alb * 0.04 * m.ao;
    return amb + (diff + spec) * ndl * m.ao;
}

vec3 heat(vec3 err) {
    float t = clamp(max(err.r, max(err.g, err.b)) * 4.0, 0.0, 1.0);
    return vec3(t, 0.15 * (1.0 - abs(t - 0.5) * 2.0), 1.0 - t);
}

void main() {
    uint mode = u_sw.x;
    bool heat_on = (u_sw.w & 1u) != 0u;
    if (!heat_on && u_sw.y > 0u && uint(gl_FragCoord.x) >= u_sw.y) mode = u_sw.z;
    vec3 rgb;
    if (heat_on) {
        Mat a = eval_ntc();
        Mat b = eval_ref();
        rgb = heat(abs(a.alb - b.alb));
    } else {
        rgb = shade(eval_mode(mode));
    }
    vec2 suv = gl_FragCoord.xy / vec2(float(u_sem2.y), float(u_sem2.z));
    vec4 ov = texture(u_overlay, suv);
    if (ov.a > 0.5) rgb = ov.rgb;
    out_color = vec4(rgb, 1.0);
}
