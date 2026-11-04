#version 450
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;
layout(set = 0, binding = 0) uniform sampler2D u_tex;
layout(push_constant) uniform Pc { uvec4 g_pc; };

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(float(g_pc.x), float(g_pc.y));
    out_color = texture(u_tex, uv);
}
