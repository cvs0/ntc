#version 450
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_nrm;
layout(location = 2) in vec2 in_uv;
layout(location = 0) out vec3 v_world;
layout(location = 1) out vec3 v_nrm;
layout(location = 2) out vec2 v_uv;

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

void main() {
    vec4 w = u_model * vec4(in_pos, 1.0);
    v_world = w.xyz;
    v_nrm = mat3(u_model) * in_nrm;
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
