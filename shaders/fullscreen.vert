#version 450
layout(location = 0) out vec2 v_uv;
void main() {
    vec2 pos = vec2(float((gl_VertexIndex & 1) << 2) - 1.0, float((gl_VertexIndex & 2) << 1) - 1.0);
    gl_Position = vec4(pos, 0.0, 1.0);
    v_uv = pos * 0.5 + 0.5;
}
