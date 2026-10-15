#version 450
// The touch controls (gpu/src/touch_controls.cpp): a quad per control, in
// window pixels, its shape drawn by touch.frag.
layout(push_constant) uniform Screen { vec2 size; } screen;
layout(location = 0) in vec4 a_circle;      // centre x, y; radius; opacity
layout(location = 1) in vec4 a_kind;        // kind; held
layout(location = 0) out vec2 v_local;      // -1.1..1.1 across the control
layout(location = 1) flat out vec4 v_kind;
layout(location = 2) flat out float v_alpha;

const vec2 kCorners[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

void main() {
    vec2 corner = kCorners[gl_VertexIndex] * 1.1;
    vec2 at = a_circle.xy + corner * a_circle.z;
    gl_Position = vec4(at / screen.size * 2.0 - 1.0, 0.0, 1.0);
    v_local = corner;
    v_kind = a_kind;
    v_alpha = a_circle.w;
}
