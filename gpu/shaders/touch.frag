#version 450
// A touch control's look, in its own unit circle: a dark disc with a light rim,
// and its symbol - the face buttons' shapes in their colours, L1, R1, START in
// a 3x5 pixel font; the stick a ring and a knob.
layout(location = 0) in vec2 v_local;
layout(location = 1) flat in vec4 v_kind;
layout(location = 2) flat in float v_alpha;
layout(location = 0) out vec4 o_color;

const int kStickBase = 0, kStickKnob = 1, kCross = 2, kCircle = 3, kSquare = 4, kTriangle = 5,
          kL1 = 6, kR1 = 7, kStart = 8;

float Segment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// Glyphs, 3 wide and 5 high, rows top first, a row's leftmost pixel its
// highest bit: L R 1 S T A.
int Glyph(int c) {
    if (c == 0) return 0x4924 | 0x7;      // L: 100 100 100 100 111
    if (c == 1) return 0x7BAD;            // R: 111 101 110 101 101
    if (c == 2) return 0x2C97;            // 1: 010 110 010 010 111
    if (c == 3) return 0x79CF;            // S: 111 100 111 001 111
    if (c == 4) return 0x7492;            // T: 111 010 010 010 010
    return 0x2BED;                        // A: 010 101 111 101 101
}

// Text of up to 5 glyphs (codes above), centred, `height` tall in local units.
float Text(vec2 p, int n, int g0, int g1, int g2, int g3, int g4, float height) {
    float cell = height / 5.0;
    float width = float(n) * 4.0 * cell - cell;
    vec2 q = (p + vec2(width * 0.5, height * 0.5)) / cell;   // in pixels of the font, top left 0
    if (q.x < 0.0 || q.y < 0.0 || q.y >= 5.0 || q.x >= float(n) * 4.0) return 0.0;
    int index = int(q.x) / 4, column = int(q.x) % 4, row = int(q.y);
    if (column == 3) return 0.0;
    int glyph = index == 0 ? g0 : index == 1 ? g1 : index == 2 ? g2 : index == 3 ? g3 : g4;
    return float((Glyph(glyph) >> ((4 - row) * 3 + (2 - column))) & 1);
}

void main() {
    int kind = int(v_kind.x + 0.5);
    bool held = v_kind.y > 0.5;
    vec2 p = v_local;
    float d = length(p);
    float aa = fwidth(d) * 1.2;
    vec3 color = vec3(0.0);
    float alpha = 0.0;

    if (kind == kStickKnob) {
        float disc = 1.0 - smoothstep(1.0 - aa, 1.0, d);
        color = held ? vec3(0.95) : vec3(0.8);
        alpha = disc * 0.75;
    } else {
        // The disc, darker inside, and its rim.
        float disc = 1.0 - smoothstep(1.0 - aa, 1.0, d);
        float rim = disc * smoothstep(0.88 - aa, 0.88, d);
        color = mix(vec3(0.06, 0.07, 0.09), vec3(0.85), rim);
        alpha = disc * (kind == kStickBase ? 0.35 : 0.45) + rim * 0.4;
        if (held) {
            color = mix(color, vec3(1.0), 0.25);
            alpha = min(1.0, alpha + 0.2 * disc);
        }
        float stroke = 0.075, symbol = 0.0;
        vec3 tint = vec3(1.0);
        if (kind == kCross) {
            float s = min(Segment(p, vec2(-0.38, -0.38), vec2(0.38, 0.38)), Segment(p, vec2(-0.38, 0.38), vec2(0.38, -0.38)));
            symbol = 1.0 - smoothstep(stroke - aa, stroke, s);
            tint = vec3(0.49, 0.70, 0.95);
        } else if (kind == kCircle) {
            symbol = 1.0 - smoothstep(stroke - aa, stroke, abs(d - 0.40));
            tint = vec3(0.95, 0.42, 0.42);
        } else if (kind == kSquare) {
            vec2 q = abs(p) - vec2(0.36);
            float box = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
            symbol = 1.0 - smoothstep(stroke - aa, stroke, abs(box));
            tint = vec3(0.93, 0.55, 0.80);
        } else if (kind == kTriangle) {
            vec2 a = vec2(0.0, -0.46), b = vec2(0.42, 0.30), c = vec2(-0.42, 0.30);
            float s = min(min(Segment(p, a, b), Segment(p, b, c)), Segment(p, c, a));
            symbol = 1.0 - smoothstep(stroke - aa, stroke, s);
            tint = vec3(0.35, 0.85, 0.70);
        } else if (kind == kL1) {
            symbol = Text(p, 2, 0, 2, 0, 0, 0, 0.75);
        } else if (kind == kR1) {
            symbol = Text(p, 2, 1, 2, 0, 0, 0, 0.75);
        } else if (kind == kStart) {
            symbol = Text(p, 5, 3, 4, 5, 1, 4, 0.36);
        }
        color = mix(color, tint, symbol);
        alpha = max(alpha, symbol * 0.95);
    }
    o_color = vec4(color, alpha * v_alpha);
}
