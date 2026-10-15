#include "touch_controls.h"

#include <algorithm>
#include <cmath>

namespace ps2gpu {

namespace {

/* DualShock 2 inputs (display.h's PadState: libpad's order). */
constexpr int kInputStart = 3, kInputL1 = 10, kInputR1 = 11, kInputTriangle = 12, kInputCircle = 13,
              kInputCross = 14, kInputSquare = 15;

int InputOf(int kind) {
    switch (kind) {
        case TouchControls::kCross: return kInputCross;
        case TouchControls::kCircle: return kInputCircle;
        case TouchControls::kSquare: return kInputSquare;
        case TouchControls::kTriangle: return kInputTriangle;
        case TouchControls::kL1: return kInputL1;
        case TouchControls::kR1: return kInputR1;
        case TouchControls::kStart: return kInputStart;
        default: return -1;
    }
}

constexpr float kHitScale = 1.35f;      // a button takes a finger a little outside what is drawn

}  // namespace

/* Sizes in the screen's shorter side (u), so they are the same under a thumb
   on any screen. */
std::vector<TouchControls::Place> TouchControls::Layout(float w, float h) {
    const float u = std::min(w, h);
    const float cx = w - 0.23f * u, cy = h - 0.25f * u, d = 0.11f * u, r = 0.064f * u;
    return {
        {kStickBase, 0.21f * u, h - 0.23f * u, 0.13f * u},
        {kTriangle, cx, cy - d, r},
        {kCircle, cx + d, cy, r},
        {kCross, cx, cy + d, r},
        {kSquare, cx - d, cy, r},
        {kR1, w - 0.12f * u, h - 0.55f * u, 0.058f * u},
        {kL1, 0.12f * u, h - 0.55f * u, 0.058f * u},
        {kStart, 0.5f * w, 0.065f * u, 0.045f * u},
    };
}

bool TouchControls::InStickZone(float x, float y, float w, float h) {
    return x < 0.40f * w && y > 0.45f * h;
}

int TouchControls::ButtonAt(float x, float y, float w, float h) const {
    int best = -1;
    float best_d = 1e30f;
    for (const Place& p : Layout(w, h)) {
        if (p.kind == kStickBase) continue;
        const float d = std::hypot(x - p.x, y - p.y);
        if (d <= p.r * kHitScale && d < best_d) {
            best = p.kind;
            best_d = d;
        }
    }
    return best;
}

bool TouchControls::Down(std::int64_t finger, float x, float y, float w, float h) {
    std::lock_guard<std::mutex> hold(lock_);
    visible_ = true;
    width_ = w;
    height_ = h;
    Finger f{finger, -1, x, y, x, y};
    const int button = ButtonAt(x, y, w, h);
    if (button >= 0) f.control = button;
    else if (InStickZone(x, y, w, h)) f.control = kStickBase;
    fingers_.erase(std::remove_if(fingers_.begin(), fingers_.end(), [&](const Finger& g) { return g.id == finger; }),
                   fingers_.end());
    fingers_.push_back(f);
    return f.control >= 0;
}

void TouchControls::Move(std::int64_t finger, float x, float y, float w, float h) {
    std::lock_guard<std::mutex> hold(lock_);
    width_ = w;
    height_ = h;
    for (Finger& f : fingers_) {
        if (f.id != finger) continue;
        f.x = x;
        f.y = y;
        // A finger on a button slides to the next one, as a thumb rolls on a pad.
        if (f.control >= 0 && f.control != kStickBase) {
            const int button = ButtonAt(x, y, w, h);
            if (button >= 0) f.control = button;
        }
    }
}

void TouchControls::Up(std::int64_t finger) {
    std::lock_guard<std::mutex> hold(lock_);
    fingers_.erase(std::remove_if(fingers_.begin(), fingers_.end(), [&](const Finger& g) { return g.id == finger; }),
                   fingers_.end());
}

void TouchControls::ControllerUsed() {
    std::lock_guard<std::mutex> hold(lock_);
    visible_ = false;
}

bool TouchControls::Visible() const {
    std::lock_guard<std::mutex> hold(lock_);
    return visible_;
}

bool TouchControls::Merge(std::uint16_t* buttons, std::uint8_t sticks[4]) const {
    std::lock_guard<std::mutex> hold(lock_);
    bool any = false;
    const float u = std::min(width_, height_);
    for (const Finger& f : fingers_) {
        if (f.control < 0) continue;
        any = true;
        if (f.control == kStickBase) {
            // Full deflection at the base's radius; a small dead zone.
            const float reach = 0.13f * u;
            float dx = (f.x - f.x0) / reach, dy = (f.y - f.y0) / reach;
            const float length = std::hypot(dx, dy);
            if (length > 1.0f) dx /= length, dy /= length;
            if (length < 0.12f) dx = dy = 0.0f;
            sticks[2] = std::uint8_t(std::clamp(128.0f + dx * 127.5f, 0.0f, 255.0f));
            sticks[3] = std::uint8_t(std::clamp(128.0f + dy * 127.5f, 0.0f, 255.0f));
            continue;
        }
        const int input = InputOf(f.control);
        if (input >= 0) *buttons |= std::uint16_t(1u << input);
    }
    return any;
}

std::vector<TouchControls::Shape> TouchControls::Shapes(float w, float h) const {
    std::lock_guard<std::mutex> hold(lock_);
    std::vector<Shape> out;
    if (!visible_) return out;
    const float u = std::min(w, h);
    for (const Place& p : Layout(w, h)) {
        bool held = false;
        for (const Finger& f : fingers_) held = held || f.control == int(p.kind);
        if (p.kind == kStickBase) {
            // The stick drawn where its finger came down, its knob under the finger.
            const Finger* stick = nullptr;
            for (const Finger& f : fingers_)
                if (f.control == kStickBase) stick = &f;
            const float bx = stick ? stick->x0 * w / std::max(width_, 1.0f) : p.x;
            const float by = stick ? stick->y0 * h / std::max(height_, 1.0f) : p.y;
            out.push_back({bx, by, p.r, 0.0f, float(kStickBase), held ? 1.0f : 0.0f, 0, 0});
            float kx = bx, ky = by;
            if (stick) {
                const float sx = stick->x * w / std::max(width_, 1.0f), sy = stick->y * h / std::max(height_, 1.0f);
                float dx = sx - bx, dy = sy - by;
                const float length = std::hypot(dx, dy);
                if (length > p.r) dx *= p.r / length, dy *= p.r / length;
                kx += dx;
                ky += dy;
            }
            out.push_back({kx, ky, 0.055f * u, 0.0f, float(kStickKnob), held ? 1.0f : 0.0f, 0, 0});
            continue;
        }
        out.push_back({p.x, p.y, p.r, 0.0f, float(p.kind), held ? 1.0f : 0.0f, 0, 0});
    }
    for (Shape& s : out) s.alpha = s.held > 0.5f ? 0.85f : 0.55f;
    return out;
}

}  // namespace ps2gpu
