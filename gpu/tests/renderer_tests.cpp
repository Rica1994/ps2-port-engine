/*
 * The GPU GS renderer's tests: draw into a target, read it back, check pixels
 * as the software GS (engine/host/src/gs_draw.cpp) would have them.
 */

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "ps2gpu/gs_renderer.h"

namespace {

int g_failures = 0;

/* The environment the renderer reads its paths from, set and cleared
   (Windows has neither setenv nor unsetenv). */
void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
void UnsetEnv(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

void Check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

std::uint32_t Rgba(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
    return std::uint32_t(r) | std::uint32_t(g) << 8 | std::uint32_t(b) << 16 | std::uint32_t(a) << 24;
}

/* Each channel within `slack` of the expected value: blending on the GPU
   rounds where the GS truncates. */
bool Near(std::uint32_t got, std::uint32_t want, int slack = 1) {
    for (int shift = 0; shift < 32; shift += 8) {
        const int g = int((got >> shift) & 0xFF), w = int((want >> shift) & 0xFF);
        if (g - w > slack || w - g > slack) return false;
    }
    return true;
}

ps2gpu::GsDrawState Target(std::uint32_t fbp) {
    ps2gpu::GsDrawState state;
    state.fbp = fbp;
    state.fbw = 1;
    state.width = 64;
    state.height = 64;
    state.scissor_x1 = 63;
    state.scissor_y1 = 63;
    return state;
}

/* A rectangle as two triangles, texture coordinates 0..1 across it. */
void Rect(ps2gpu::GsRenderer& r, const ps2gpu::GsDrawState& state, float x0, float y0, float x1, float y1,
          std::uint32_t rgba) {
    const std::uint8_t cr = rgba & 0xFF, cg = (rgba >> 8) & 0xFF, cb = (rgba >> 16) & 0xFF, ca = rgba >> 24;
    const ps2gpu::GsVertex v[6] = {
        {x0, y0, 0, cr, cg, cb, ca, 0, 0, 1}, {x1, y0, 0, cr, cg, cb, ca, 1, 0, 1},
        {x0, y1, 0, cr, cg, cb, ca, 0, 1, 1}, {x1, y0, 0, cr, cg, cb, ca, 1, 0, 1},
        {x1, y1, 0, cr, cg, cb, ca, 1, 1, 1}, {x0, y1, 0, cr, cg, cb, ca, 0, 1, 1}};
    r.DrawTriangles(state, v, 6);
}

std::uint32_t Pixel(ps2gpu::GsRenderer& r, std::uint32_t fbp, int x, int y) {
    std::vector<std::uint32_t> pixels;
    std::uint32_t w = 0, h = 0;
    if (!r.ReadTarget(fbp, 1, 0, &pixels, &w, &h) || pixels.size() != std::size_t(w) * h) return 0xDEADBEEF;
    return pixels[std::size_t(y) * w + x];
}

void Untextured(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x10);
    state.gouraud = false;
    // A flat right triangle covering the top-left half of 0..32.
    const ps2gpu::GsVertex flat[3] = {{0, 0, 0, 0, 0, 0, 0x80, 0, 0, 1},
                                      {32, 0, 0, 0, 0, 0, 0x80, 0, 0, 1},
                                      {0, 32, 0, 0x40, 0x80, 0xFF, 0x80, 0, 0, 1}};
    renderer.DrawTriangles(state, flat, 3);
    // Scissored away entirely.
    ps2gpu::GsDrawState clipped = state;
    clipped.scissor_x0 = 60;
    clipped.scissor_y0 = 60;
    const ps2gpu::GsVertex hidden[3] = {{0, 40, 0, 0xFF, 0, 0, 0x80, 0, 0, 1},
                                        {20, 40, 0, 0xFF, 0, 0, 0x80, 0, 0, 1},
                                        {0, 60, 0, 0xFF, 0, 0, 0x80, 0, 0, 1}};
    renderer.DrawTriangles(clipped, hidden, 3);

    std::vector<std::uint32_t> pixels;
    std::uint32_t w = 0, h = 0;
    Check(renderer.ReadTarget(0x10, 1, 0, &pixels, &w, &h) && w == 64 && h == 64, "the target reads back");
    if (pixels.size() == 64 * 64) {
        Check(pixels[5 * 64 + 5] == Rgba(0x40, 0x80, 0xFF, 0x80), "flat shading takes the last vertex's colour");
        Check(pixels[40 * 64 + 40] == 0, "outside the triangle stays as local memory was (zero)");
        Check(pixels[45 * 64 + 5] == 0, "a scissored triangle draws nothing");
        Check(pixels[0] == Rgba(0x40, 0x80, 0xFF, 0x80), "the GS samples pixel (0,0) at its corner");
    }
}

void Textured(ps2gpu::GsRenderer& renderer) {
    // 2x2: red, green / blue, white at half alpha.
    const std::uint32_t texels[4] = {Rgba(0xFF, 0, 0, 0x80), Rgba(0, 0xFF, 0, 0x80), Rgba(0, 0, 0xFF, 0x80),
                                     Rgba(0xFF, 0xFF, 0xFF, 0x40)};
    renderer.UploadTexture(7, texels, 2, 2);
    ps2gpu::GsDrawState state = Target(0x20);
    state.textured = true;
    state.texture = 7;
    state.wms = state.wmt = 1;                     // CLAMP

    Rect(renderer, state, 0, 0, 32, 32, Rgba(0x80, 0x80, 0x80, 0x80));
    Check(Pixel(renderer, 0x20, 4, 4) == Rgba(0xFF, 0, 0, 0x80), "MODULATE by 0x80 is the texel");
    Check(Pixel(renderer, 0x20, 20, 4) == Rgba(0, 0xFF, 0, 0x80), "the texel to the right");
    Check(Pixel(renderer, 0x20, 20, 20) == Rgba(0xFF, 0xFF, 0xFF, 0x40), "the texel below and right");

    Rect(renderer, state, 32, 0, 64, 32, Rgba(0x40, 0x40, 0x40, 0x40));
    Check(Pixel(renderer, 0x20, 36, 4) == Rgba(0x7F, 0, 0, 0x40), "MODULATE by 0x40 halves, truncating");

    ps2gpu::GsDrawState decal = state;
    decal.tfx = ps2gpu::TextureFunction::Decal;
    Rect(renderer, decal, 0, 32, 32, 64, Rgba(0x10, 0x10, 0x10, 0x10));
    Check(Pixel(renderer, 0x20, 4, 36) == Rgba(0xFF, 0, 0, 0x80), "DECAL ignores the vertex colour");

    ps2gpu::GsDrawState no_tcc = decal;
    no_tcc.tcc = false;
    Rect(renderer, no_tcc, 32, 32, 64, 64, Rgba(0x10, 0x10, 0x10, 0x22));
    Check(Pixel(renderer, 0x20, 36, 36) == Rgba(0xFF, 0, 0, 0x22), "without TCC the alpha is the vertex's");
}

void Blended(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x30);
    Rect(renderer, state, 0, 0, 64, 64, Rgba(0xFF, 0xFF, 0xFF, 0x80));

    // (Cs - Cd) * As + Cd: the usual transparency, As 0x40 is a quarter... of
    // 0x80 = 1.0, so half.
    ps2gpu::GsDrawState over = state;
    over.blend = true;
    over.a = ps2gpu::BlendColour::Source;
    over.b = ps2gpu::BlendColour::Dest;
    over.c = ps2gpu::BlendAlpha::Source;
    over.d = ps2gpu::BlendColour::Dest;
    Rect(renderer, over, 0, 0, 32, 32, Rgba(0, 0, 0xFF, 0x40));
    Check(Near(Pixel(renderer, 0x30, 4, 4), Rgba(0x80, 0x80, 0xFF, 0x40)),
          "(Cs - Cd) * As + Cd blends half way at As 0x40, and writes As");

    // (Cs - 0) * FIX + Cd: additive.
    Rect(renderer, state, 32, 0, 64, 32, Rgba(10, 20, 30, 0x80));
    ps2gpu::GsDrawState add = state;
    add.blend = true;
    add.a = ps2gpu::BlendColour::Source;
    add.b = ps2gpu::BlendColour::Zero;
    add.c = ps2gpu::BlendAlpha::Fix;
    add.d = ps2gpu::BlendColour::Dest;
    add.fix = 0x80;
    Rect(renderer, add, 32, 0, 64, 32, Rgba(5, 6, 7, 0x80));
    Check(Near(Pixel(renderer, 0x30, 36, 4), Rgba(15, 26, 37, 0x80)), "(Cs - 0) * FIX + Cd adds");

    // (Cd - Cs) * FIX + 0 with FIX 0x80: subtracts, clamped at zero.
    Rect(renderer, state, 0, 32, 32, 64, Rgba(100, 100, 100, 0x80));
    ps2gpu::GsDrawState sub = add;
    sub.a = ps2gpu::BlendColour::Dest;
    sub.b = ps2gpu::BlendColour::Source;
    sub.d = ps2gpu::BlendColour::Zero;
    Rect(renderer, sub, 0, 32, 32, 64, Rgba(30, 120, 0, 0x80));
    Check(Near(Pixel(renderer, 0x30, 4, 36), Rgba(70, 0, 100, 0x80)), "(Cd - Cs) * FIX subtracts, clamped");
}

void AlphaTested(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x40);
    state.alpha_test = true;
    state.atst = 5;                                 // GEQUAL
    state.aref = 0x80;
    Rect(renderer, state, 0, 0, 32, 32, Rgba(0xFF, 0, 0, 0x40));
    Rect(renderer, state, 32, 0, 64, 32, Rgba(0, 0xFF, 0, 0x80));
    Check(Pixel(renderer, 0x40, 4, 4) == 0, "alpha 0x40 fails GEQUAL 0x80: nothing is written");
    Check(Pixel(renderer, 0x40, 36, 4) == Rgba(0, 0xFF, 0, 0x80), "alpha 0x80 passes GEQUAL 0x80");

    // NEVER with AFAIL FB_ONLY: the frame buffer is written all the same
    // (Burnout 3 copies its frame this way, leaving depth alone).
    ps2gpu::GsDrawState never = state;
    never.atst = 0;
    never.afail = 1;
    Rect(renderer, never, 0, 32, 32, 64, Rgba(0, 0, 0xFF, 0x80));
    Check(Pixel(renderer, 0x40, 4, 36) == Rgba(0, 0, 0xFF, 0x80), "a failed test with AFAIL FB_ONLY writes colour");
}

/* A rectangle at one depth. */
void DepthRect(ps2gpu::GsRenderer& r, const ps2gpu::GsDrawState& state, float x0, float y0, float x1, float y1,
               std::uint32_t z, std::uint32_t rgba) {
    const std::uint8_t cr = rgba & 0xFF, cg = (rgba >> 8) & 0xFF, cb = (rgba >> 16) & 0xFF, ca = rgba >> 24;
    const ps2gpu::GsVertex v[6] = {
        {x0, y0, z, cr, cg, cb, ca, 0, 0, 1}, {x1, y0, z, cr, cg, cb, ca, 1, 0, 1},
        {x0, y1, z, cr, cg, cb, ca, 0, 1, 1}, {x1, y0, z, cr, cg, cb, ca, 1, 0, 1},
        {x1, y1, z, cr, cg, cb, ca, 1, 1, 1}, {x0, y1, z, cr, cg, cb, ca, 0, 1, 1}};
    r.DrawTriangles(state, v, 6);
}

void DepthTested(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x50);
    state.depth = true;
    state.zbp = 0x60;
    state.ztst = 2;                                 // GEQUAL: the larger Z wins
    // Near then far: the far one is hidden.
    DepthRect(renderer, state, 0, 0, 32, 32, 0x200000, Rgba(0xFF, 0, 0, 0x80));
    DepthRect(renderer, state, 0, 0, 32, 32, 0x100000, Rgba(0, 0xFF, 0, 0x80));
    Check(Pixel(renderer, 0x50, 4, 4) == Rgba(0xFF, 0, 0, 0x80), "GEQUAL keeps the nearer (larger) Z");
    // Far then near: the near one covers it.
    DepthRect(renderer, state, 32, 0, 64, 32, 0x100000, Rgba(0, 0xFF, 0, 0x80));
    DepthRect(renderer, state, 32, 0, 64, 32, 0x200000, Rgba(0xFF, 0, 0, 0x80));
    Check(Pixel(renderer, 0x50, 36, 4) == Rgba(0xFF, 0, 0, 0x80), "a nearer Z draws over a farther one");
    // Equal Z: GEQUAL passes, GREATER does not.
    DepthRect(renderer, state, 0, 32, 32, 64, 0x300000, Rgba(0, 0, 0xFF, 0x80));
    DepthRect(renderer, state, 0, 32, 32, 64, 0x300000, Rgba(0xFF, 0xFF, 0, 0x80));
    Check(Pixel(renderer, 0x50, 4, 36) == Rgba(0xFF, 0xFF, 0, 0x80), "GEQUAL passes an equal Z");
    ps2gpu::GsDrawState greater = state;
    greater.ztst = 3;
    DepthRect(renderer, greater, 0, 32, 32, 64, 0x300000, Rgba(0, 0xFF, 0xFF, 0x80));
    Check(Pixel(renderer, 0x50, 4, 36) == Rgba(0xFF, 0xFF, 0, 0x80), "GREATER fails an equal Z");
    // ZMSK: tested, not written.
    ps2gpu::GsDrawState masked = state;
    masked.depth_write = false;
    DepthRect(renderer, masked, 32, 32, 64, 64, 0x500000, Rgba(0xFF, 0, 0xFF, 0x80));
    DepthRect(renderer, state, 32, 32, 64, 64, 0x400000, Rgba(0, 0, 0xFF, 0x80));
    Check(Pixel(renderer, 0x50, 36, 36) == Rgba(0, 0, 0xFF, 0x80), "with ZMSK the Z is not written");
    // What local memory held: uploaded depth hides a farther draw.
    std::vector<std::uint32_t> z(64 * 64, 0x800000);
    renderer.UploadDepth(0x61, 64, 64, z.data());
    ps2gpu::GsDrawState uploaded = state;
    uploaded.fbp = 0x51;
    uploaded.zbp = 0x61;
    DepthRect(renderer, uploaded, 0, 0, 64, 64, 0x700000, Rgba(0xFF, 0, 0, 0x80));
    Check(Pixel(renderer, 0x51, 10, 10) == 0, "an uploaded depth buffer hides what is behind it");
    // Read back: what was uploaded, and what a draw wrote, as the GS's Z.
    DepthRect(renderer, uploaded, 0, 0, 16, 16, 0x900000, Rgba(0xFF, 0, 0, 0x80));
    std::vector<std::uint32_t> back;
    Check(renderer.ReadDepth(0x61, 64, 64, &back) && back.size() == 64 * 64, "a depth buffer reads back");
    Check(back[40 * 64 + 40] == 0x800000, "...uploaded Z as it was");
    Check(back[4 * 64 + 4] == 0x900000, "...drawn Z as the GS's Z");
    Check(!renderer.ReadDepth(0x62, 64, 64, &back), "...and a buffer never made does not");
}

void AlphaFailWrites(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x7A);
    state.depth = true;
    state.zbp = 0x7B;
    state.ztst = 2;                                 // GEQUAL
    ps2gpu::GsDrawState failing = state;
    failing.alpha_test = true;
    failing.atst = 5;                               // GEQUAL 0x80: alpha 0x40 fails
    failing.aref = 0x80;
    // FB_ONLY: the colour is written, the depth is not - a farther draw still lands.
    failing.afail = 1;
    DepthRect(renderer, failing, 0, 0, 16, 16, 0x500000, Rgba(0xFF, 0, 0, 0x40));
    Check(Pixel(renderer, 0x7A, 4, 4) == Rgba(0xFF, 0, 0, 0x40), "FB_ONLY: a failing pixel's colour is written");
    DepthRect(renderer, state, 0, 0, 16, 16, 0x100000, Rgba(0, 0xFF, 0, 0x80));
    Check(Pixel(renderer, 0x7A, 4, 4) == Rgba(0, 0xFF, 0, 0x80), "FB_ONLY: ...and its depth is not");
    // ZB_ONLY: the depth is written, the colour is not.
    failing.afail = 2;
    DepthRect(renderer, failing, 16, 0, 32, 16, 0x500000, Rgba(0xFF, 0, 0, 0x40));
    Check(Pixel(renderer, 0x7A, 20, 4) == 0, "ZB_ONLY: a failing pixel's colour is not written");
    DepthRect(renderer, state, 16, 0, 32, 16, 0x100000, Rgba(0, 0xFF, 0, 0x80));
    Check(Pixel(renderer, 0x7A, 20, 4) == 0, "ZB_ONLY: ...but its depth is, and hides a farther draw");
    // RGB_ONLY: colour without alpha (and no depth).
    DepthRect(renderer, state, 32, 0, 48, 16, 0x100000, Rgba(0, 0, 0xFF, 0x22));
    failing.afail = 3;
    DepthRect(renderer, failing, 32, 0, 48, 16, 0x500000, Rgba(0xFF, 0xFF, 0, 0x40));
    Check(Pixel(renderer, 0x7A, 36, 4) == Rgba(0xFF, 0xFF, 0, 0x22), "RGB_ONLY: colour written, alpha kept");
    DepthRect(renderer, state, 32, 0, 48, 16, 0x200000, Rgba(0, 0xFF, 0xFF, 0x80));
    Check(Pixel(renderer, 0x7A, 36, 4) == Rgba(0, 0xFF, 0xFF, 0x80), "RGB_ONLY: ...and no depth");
}

void DestinationAlphaTested(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x7C);
    Rect(renderer, state, 0, 0, 32, 64, Rgba(0x10, 0x10, 0x10, 0x80));    // alpha bit 1
    Rect(renderer, state, 32, 0, 64, 64, Rgba(0x20, 0x20, 0x20, 0x7F));   // alpha bit 0
    ps2gpu::GsDrawState date = state;
    date.date = true;
    date.datm = 1;
    Rect(renderer, date, 0, 0, 64, 32, Rgba(0xFF, 0, 0, 0x80));
    Check(Pixel(renderer, 0x7C, 4, 4) == Rgba(0xFF, 0, 0, 0x80), "DATM 1 draws where the alpha bit is 1");
    Check(Pixel(renderer, 0x7C, 36, 4) == Rgba(0x20, 0x20, 0x20, 0x7F), "...and not where it is 0");
    date.datm = 0;
    Rect(renderer, date, 0, 32, 64, 64, Rgba(0, 0xFF, 0, 0x80));
    Check(Pixel(renderer, 0x7C, 4, 36) == Rgba(0x10, 0x10, 0x10, 0x80), "DATM 0 does not draw where the bit is 1");
    Check(Pixel(renderer, 0x7C, 36, 36) == Rgba(0, 0xFF, 0, 0x80), "...and draws where it is 0");
}

void Addressed(ps2gpu::GsRenderer& renderer) {
    // 4x1: texels 0..3 with red 10, 20, 30, 40.
    const std::uint32_t texels[4] = {Rgba(10, 0, 0, 0x80), Rgba(20, 0, 0, 0x80), Rgba(30, 0, 0, 0x80),
                                     Rgba(40, 0, 0, 0x80)};
    renderer.UploadTexture(9, texels, 4, 1);
    ps2gpu::GsDrawState state = Target(0x7F);
    state.textured = true;
    state.texture = 9;
    state.tfx = ps2gpu::TextureFunction::Decal;
    // REGION_CLAMP to texels 1..2: the left quarter of the rectangle clamps up to 1.
    state.wms = 2;
    state.minu = 1;
    state.maxu = 2;
    Rect(renderer, state, 0, 0, 64, 8, 0);
    Check(Pixel(renderer, 0x7F, 4, 4) == Rgba(20, 0, 0, 0x80), "REGION_CLAMP raises texel 0 to MINU");
    Check(Pixel(renderer, 0x7F, 60, 4) == Rgba(30, 0, 0, 0x80), "...and lowers texel 3 to MAXU");
    // REGION_REPEAT: (u & MINU) | MAXU - MINU 1, MAXU 2 maps 0, 1, 2, 3 to 2, 3, 2, 3.
    state.wms = 3;
    Rect(renderer, state, 0, 8, 64, 16, 0);
    Check(Pixel(renderer, 0x7F, 4, 12) == Rgba(30, 0, 0, 0x80), "REGION_REPEAT: texel 0 reads texel 2");
    Check(Pixel(renderer, 0x7F, 20, 12) == Rgba(40, 0, 0, 0x80), "REGION_REPEAT: texel 1 reads texel 3");
    // Bilinear, clamped: between texels 1 and 2 at u = 2.0 the mix is half way, 25.
    state.wms = 1;
    state.bilinear = true;
    Rect(renderer, state, 0, 16, 64, 24, 0);
    Check(Pixel(renderer, 0x7F, 32, 20) == Rgba(25, 0, 0, 0x80), "bilinear mixes the two texels around u - 0.5");
}

void RenderToTexture(ps2gpu::GsRenderer& renderer) {
    // Draw into target 0x90, then sample it into 0x91.
    ps2gpu::GsDrawState a = Target(0x90);
    Rect(renderer, a, 0, 0, 32, 64, Rgba(0x11, 0x22, 0x33, 0x80));
    Rect(renderer, a, 32, 0, 64, 64, Rgba(0x44, 0x55, 0x66, 0x10));
    ps2gpu::GsDrawState b = Target(0x91);
    b.textured = true;
    b.tfx = ps2gpu::TextureFunction::Decal;
    b.texture_target = true;
    b.tex_fbp = 0x90;
    b.tex_fbw = 1;
    b.tex_target_psm = 0;
    b.tw = b.th = 64;
    Rect(renderer, b, 0, 0, 64, 64, 0);
    Check(Pixel(renderer, 0x91, 4, 4) == Rgba(0x11, 0x22, 0x33, 0x80), "a target sampled as a texture");
    Check(Pixel(renderer, 0x91, 40, 4) == Rgba(0x44, 0x55, 0x66, 0x10), "...all of it");
    // Read as 16-bit texels: 5 bits a channel, alpha TA1 or TA0 by the alpha bit.
    ps2gpu::GsDrawState b16 = b;
    b16.tex_format = 2;
    b16.ta0 = 0x20;
    b16.ta1 = 0x70;
    Rect(renderer, b16, 0, 0, 64, 32, 0);
    Check(Pixel(renderer, 0x91, 4, 4) == Rgba(0x10, 0x20, 0x30, 0x70), "16-bit texels: 5 bits, TA1 for the bit");
    Check(Pixel(renderer, 0x91, 40, 4) == Rgba(0x40, 0x50, 0x60, 0x20), "...TA0 without it");
    // Sampling the target being drawn: a copy is taken first.
    ps2gpu::GsDrawState self = b;
    self.fbp = 0x90;
    self.texture_target = true;
    self.tex_fbp = 0x90;
    Rect(renderer, self, 0, 0, 32, 64, 0);                        // left half reads itself: unchanged
    Check(Pixel(renderer, 0x90, 4, 4) == Rgba(0x11, 0x22, 0x33, 0x80), "a target drawing itself reads a copy");
}

// FBMSK masking part of a channel keeps exactly those bits: Burnout 3's race
// writes two bits of each half of its frame's pixels through a 16-bit view.
void PartlyMaskedChannels(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x94);
    Rect(renderer, state, 0, 0, 16, 16, 0x12345678u);
    state.fbmsk = 0x0F0F0F0Fu;
    Rect(renderer, state, 0, 0, 8, 16, 0xFFFFFFFFu);
    Check(Pixel(renderer, 0x94, 2, 2) == 0xF2F4F6F8u, "a partly masked channel keeps its masked bits");
    Check(Pixel(renderer, 0x94, 12, 2) == 0x12345678u, "...and nothing outside the draw changes");
    // A 16-bit target: FBMSK in RGBA32 bit positions all the same, as the GS
    // masks the colour before making it 16-bit (KeptBits): 0x3FFF keeps R and
    // G's low six bits; G's top two, B and A are written.
    ps2gpu::GsDrawState sixteen = Target(0x95);
    sixteen.psm = 0x02;
    Rect(renderer, sixteen, 0, 0, 16, 16, 0x00102030u);        // A 0, B 0x10, G 0x20, R 0x30
    sixteen.fbmsk = 0x3FFF;
    Rect(renderer, sixteen, 0, 0, 16, 16, 0x80F8F8F8u);
    std::vector<std::uint32_t> pixels;
    std::uint32_t w = 0, h = 0;
    renderer.ReadTarget(0x95, 1, 0x02, &pixels, &w, &h);
    Check(!pixels.empty() && pixels[0] == 0x80F8E030u, "a 16-bit target's FBMSK in RGBA32 bit positions");
}

void TargetFromMemory(ps2gpu::GsRenderer& renderer) {
    std::vector<std::uint32_t> pixels(64 * 64);
    for (std::uint32_t i = 0; i < pixels.size(); ++i) pixels[i] = Rgba(std::uint8_t(i), std::uint8_t(i >> 6), 7, 0x80);
    renderer.UploadTarget(0x92, 1, 0, pixels.data(), 64, 64);
    Check(Pixel(renderer, 0x92, 5, 3) == pixels[3 * 64 + 5], "a target loaded from memory reads back the same");
    ps2gpu::GsDrawState state = Target(0x92);
    Rect(renderer, state, 0, 0, 8, 8, Rgba(1, 2, 3, 4));
    Check(Pixel(renderer, 0x92, 2, 2) == Rgba(1, 2, 3, 4) && Pixel(renderer, 0x92, 20, 20) == pixels[20 * 64 + 20],
          "drawing over it keeps the rest");
}

void PerPixelBlending(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x7E);
    Rect(renderer, state, 0, 0, 64, 64, Rgba(0xFF, 0xFF, 0xFF, 0x80));
    ps2gpu::GsDrawState pabe = state;
    pabe.blend = true;                              // (Cs - Cd) * FIX + Cd, FIX half
    pabe.c = ps2gpu::BlendAlpha::Fix;
    pabe.fix = 0x40;
    pabe.pabe = true;
    Rect(renderer, pabe, 0, 0, 32, 32, Rgba(0, 0, 0, 0x80));
    Rect(renderer, pabe, 32, 0, 64, 32, Rgba(0, 0, 0, 0x40));
    Check(Near(Pixel(renderer, 0x7E, 4, 4), Rgba(0x80, 0x80, 0x80, 0x80)), "PABE: As 0x80 blends");
    Check(Pixel(renderer, 0x7E, 36, 4) == Rgba(0, 0, 0, 0x40), "PABE: As below 0x80 does not");
}

void Masked(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x78);
    Rect(renderer, state, 0, 0, 32, 32, Rgba(0x10, 0x20, 0x30, 0x40));
    ps2gpu::GsDrawState masked = state;
    masked.fbmsk = 0xFF000000u;                     // alpha kept, as Burnout 3 draws
    Rect(renderer, masked, 0, 0, 32, 32, Rgba(0xA0, 0xB0, 0xC0, 0x80));
    Check(Pixel(renderer, 0x78, 4, 4) == Rgba(0xA0, 0xB0, 0xC0, 0x40), "FBMSK 0xFF000000 keeps the frame's alpha");
    masked.fbmsk = 0x0000FF00u;
    Rect(renderer, masked, 0, 0, 32, 32, Rgba(1, 2, 3, 4));
    Check(Pixel(renderer, 0x78, 4, 4) == Rgba(1, 0xB0, 3, 4), "FBMSK 0x0000FF00 keeps green");
}

void Fogged(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x70);
    state.fog = true;
    state.fogcol = 0x0000FF;                        // red
    ps2gpu::GsVertex v[6];
    const float corners[6][2] = {{0, 0}, {32, 0}, {0, 32}, {32, 0}, {32, 32}, {0, 32}};
    for (int i = 0; i < 6; ++i) v[i] = {corners[i][0], corners[i][1], 0, 0xFF, 0xFF, 0xFF, 0x80, 0, 0, 1, 0x80};
    renderer.DrawTriangles(state, v, 6);
    Check(Pixel(renderer, 0x70, 4, 4) == Rgba(0xFE, 0x7F, 0x7F, 0x80),
          "fog F 0x80: (128 * C + 127 * FOGCOL) >> 8, as the software GS");
}

/* The renderer's ways of drawing the same thing, compared pixel for pixel:
   blending in the shader from a copy of the target, or reading it in place
   (VK_KHR_dynamic_rendering_local_read) - and in place as a tile-based GPU
   reads it (an iPad's, through MoltenVK: what memory held when the pass
   began or since its last barrier, which PS2GPU_LOCAL_READ_EMULATE
   reproduces on any GPU); passes ended as work comes, or deferred, the work
   going before them (PS2GPU_DEFER_PASSES); attachments loaded and stored
   whole or only as used (PS2GPU_TILE_HINTS). The same random stream through
   each: blends the blender cannot do, partly masked channels, alpha tests,
   PABE, a target sampling itself or another, through the CLUT too, depth,
   and between the draws work that touches what a pass uses or does not -
   textures uploaded again, targets written back to GPU memory and loaded
   from it, textures decoded and CLUTs loaded from it, depth buffers in and
   out. Every target, the depth buffer and GPU memory must come out the
   same: a barrier left out, or work put before a pass it should follow,
   shows here. */
void PathsAgree() {
    struct Setup {
        const char* name;
        const char* local_read;
        const char* emulate;
        const char* defer;
        const char* hints;
        const char* reorder = "0";
    };
    std::vector<Setup> setups = {{"as before", "0", "0", "0", "0"},
                                 {"tile hints", "0", "0", "0", "1"},
                                 {"deferred passes", "0", "0", "1", "1"},
                                 {"emulated tile-based local read", "0", "1", "0", "1"},
                                 {"emulated tile-based local read, deferred", "0", "1", "1", "1"},
                                 {"local read", "1", "0", "0", "1"},
                                 {"local read, deferred", "1", "0", "1", "1"},
                                 {"reordered passes", "0", "0", "1", "1", "1"},
                                 {"reordered passes, no tile hints", "0", "0", "1", "0", "1"}};
    // PS2GPU_TEST_PATHS="LEDHR,LEDHR,...": the ways compared instead (each
    // five digits: local read, its emulation, deferred passes, tile hints,
    // reordered passes), the first the reference.
    std::vector<std::string> custom;
    if (const char* paths = std::getenv("PS2GPU_TEST_PATHS")) {
        setups.clear();
        std::string list = paths;
        for (std::size_t at = 0; at + 5 <= list.size(); at += 6) custom.push_back(list.substr(at, 5));
        static const char* const kDigit[2] = {"0", "1"};
        for (const std::string& c : custom)
            setups.push_back({c.c_str(), kDigit[c[0] == '1'], kDigit[c[1] == '1'], kDigit[c[2] == '1'], kDigit[c[3] == '1'],
                              kDigit[c[4] == '1']});
    }
    std::vector<std::unique_ptr<ps2gpu::Gpu>> gpus;
    std::vector<std::unique_ptr<ps2gpu::GsRenderer>> renderers;
    std::vector<const char*> names;
    // The variables as the run had them, back after (the other tests' renderer).
    static const char* const kWays[] = {"PS2GPU_LOCAL_READ", "PS2GPU_LOCAL_READ_EMULATE", "PS2GPU_DEFER_PASSES",
                                        "PS2GPU_TILE_HINTS", "PS2GPU_REORDER"};
    std::vector<std::pair<bool, std::string>> before;
    for (const char* name : kWays) {
        const char* value = std::getenv(name);
        before.emplace_back(value != nullptr, value ? value : "");
    }
    for (const Setup& setup : setups) {
        SetEnv("PS2GPU_LOCAL_READ", setup.local_read);
        SetEnv("PS2GPU_LOCAL_READ_EMULATE", setup.emulate);
        SetEnv("PS2GPU_DEFER_PASSES", setup.defer);
        SetEnv("PS2GPU_TILE_HINTS", setup.hints);
        SetEnv("PS2GPU_REORDER", setup.reorder);
        std::string error;
        auto gpu = ps2gpu::CreateGpu(&error);
        if (!gpu) continue;
        auto renderer = ps2gpu::CreateGsRenderer(*gpu, &error);
        if (!renderer) {
            std::printf("FAIL: renderer (%s): %s\n", setup.name, error.c_str());
            ++g_failures;
            continue;
        }
        gpus.push_back(std::move(gpu));
        renderers.push_back(std::move(renderer));
        names.push_back(setup.name);
    }
    for (std::size_t i = 0; i < before.size(); ++i) {
        if (before[i].first) SetEnv(kWays[i], before[i].second.c_str());
        else UnsetEnv(kWays[i]);
    }
    if (renderers.size() < 2) return;
    // Two contents for a texture the draws sample, uploaded in turn (work a
    // pass sampling it must come before), and one they never sample.
    std::vector<std::uint32_t> texels[2] = {std::vector<std::uint32_t>(16 * 16), std::vector<std::uint32_t>(16 * 16)};
    for (std::uint32_t i = 0; i < 16 * 16; ++i) {
        texels[0][i] = Rgba(std::uint8_t(i * 7), std::uint8_t(i * 13), std::uint8_t(i * 29), std::uint8_t(0x20 + (i & 0x7F)));
        texels[1][i] = Rgba(std::uint8_t(255 - i), std::uint8_t(i * 3), std::uint8_t(i * 5), std::uint8_t(0x80 - (i & 0x3F)));
    }
    int texture_content = 0;
    for (auto& r : renderers) r->UploadTexture(0x1234, texels[0].data(), 16, 16);
    // Three 64x64 targets four pages apart (a 64x64 CT32 target takes two), a
    // 16-bit one among them, and a depth buffer.
    constexpr std::uint32_t kA = 0xA0, kB = 0xA4, kC = 0xA8, kZ = 0x80;
    for (auto& r : renderers) {
        Rect(*r, Target(kA), 0, 0, 64, 64, Rgba(10, 20, 30, 0x40));
        Rect(*r, Target(kB), 0, 0, 64, 64, Rgba(40, 50, 60, 0x90));
        ps2gpu::GsDrawState c = Target(kC);
        c.psm = 0x02;
        Rect(*r, c, 0, 0, 64, 64, Rgba(80, 90, 100, 0x80));
    }
    bool decoded = false, clut_loaded = false;
    std::mt19937 random(20261016);
    auto pick = [&](int n) { return int(random() % std::uint32_t(n)); };
    for (int op = 0; op < 2500; ++op) {
        ps2gpu::GsDrawState s = Target(kA);
        const int which = pick(10);
        if (which == 0) {
            s = Target(kB);
        } else if (which == 1) {                    // the 16-bit one (its blends are the shader's)
            s = Target(kC);
            s.psm = 0x02;
        }
        // Blending: the blender's, or one it cannot do (the shader's).
        const int blend = pick(8);
        if (blend > 0) {
            s.blend = true;
            s.a = ps2gpu::BlendColour::Source;
            s.b = ps2gpu::BlendColour::Dest;
            s.d = ps2gpu::BlendColour::Dest;
            switch (blend) {
                case 1: s.c = ps2gpu::BlendAlpha::Source; break;
                case 2: case 3: s.c = ps2gpu::BlendAlpha::Dest; break;             // Ad: in the shader
                case 4: s.c = ps2gpu::BlendAlpha::Fix; s.fix = 0x80 + 1 + pick(0x7F); break;   // over 1: shader
                case 5: s.c = ps2gpu::BlendAlpha::Fix; s.fix = pick(0x81); s.b = ps2gpu::BlendColour::Zero; break;
                case 6: s.c = ps2gpu::BlendAlpha::Fix; s.fix = 0xC0; s.a = ps2gpu::BlendColour::Dest;
                        s.b = ps2gpu::BlendColour::Source; s.d = ps2gpu::BlendColour::Zero; break;
                default: s.c = ps2gpu::BlendAlpha::Source; s.pabe = true; break;
            }
        }
        const int mask = pick(10);
        if (mask == 0) s.fbmsk = 0x0F0F0F0Fu;       // partly masked: the shader merges without logicOp
        else if (mask == 1) s.fbmsk = 0xFF000000u;
        if (pick(6) == 0) {                         // an alpha test whose failures still write
            s.alpha_test = true;
            s.atst = 5;
            s.aref = 0x60;
            s.afail = pick(2) ? 1 : 3;
        }
        if (s.fbp == kA && pick(3) == 0) {          // depth: tested, written or not
            s.depth = true;
            s.zbp = kZ;
            s.ztst = 2 + pick(2);
            s.depth_write = pick(4) != 0;
        }
        if (pick(12) == 0) {                        // the destination alpha test
            s.date = true;
            s.datm = pick(2);
        }
        s.sprite_prim = pick(2) == 0;
        const int texture = pick(12);
        if (texture < 2) {                          // a texture uploaded
            s.textured = true;
            s.texture = 0x1234;
            s.tfx = ps2gpu::TextureFunction(pick(4));
        } else if (texture == 2 && decoded) {       // one decoded from GPU memory
            s.textured = true;
            s.texture = 0x2000;
            s.tw = s.th = 64;
            s.tfx = ps2gpu::TextureFunction::Decal;
        } else if (texture == 3 && s.fbp != kB) {   // another target as the texture
            s.textured = true;
            s.texture_target = true;
            s.tex_fbp = kB;
            s.tex_fbw = 1;
            s.tw = s.th = 64;
            s.tfx = ps2gpu::TextureFunction::Modulate;
            if (clut_loaded && pick(2)) s.tex_format = 4;   // its alpha through the CLUT (PSMT8H)
        } else if (texture == 4 && s.fbp == kA) {   // its own target: a copy first
            s.textured = true;
            s.texture_target = true;
            s.tex_fbp = kA;
            s.tex_fbw = 1;
            s.tw = s.th = 64;
            s.tfx = ps2gpu::TextureFunction::Modulate;
        }
        // Small rectangles here and there - most beside the ones before, some
        // over them - or one over everything.
        float x0 = float(pick(60)), y0 = float(pick(60));
        float x1 = x0 + 1 + float(pick(12)), y1 = y0 + 1 + float(pick(12));
        if (pick(20) == 0) {
            x0 = y0 = 0;
            x1 = y1 = 64;
        }
        const std::uint32_t colour = Rgba(std::uint8_t(pick(256)), std::uint8_t(pick(256)), std::uint8_t(pick(256)),
                                          std::uint8_t(pick(256)));
        const std::uint32_t z = std::uint32_t(pick(0x1000000));
        for (auto& r : renderers) DepthRect(*r, s, x0, y0, x1, y1, z, colour);
        // Work between the draws.
        switch (pick(60)) {
            case 0:                                 // the sampled texture's other content
                texture_content ^= 1;
                for (auto& r : renderers) r->UploadTexture(0x1234, texels[texture_content].data(), 16, 16);
                break;
            case 1:                                 // a texture never drawn with
                for (auto& r : renderers) r->UploadTexture(0x1235, texels[1].data(), 16, 16);
                break;
            case 2: {                               // a target to GPU memory, and a texture decoded from it
                const std::uint32_t from = pick(2) ? kA : kB;
                ps2gpu::GsTextureDecode d;
                d.tbp = from * 32;
                d.tbw = 1;
                d.psm = 0;
                d.tw = d.th = 64;
                for (auto& r : renderers) {
                    r->TargetToMemory(from, 1, 0, 64, 64, 0);
                    r->TextureFromMemory(0x2000, d);
                }
                decoded = true;
                break;
            }
            case 3: {                               // the CLUT from GPU memory (A's pixels)
                ps2gpu::GsClutLoad load;
                load.cbp = kA * 32;
                load.cpsm = 0;
                load.eight = true;
                for (auto& r : renderers) {
                    r->TargetToMemory(kA, 1, 0, 64, 64, 0);
                    r->ClutFromMemory(load);
                }
                clut_loaded = true;
                break;
            }
            case 4: {                               // B back from GPU memory (the rows drawn into another way)
                const std::uint32_t first_row = std::uint32_t(pick(32));
                for (auto& r : renderers) r->TargetRowsFromMemory(kB, 1, 0, 64, 64, first_row, 16);
                break;
            }
            case 5: {                               // the depth buffer out to memory and back
                for (auto& r : renderers) {
                    r->DepthToMemory(kZ, 64, 64, 1, 0x30, 64, 64, 0);
                    r->DepthFromMemory(kZ, 64, 64, 1, 0x30);
                }
                break;
            }
            case 6: {                               // memory written by the CPU (blocks under no target)
                std::vector<std::uint8_t> bytes(256 * 4, std::uint8_t(pick(256)));
                for (auto& r : renderers) r->MemoryWrite(0x3000, 0x3004, bytes.data());
                break;
            }
            case 7:                                 // a read-back: everything submitted, waited for
                for (auto& r : renderers) Pixel(*r, kA, 1, 1);
                break;
            default:
                break;
        }
    }
    auto compare = [&](const char* what, const std::vector<std::uint32_t>& reference,
                       const std::vector<std::uint32_t>& got, const char* name) {
        std::size_t differ = 0, first = 0;
        for (std::size_t p = 0; p < got.size() && p < reference.size(); ++p)
            if (got[p] != reference[p] && !differ++) first = p;
        char text[240];
        std::snprintf(text, sizeof text, "%s: %s as before (%zu differ, the first at %zu)", name, what, differ, first);
        Check(differ == 0 && got.size() == reference.size() && !got.empty(), text);
    };
    const std::uint32_t targets[3][2] = {{kA, 0}, {kB, 0}, {kC, 0x02}};
    for (const auto& [fbp, psm] : targets) {
        std::vector<std::uint32_t> reference, pixels;
        std::uint32_t w = 0, h = 0;
        renderers[0]->ReadTarget(fbp, 1, psm, &reference, &w, &h);
        for (std::size_t i = 1; i < renderers.size(); ++i) {
            renderers[i]->ReadTarget(fbp, 1, psm, &pixels, &w, &h);
            char what[64];
            std::snprintf(what, sizeof what, "target %X drawn", unsigned(fbp));
            compare(what, reference, pixels, names[i]);
        }
    }
    std::vector<std::uint32_t> reference_z, z;
    renderers[0]->ReadDepth(kZ, 64, 64, &reference_z);
    std::vector<std::uint32_t> reference_memory(1u << 20), memory(1u << 20);
    renderers[0]->MemoryRead(0, 16384, reinterpret_cast<std::uint8_t*>(reference_memory.data()));
    for (std::size_t i = 1; i < renderers.size(); ++i) {
        renderers[i]->ReadDepth(kZ, 64, 64, &z);
        compare("the depth buffer", reference_z, z, names[i]);
        renderers[i]->MemoryRead(0, 16384, reinterpret_cast<std::uint8_t*>(memory.data()));
        compare("GPU memory", reference_memory, memory, names[i]);
    }
    std::printf("the renderer's paths agree: %zu ways (%s", renderers.size(), names[0]);
    for (std::size_t i = 1; i < names.size(); ++i) std::printf("; %s", names[i]);
    std::printf(")\n");
}

}  // namespace

/* ConvertImage and ConvertFromDepth against the way through memory they
   replace: an image swizzled into GPU memory and the other loaded from it -
   every pair of the formats made, and Z16 views of Z24 and Z32 depth
   buffers. Each image starts from other contents than the source, so a pixel
   the conversion missed shows. */
void Converted(ps2gpu::GsRenderer& renderer) {
    const std::uint32_t formats[6] = {0x00, 0x02, 0x0A, 0x30, 0x32, 0x3A};
    auto is16 = [](std::uint32_t f) { return f == 0x02 || f == 0x0A || f == 0x32 || f == 0x3A; };
    std::uint32_t seed = 12345;
    auto random_bytes = [&](std::vector<std::uint8_t>& bytes) {
        for (auto& b : bytes) {
            seed = seed * 1664525u + 1013904223u;
            b = std::uint8_t(seed >> 24);
        }
    };
    // 128 pixels wide (FBW 2), two page rows of 32-bit pixels: 128 blocks.
    const std::uint32_t fbw = 2, blocks = 128;
    std::uint32_t fbp = 0x100;
    auto read = [&](std::uint32_t at, std::uint32_t psm, std::vector<std::uint32_t>* pixels) {
        std::uint32_t w = 0, h = 0;
        return renderer.ReadTarget(at, fbw, psm, pixels, &w, &h);
    };
    for (std::uint32_t src : formats)
        for (std::uint32_t dst : formats) {
            if (src == dst) continue;
            fbp += 4;                           // (four pages a pair: images of their own)
            std::vector<std::uint8_t> a(blocks * 256), b(blocks * 256);
            random_bytes(a);
            random_bytes(b);
            renderer.MemoryWrite(fbp * 32, fbp * 32 + blocks, a.data());
            renderer.TargetFromMemory(fbp, fbw, src, 128, is16(src) ? 128 : 64);
            renderer.MemoryWrite(fbp * 32, fbp * 32 + blocks, b.data());
            renderer.TargetFromMemory(fbp, fbw, dst, 128, is16(dst) ? 128 : 64);
            const bool made = renderer.ConvertImage(fbp, fbw, dst, is16(dst) ? 128 : 64, fbp, fbw, src);
            std::vector<std::uint32_t> got, want;
            read(fbp, dst, &got);
            renderer.TargetToMemory(fbp, fbw, src, 128, is16(src) ? 128 : 64, 0);
            renderer.TargetFromMemory(fbp, fbw, dst, 128, is16(dst) ? 128 : 64);
            read(fbp, dst, &want);
            char what[96];
            std::snprintf(what, sizeof what, "an image in %02X made from one in %02X as through memory", unsigned(dst),
                          unsigned(src));
            Check(made && !got.empty() && got == want, what);
        }
    for (std::uint32_t zpsm : {0x31u, 0x30u})
        for (std::uint32_t view : {0x32u, 0x3Au, 0x02u, 0x00u}) {
            fbp += 4;
            std::vector<std::uint8_t> a(blocks * 256), b(blocks * 256);
            random_bytes(a);
            if (zpsm == 0x31)                   // Z24: a word's top byte is memory's, Z within 24 bits
                for (std::size_t i = 0; i < a.size(); i += 4) a[i + 2] &= 0x7F;
            random_bytes(b);
            renderer.MemoryWrite(fbp * 32, fbp * 32 + blocks, a.data());
            renderer.DepthFromMemory(fbp, 128, 64, fbw, zpsm);
            renderer.MemoryWrite(fbp * 32, fbp * 32 + blocks, b.data());
            renderer.TargetFromMemory(fbp, fbw, view, 128, is16(view) ? 128 : 64);
            renderer.MemoryWrite(fbp * 32, fbp * 32 + blocks, a.data());     // (the top bytes as the depth buffer's own)
            const bool made = renderer.ConvertFromDepth(fbp, fbw, view, is16(view) ? 128 : 64, fbp, 128, 64, fbw, zpsm);
            std::vector<std::uint32_t> got, want;
            read(fbp, view, &got);
            renderer.DepthToMemory(fbp, 128, 64, fbw, zpsm, 128, 64, 0);
            renderer.TargetFromMemory(fbp, fbw, view, 128, is16(view) ? 128 : 64);
            read(fbp, view, &want);
            char what[96];
            std::snprintf(what, sizeof what, "a %02X view made from a %02X depth buffer as through memory",
                          unsigned(view), unsigned(zpsm));
            Check(made && !got.empty() && got == want, what);
        }
}

/* Write-backs of part of an image (Columns, and a rectangle not a multiple
   of 16 wide - a 16-bit word's two pixels are x and x + 8): the pixels inside
   reach memory, the ones beside them leave memory as it was. */
void PartWrittenBack(ps2gpu::GsRenderer& renderer) {
    std::uint32_t seed = 777;
    auto random_bytes = [&](std::vector<std::uint8_t>& bytes) {
        for (auto& b : bytes) {
            seed = seed * 1664525u + 1013904223u;
            b = std::uint8_t(seed >> 24);
        }
    };
    const std::uint32_t fbw = 2, blocks = 128;
    std::uint32_t fbp = 0x180;
    for (std::uint32_t psm : {0x00u, 0x02u, 0x0Au, 0x32u})
        for (int way = 0; way < 2; ++way) {
            fbp += 4;
            const bool is16 = psm != 0x00;
            const std::uint32_t rows = is16 ? 128 : 64;
            std::vector<std::uint8_t> a(blocks * 256), b(blocks * 256);
            random_bytes(a);
            random_bytes(b);
            std::vector<std::uint32_t> pa, pb, got;
            std::uint32_t w = 0, h = 0;
            renderer.MemoryWrite(fbp * 32, fbp * 32 + blocks, b.data());
            renderer.TargetFromMemory(fbp, fbw, psm, 128, rows);
            renderer.ReadTarget(fbp, fbw, psm, &pb, &w, &h);
            renderer.MemoryWrite(fbp * 32, fbp * 32 + blocks, a.data());
            renderer.TargetFromMemory(fbp, fbw, psm, 128, rows);
            renderer.ReadTarget(fbp, fbw, psm, &pa, &w, &h);
            renderer.MemoryWrite(fbp * 32, fbp * 32 + blocks, b.data());
            // Way 0: columns [64, 128) (a page column); way 1: columns [0, 72).
            const std::uint32_t x0 = way == 0 ? 64 : 0, x1 = way == 0 ? 128 : 72;
            renderer.Columns(x0, x1);
            renderer.TargetToMemory(fbp, fbw, psm, 128, rows, 0);
            renderer.Columns(0, ~0u);
            renderer.TargetFromMemory(fbp, fbw, psm, 128, rows);
            renderer.ReadTarget(fbp, fbw, psm, &got, &w, &h);
            bool ok = got.size() == pa.size() && pa.size() == pb.size() && !got.empty();
            for (std::size_t i = 0; ok && i < got.size(); ++i) {
                const std::uint32_t x = std::uint32_t(i % w);
                ok = got[i] == ((x >= x0 && x < x1) ? pa[i] : pb[i]);
            }
            char what[96];
            std::snprintf(what, sizeof what, "%02X: columns %u-%u to memory, the rest left as it was", unsigned(psm),
                          unsigned(x0), unsigned(x1));
            Check(ok, what);
        }
}

int main() {
    // The renderers that read their targets in place, against the copies,
    // first - each with a device of its own.
    PathsAgree();
    std::string error;
    auto gpu = ps2gpu::CreateGpu(&error);
    if (!gpu) {
        std::printf("no GPU: %s (skipped)\n", error.c_str());
        return 0;
    }
    auto renderer = ps2gpu::CreateGsRenderer(*gpu, &error);
    if (!renderer) {
        std::printf("FAIL: renderer: %s\n", error.c_str());
        return 1;
    }
    Untextured(*renderer);
    Textured(*renderer);
    Blended(*renderer);
    AlphaTested(*renderer);
    DepthTested(*renderer);
    Fogged(*renderer);
    Masked(*renderer);
    AlphaFailWrites(*renderer);
    DestinationAlphaTested(*renderer);
    PerPixelBlending(*renderer);
    Addressed(*renderer);
    RenderToTexture(*renderer);
    TargetFromMemory(*renderer);
    PartlyMaskedChannels(*renderer);
    Converted(*renderer);
    PartWrittenBack(*renderer);
    if (g_failures) std::printf("gpu renderer tests: %d failed\n", g_failures);
    else std::printf("gpu renderer tests: all passed (%s)\n", gpu->Describe().c_str());
    return g_failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
