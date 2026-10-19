#include "ps2host/texture_codec.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>

#include "bcdec.h"                      // (its implementation is texture_pack.cpp's)
#include "bc7enc.h"
#include "astcenc.h"

namespace ps2host::codec {

const char* Name(Format format) {
    switch (format) {
        case kRgba8: return "RGBA8";
        case kBc7: return "BC7";
        case kAstc4x4: return "ASTC 4x4";
    }
    return "?";
}

bool Parse(const std::string& text, Format* format) {
    std::string t;
    for (char ch : text) t += char(ch >= 'A' && ch <= 'Z' ? ch - 'A' + 'a' : ch);
    if (t == "rgba8" || t == "rgba") *format = kRgba8;
    else if (t == "bc7") *format = kBc7;
    else if (t == "astc" || t == "astc4x4") *format = kAstc4x4;
    else return false;
    return true;
}

std::size_t Size(Format format, std::uint32_t width, std::uint32_t height) {
    if (format == kRgba8) return std::size_t(width) * height * 4;
    return std::size_t((width + 3) / 4) * ((height + 3) / 4) * 16;     // both 16 bytes a 4x4 block
}

namespace {

// ------------------------------------------------------------------ BC7 (bc7enc)

void Bc7Init() {
    static std::once_flag once;
    std::call_once(once, [] { bc7enc_compress_block_init(); });
}

bool EncodeBc7(const std::uint32_t* rgba, std::uint32_t w, std::uint32_t h, int effort,
               std::vector<std::uint8_t>* out) {
    Bc7Init();
    bc7enc_compress_block_params params;
    bc7enc_compress_block_params_init(&params);
    // Plain RGB error (the texels are data the GS combines, not a photo), and
    // the effort as bc7enc's levels.
    bc7enc_compress_block_params_init_linear_weights(&params);
    params.m_uber_level = effort >= 90 ? 4 : effort >= 60 ? 2 : effort >= 30 ? 1 : 0;
    params.m_max_partitions_mode = effort >= 30 ? BC7ENC_MAX_PARTITIONS1 : 16;
    const std::uint32_t bw = (w + 3) / 4, bh = (h + 3) / 4;
    out->assign(std::size_t(bw) * bh * 16, 0);
    std::uint32_t block[16];
    for (std::uint32_t by = 0; by < bh; ++by)
        for (std::uint32_t bx = 0; bx < bw; ++bx) {
            // An edge's block repeats its last row and column.
            for (std::uint32_t y = 0; y < 4; ++y)
                for (std::uint32_t x = 0; x < 4; ++x) {
                    const std::uint32_t sx = std::min(w - 1, bx * 4 + x), sy = std::min(h - 1, by * 4 + y);
                    block[y * 4 + x] = rgba[std::size_t(sy) * w + sx];
                }
            bc7enc_compress_block(out->data() + (std::size_t(by) * bw + bx) * 16, block, &params);
        }
    return true;
}

bool DecodeBc7(const std::uint8_t* data, std::size_t bytes, std::uint32_t w, std::uint32_t h,
               std::vector<std::uint32_t>* rgba) {
    const std::uint32_t bw = (w + 3) / 4, bh = (h + 3) / 4;
    if (bytes < std::size_t(bw) * bh * 16) return false;
    rgba->assign(std::size_t(w) * h, 0);
    std::uint32_t block[16];
    for (std::uint32_t by = 0; by < bh; ++by)
        for (std::uint32_t bx = 0; bx < bw; ++bx) {
            bcdec_bc7(data + (std::size_t(by) * bw + bx) * 16, block, 16);
            for (std::uint32_t y = 0; y < 4 && by * 4 + y < h; ++y)
                for (std::uint32_t x = 0; x < 4 && bx * 4 + x < w; ++x)
                    (*rgba)[std::size_t(by * 4 + y) * w + bx * 4 + x] = block[y * 4 + x];
        }
    return true;
}

// ------------------------------------------------------------------ ASTC (astcenc)

/* An astcenc context a thread, by effort: making one is slow (tables). */
struct AstcContext {
    astcenc_context* context = nullptr;
    int effort = -1;
    ~AstcContext() {
        if (context) astcenc_context_free(context);
    }
};

astcenc_context* Astc(int effort) {
    thread_local AstcContext held;
    if (held.context && held.effort == effort) return held.context;
    if (held.context) astcenc_context_free(held.context);
    held.context = nullptr;
    astcenc_config config;
    // Decoded as the GPU decodes LDR blocks to 8 bits a channel.
    if (astcenc_config_init(ASTCENC_PRF_LDR, 4, 4, 1, float(std::clamp(effort, 0, 100)), ASTCENC_FLG_USE_DECODE_UNORM8,
                            &config) != ASTCENC_SUCCESS)
        return nullptr;
    // Alpha decides the alpha test and PABE: it weighs more than a colour.
    config.cw_a_weight = 5.0f;
    if (astcenc_context_alloc(&config, 1, &held.context) != ASTCENC_SUCCESS) {
        held.context = nullptr;
        return nullptr;
    }
    held.effort = effort;
    return held.context;
}

bool EncodeAstc(const std::uint32_t* rgba, std::uint32_t w, std::uint32_t h, int effort,
                std::vector<std::uint8_t>* out) {
    astcenc_context* context = Astc(effort);
    if (!context) return false;
    void* slices[1] = {const_cast<std::uint32_t*>(rgba)};
    astcenc_image image{w, h, 1, ASTCENC_TYPE_U8, slices};
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    out->assign(Size(kAstc4x4, w, h), 0);
    const bool ok = astcenc_compress_image(context, &image, &swizzle, out->data(), out->size(), 0) == ASTCENC_SUCCESS;
    astcenc_compress_reset(context);
    return ok;
}

bool DecodeAstc(const std::uint8_t* data, std::size_t bytes, std::uint32_t w, std::uint32_t h,
                std::vector<std::uint32_t>* rgba) {
    if (bytes < Size(kAstc4x4, w, h)) return false;
    astcenc_context* context = Astc(10);
    if (!context) return false;
    rgba->assign(std::size_t(w) * h, 0);
    void* slices[1] = {rgba->data()};
    astcenc_image image{w, h, 1, ASTCENC_TYPE_U8, slices};
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    const bool ok = astcenc_decompress_image(context, data, bytes, &image, &swizzle, 0) == ASTCENC_SUCCESS;
    astcenc_decompress_reset(context);
    return ok;
}

}  // namespace

bool Encode(Format format, const std::uint32_t* rgba, std::uint32_t width, std::uint32_t height, int effort,
            std::vector<std::uint8_t>* out) {
    if (!width || !height) return false;
    switch (format) {
        case kRgba8:
            out->resize(Size(kRgba8, width, height));
            std::memcpy(out->data(), rgba, out->size());
            return true;
        case kBc7: return EncodeBc7(rgba, width, height, effort, out);
        case kAstc4x4: return EncodeAstc(rgba, width, height, effort, out);
    }
    return false;
}

bool Decode(Format format, const std::uint8_t* data, std::size_t bytes, std::uint32_t width, std::uint32_t height,
            std::vector<std::uint32_t>* rgba) {
    if (!width || !height) return false;
    switch (format) {
        case kRgba8:
            if (bytes < Size(kRgba8, width, height)) return false;
            rgba->resize(std::size_t(width) * height);
            std::memcpy(rgba->data(), data, rgba->size() * 4);
            return true;
        case kBc7: return DecodeBc7(data, bytes, width, height, rgba);
        case kAstc4x4: return DecodeAstc(data, bytes, width, height, rgba);
    }
    return false;
}

Quality Compare(const std::uint32_t* original, const std::uint32_t* encoded, std::size_t count) {
    Quality q;
    double squares = 0;
    // 0x80 is a threshold only for a texture that reaches it: one whose
    // opaque texels are 0x7F (most of Burnout 3's) draws the same at 0x80.
    std::uint32_t top = 0;
    for (std::size_t i = 0; i < count; ++i) top = std::max(top, original[i] >> 24);
    const bool threshold = top >= 0x80;
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t a = original[i], b = encoded[i];
        for (int c = 0; c < 3; ++c) {
            const int d = int((a >> (8 * c)) & 0xFF) - int((b >> (8 * c)) & 0xFF);
            squares += double(d * d);
        }
        const std::uint32_t aa = a >> 24, ba = b >> 24;
        const std::uint32_t error = aa > ba ? aa - ba : ba - aa;
        q.max_alpha_error = std::max(q.max_alpha_error, error);
        if (((aa == 0) != (ba == 0)) || (threshold && (aa >= 0x80) != (ba >= 0x80))) ++q.alpha_breaks;
    }
    const double mse = count ? squares / double(count * 3) : 0;
    q.psnr = mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;
    return q;
}

}  // namespace ps2host::codec
