#ifndef PS2HOST_TEXTURE_CODEC_H
#define PS2HOST_TEXTURE_CODEC_H

/* Texture formats the asset cache cooks to (asset_cache.h), and their
   encoders: what a modern engine's texture compressor is, for the GS's
   textures once decoded to RGBA.

   Texels are RGBA, R in the low byte, alpha as the GS has it (0x80 is 1.0):
   the encoders treat the four channels as plain data and keep alpha as it is
   given. The numbers are the renderer's (ps2gpu::GsRenderer::UploadCooked). */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ps2host::codec {

enum Format : std::uint32_t {
    kRgba8 = 0,         // as decoded: no loss, the reference (and a test of the path)
    kBc7 = 4,           // 8 bits a texel; desktop GPUs (textureCompressionBC)
    kAstc4x4 = 5,       // 8 bits a texel; Apple's and Android's GPUs (textureCompressionASTC_LDR)
};

const char* Name(Format format);
/* A format by its name as settings give it ("rgba8", "bc7", "astc"); false if none. */
bool Parse(const std::string& text, Format* format);

/* The bytes an image of the format takes (one level). */
std::size_t Size(Format format, std::uint32_t width, std::uint32_t height);

/* `rgba` (width x height) encoded into `out`. `effort` 0-100 (astcenc's
   scale: 10 fast, 60 medium); BC7 maps it to its own levels. False when the
   format cannot be made here. Thread-safe: each thread keeps its own
   encoder state. */
bool Encode(Format format, const std::uint32_t* rgba, std::uint32_t width, std::uint32_t height, int effort,
            std::vector<std::uint8_t>* out);
/* Back to RGBA, as the GPU will read it (to judge an encoding, and to draw it
   where the GPU lacks the format: PS2HOST_ASSET_SIMULATE). */
bool Decode(Format format, const std::uint8_t* data, std::size_t bytes, std::uint32_t width, std::uint32_t height,
            std::vector<std::uint32_t>* rgba);

/* How far an encoding is from what it encoded: the colour's PSNR (dB, over
   R, G and B), the largest alpha error, and how many texels changed alpha in
   a way the GS would draw differently - across 0x80 in a texture that
   reaches it (PABE, and an alpha test's usual reference: an opaque texel no
   longer passing it), or from none to some or back (a cut-out's edge
   moving). */
struct Quality {
    double psnr = 0;
    std::uint32_t max_alpha_error = 0;
    std::uint64_t alpha_breaks = 0;
    /* Whether it is close enough to draw instead: the picture looks the same
       (alpha a few 128ths off is no visible change in a blend). */
    bool Acceptable() const { return psnr >= 38.0 && max_alpha_error <= 8 && alpha_breaks == 0; }
};
Quality Compare(const std::uint32_t* original, const std::uint32_t* encoded, std::size_t count);

}  // namespace ps2host::codec

#endif  /* PS2HOST_TEXTURE_CODEC_H */
