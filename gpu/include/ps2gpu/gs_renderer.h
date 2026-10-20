/*
 * The GS on the GPU (engine/docs/GPU.md, stage two): primitives the host's GS
 * model has assembled, drawn with Vulkan into render targets that stand for
 * regions of the GS's local memory.
 *
 * Drawn so far: triangles - flat or Gouraud, scissored - textured or not, with
 * the GS's four texture functions, its alpha test, its blending and its depth
 * test, into 32-bit targets at the native resolution, which read back for
 * testing against the software GS (engine/host/src/gs_draw.cpp). Textures arrive decoded: the host
 * reads them out of local memory (swizzle, CLUT, TEXA) into RGBA and names
 * them with a key of its choosing. Depth is a 32-bit float buffer holding
 * Z / 2^32. The destination alpha test, fog and FBMSK are next.
 *
 * Colours are in the GS's terms throughout: 0-255 per channel, with 0x80 as
 * 1.0 where the GS multiplies (vertex colour against texture, alpha as a blend
 * factor).
 *
 * Gpu is the device without a window (tests, off-screen work); a Display's
 * device can stand in for it later.
 */

#ifndef PS2GPU_GS_RENDERER_H
#define PS2GPU_GS_RENDERER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ps2gpu {

/* Told each time the CPU waited for the GPU (a command buffer's slot to come
   back): why, and from when to when (steady clock, nanoseconds) - for the
   host's profiler (PS2HOST_PROFILE). On the waiting thread. */
using GpuWaitObserver = void (*)(const char* why, std::uint64_t from_ns, std::uint64_t to_ns);
void SetGpuWaitObserver(GpuWaitObserver observer);

/* A Vulkan instance and device with one graphics queue (Vulkan 1.3: dynamic
   rendering). Opaque; created headless. */
class Gpu {
public:
    virtual ~Gpu() = default;
    virtual std::string Describe() const = 0;
};
std::unique_ptr<Gpu> CreateGpu(std::string* error);

/* A vertex as the GS has it after XYOFFSET: position in pixels (the GS samples
   at whole pixel coordinates), colour 0-255, and the texture coordinate as
   S/Q, T/Q over the texture's size (with UV, the host passes u / 16 / width,
   v / 16 / height and q = 1). */
struct GsVertex {
    float x, y;
    std::uint32_t z;
    std::uint8_t r, g, b, a;
    float s, t, q;
    std::uint8_t fog = 0xFF;            /* F: 255 keeps the colour, 0 is all FOGCOL */
};

/* TEX0.TFX: how the texel and the vertex colour combine. */
enum class TextureFunction : std::uint32_t { Modulate = 0, Decal = 1, Highlight = 2, Highlight2 = 3 };

/* ALPHA's operands: Cv = (A - B) * C / 128 + D. */
enum class BlendColour : std::uint32_t { Source = 0, Dest = 1, Zero = 2 };
enum class BlendAlpha : std::uint32_t { Source = 0, Dest = 1, Fix = 2 };

/* The state a batch of triangles is drawn with. */
struct GsDrawState {
    std::uint32_t fbp = 0, fbw = 1, psm = 0;            /* FRAME: the target */
    std::uint32_t fbmsk = 0;                             /* FRAME.FBMSK: bits not written */
    std::uint32_t width = 640, height = 448;            /* how much of it to make */
    std::int32_t scissor_x0 = 0, scissor_y0 = 0, scissor_x1 = 639, scissor_y1 = 447;   /* inclusive */
    bool gouraud = true;                                 /* else the last vertex's colour */
    /* A sprite (PRIM 6): on a target drawn larger, sampled where the GS
       samples - at its pixel's centre - not between (a 2D blit's strips met
       at seams of texels from beyond them). */
    bool sprite = false;
    /* A sprite (PRIM 6) at any internal resolution: what blends by the
       destination's alpha in the shader (ShaderBlend). */
    bool sprite_prim = false;

    /* Texturing (PRIM.TME): a texture uploaded under this key. */
    bool textured = false;
    std::uint64_t texture = 0;
    TextureFunction tfx = TextureFunction::Modulate;
    bool tcc = true;                                     /* TEX0.TCC: the texel's alpha counts */
    bool bilinear = false;                               /* TEX1: the four texels around */
    /* TEX1's mipmapping, per pixel (0: none - `bilinear` throughout): bit 1
       set, MXL in bits 2-4, MMIN 5-7, MMAG 8, LCM 9, L 10-11, K 16-27 (as
       TEX1 has them). A texture decoded with MXL + 1 levels holds levels 1
       and on below level 0 (GsTextureDecode). */
    std::uint32_t mip = 0;
    /* CLAMP: WMS/WMT 0 REPEAT, 1 CLAMP, 2 REGION_CLAMP (to MIN..MAX), 3
       REGION_REPEAT (MIN is a mask, MAX ORed in). */
    std::uint32_t wms = 0, wmt = 0, minu = 0, maxu = 0, minv = 0, maxv = 0;
    std::uint32_t tw = 0, th = 0;                        /* TEX0's size (0: the texture's own) */
    /* ...or a target sampled as the texture (render to texture): TEX0's TBP is
       the target's FBP * 32 and TBW its FBW. 24- and 16-bit texture formats
       read it as the GS would - TEXA's alpha, 16-bit colour to 5 bits. */
    bool texture_target = false;
    std::uint32_t tex_fbp = 0, tex_fbw = 0, tex_target_psm = 0;
    std::uint32_t tex_format = 0;                        /* 0 32-bit, 1 24-bit, 2 16-bit, through the CLUT 4 PSMT8H, 5 PSMT4HL, 6 PSMT4HH (CSA in bits 24-28) */
    std::uint32_t ta0 = 0, ta1 = 0x80;                   /* TEXA */
    bool aem = false;

    /* The alpha test (TEST.ATE, ATST, AREF, AFAIL). What a failing pixel
       writes: AFAIL 0 KEEP nothing, 1 FB_ONLY its colour (not depth), 2
       ZB_ONLY its depth (not colour), 3 RGB_ONLY its colour without alpha
       (not depth) - drawn as a second pass over the failing pixels. */
    bool alpha_test = false;
    std::uint32_t atst = 1, aref = 0, afail = 0;         /* ATST 1 is ALWAYS */

    /* The destination alpha test (TEST.DATE, DATM): a pixel is drawn only
       where the target's alpha bit (its alpha's top bit) equals DATM. */
    bool date = false;
    std::uint32_t datm = 0;

    /* Blending (PRIM.ABE) with ALPHA's operands and FIX. */
    bool blend = false;
    BlendColour a = BlendColour::Source, b = BlendColour::Dest, d = BlendColour::Dest;
    BlendAlpha c = BlendAlpha::Source;
    std::uint32_t fix = 0x80;
    bool pabe = false;                                   /* PABE: blend only where As >= 0x80 */

    /* Fog (PRIM.FGE): C = (F * C + (255 - F) * FOGCOL) / 256, colour only. */
    bool fog = false;
    std::uint32_t fogcol = 0;                            /* FOGCOL: R, G, B in bytes 0-2 */

    /* Depth (ZBUF, TEST.ZTE/ZTST): a depth buffer named by ZBP, as large as
       the target. ZTST 0 NEVER, 1 ALWAYS, 2 GEQUAL, 3 GREATER - the GS keeps
       the larger Z. Without `depth` nothing is tested or written. */
    bool depth = false;
    std::uint32_t zbp = 0, ztst = 1;
    bool depth_write = true;                             /* !ZBUF.ZMSK */
};

/* DepthIntoView's draw (haze.comp's push constants): the target CT32 at
   bp (blocks) / bw, its 16-bit view's PSM; the target's pixels looked at;
   the view's pixels drawn; a view pixel's texel (x + du, y + dv), wrapping
   (wrap bit 0 S, bit 1 T) or clamped in tw x th; the Z16 texture at tbp
   (blocks) / tbw / tpsm over the depth buffer at zbp (blocks) / zbw / zpsm
   (zwidth x zheight); TEXA; the 16 bits each half keeps (FBMSK's). */
struct DepthIntoViewParams {
    std::uint32_t bp = 0, bw = 1, view_psm = 0x02;
    std::uint32_t x0 = 0, y0 = 0, cols = 0, rows = 0;
    std::uint32_t vx0 = 0, vy0 = 0, vx1 = 0, vy1 = 0;
    std::int32_t du = 0, dv = 0;
    std::uint32_t tw = 1, th = 1, wrap = 0;
    std::uint32_t tbp = 0, tbw = 1, tpsm = 0x32;
    std::uint32_t zbp = 0, zbw = 1, zpsm = 0x31, zwidth = 0, zheight = 0;
    std::uint32_t ta0 = 0, ta1 = 0x80, aem = 0;
    std::uint32_t keep = 0;
};

/* A texture in GS local memory, and how its texels expand: TEX0's TBP0 (in
   blocks), TBW, PSM and size, CLAMP's modes and region, TEXA, CSA, and the
   CLUT as the GS model holds it (256 RGBA entries; palettized formats). */
struct GsTextureDecode {
    std::uint32_t tbp = 0, tbw = 1, psm = 0, tw = 1, th = 1;
    std::uint32_t wms = 0, wmt = 0, minu = 0, maxu = 0, minv = 0, maxv = 0;
    std::uint32_t texa_lo = 0;      // TA0, and AEM at bit 15
    std::uint32_t ta1 = 0, csa = 0;
    /* Mipmap levels (1-7): level n, TW-n by TH-n from level_tbp[n] and
       level_tbw[n] (MIPTBP1/2), goes below level n-1 in the same image, at
       its left edge - the image tw wide, as tall as all of them. */
    std::uint32_t levels = 1;
    std::uint32_t level_tbp[7] = {}, level_tbw[7] = {};
    /* The CLUT as the GS model holds it, or null: the CLUT the GPU holds
       (ClutFromMemory, ClutRows) at this point of what is recorded. */
    const std::uint32_t* clut = nullptr;
};

/* A CLUT load (TEX0's CLD) from GS local memory, as the host's GS model does
   one: CBP (blocks), CPSM, CSA, an 8-bit texture's 256 entries or a 4-bit
   one's 16, 16-bit entries expanded by TEXA (TA0 and AEM at bit 15, TA1). */
struct GsClutLoad {
    std::uint32_t cbp = 0, cpsm = 0, csa = 0;
    bool eight = true;
    std::uint32_t texa_lo = 0, ta1 = 0;
};

class GsRenderer {
public:
    virtual ~GsRenderer() = default;
    /* The internal resolution: targets and depth buffers drawn `scale` times
       as large each way (1-4); textures, memory and everything read back
       stay in GS pixels. Changing it drops every target and depth buffer:
       what the GPU drew must be in local memory first; they come back from
       it. */
    virtual std::uint32_t Scale() const { return 1; }
    virtual void SetScale(std::uint32_t scale) { (void)scale; }
    /* A texture, RGBA in memory (alpha as the GS has it, 0x80 is 1.0), rows
       top-down. Replaces what the key named. */
    virtual void UploadTexture(std::uint64_t key, const std::uint32_t* pixels, std::uint32_t width,
                               std::uint32_t height) = 0;
    /* A texture pack's replacement: an image of its own under `id`, shared by
       every texture it replaces. `format` 0 RGBA8 (alpha as the GS's: 0x80
       is 1.0), 1-4 BC1, BC2, BC3, BC7 as a DDS stores them; `levels` (mipmap
       levels) largest first, one after another in `data`. False if the GPU
       cannot take it (block compression it lacks). Replacements the draws
       have not used for a while go when they take more than a budget of
       GPU memory: HasReplacement says whether one is still there. */
    virtual bool UploadReplacement(std::uint64_t id, std::uint32_t format, std::uint32_t width, std::uint32_t height,
                                   std::uint32_t levels, const std::uint8_t* data, std::size_t bytes) {
        (void)id, (void)format, (void)width, (void)height, (void)levels, (void)data, (void)bytes;
        return false;
    }
    virtual bool HasReplacement(std::uint64_t id) const { (void)id; return false; }
    /* The texture under `key` drawn as replacement `id`: finer than the GS's
       TW x TH (the draw's tw, th), sampled in its own texels at the same
       places. Until the key is uploaded or decoded again. */
    virtual void UseReplacement(std::uint64_t key, std::uint64_t id) { (void)key, (void)id; }
    /* Whether block-compressed replacements can be uploaded as they are. */
    virtual bool BlockCompression() const { return false; }
    /* The smallest piece of GPU memory a texture's image takes (images take
       a power of two of bytes, no less): what a cache counting its memory
       counts a small one as. */
    virtual std::uint64_t ImagePiece() const { return 64 << 10; }
    /* A cooked texture (the asset cache's): the image TextureFromMemory
       would decode under `key` - the same size, its mipmap levels below
       level 0 the same way - in `format`: 0 RGBA8, 4 BC7, 5 ASTC 4x4
       (ps2host::codec::Format). Drawn exactly as a decoded texture is; false
       if the GPU cannot take the format (CookedFormat). `flags` bits 0-7: its
       alpha stored stretched so that this value is 255 (block compression
       keeps the ends of the range exact), scaled back as it is read; 0 for
       alpha as it is. */
    virtual bool UploadCooked(std::uint64_t key, std::uint32_t format, std::uint32_t width, std::uint32_t height,
                              const std::uint8_t* data, std::size_t bytes, std::uint32_t flags = 0) {
        (void)key, (void)format, (void)width, (void)height, (void)data, (void)bytes, (void)flags;
        return false;
    }
    virtual bool CookedFormat(std::uint32_t format) const { (void)format; return false; }
    /* A depth buffer's contents, one Z per pixel, rows top-down: what local
       memory holds there (the GS's Z, unsigned). */
    virtual void UploadDepth(std::uint32_t zbp, std::uint32_t width, std::uint32_t height,
                             const std::uint32_t* z) = 0;
    /* A target's contents from local memory, RGBA rows top-down (made, at
       this size, if there was none). */
    virtual void UploadTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, const std::uint32_t* pixels,
                              std::uint32_t width, std::uint32_t height) = 0;
    /* Forgets a texture (after what is recorded has used it). */
    virtual void ReleaseTexture(std::uint64_t key) = 0;
    /* Whether a target exists for (FBP, FBW, PSM) - what a draw can sample. */
    virtual bool HasTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm) const = 0;
    /* Triangles, three vertices each, in GS order. */
    virtual void DrawTriangles(const GsDrawState& state, const GsVertex* vertices, std::size_t count) = 0;
    /* Runs what has been drawn. */
    virtual void Flush() = 0;
    /* Hands what has been recorded to the GPU without waiting for it (once a
       frame, say), so that the GPU draws while the CPU goes on rather than
       all at once when something next waits. */
    virtual void Kick() {}
    /* Whether the GPU has finished all the work handed to it (Kick, and any
       before) - asked without waiting; what has finished is retired. A
       port's frame skipping draws again once its last drawn frame is done. */
    virtual bool GpuCaughtUp() { return true; }
    /* GS local memory on the GPU: 4 MB, as the GS lays it out, which targets
       and depth buffers are swizzled into and out of without the CPU. Blocks
       are 256 bytes; [begin, end) in blocks. Nothing waits but MemoryRead. */
    /* The CPU's copy of blocks, into the GPU's. */
    virtual void MemoryWrite(std::uint32_t begin, std::uint32_t end, const std::uint8_t* bytes) = 0;
    /* The GPU's copy of blocks, into `bytes` (runs what is recorded, waits). */
    virtual void MemoryRead(std::uint32_t begin, std::uint32_t end, std::uint8_t* bytes) = 0;
    /* Words of the GPU's copy patched under masks - `count` of (word index,
       value, mask), the mask's bits set taken from the value - in turn with
       what is recorded, nothing waiting: part of a block written over what
       the GPU drew there, without bringing it back. */
    virtual void MemoryPatch(const std::uint32_t* words, std::size_t count) { (void)words, (void)count; }
    /* A target's rows [first_row, rows), `cols` wide, into GPU memory at FBP
       as its PSM lays them out (CT32, CT24, CT16, CT16S); false if there is
       none. */
    virtual bool TargetToMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t cols,
                                std::uint32_t rows, std::uint32_t first_row = 0) = 0;
    /* A target, made at this size if there is none, loaded from GPU memory. */
    virtual void TargetFromMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t width,
                                  std::uint32_t height) = 0;
    /* The same for the depth buffer at ZBP of this size, laid out by ZPSM
       (Z32, Z24, Z16, Z16S) FBW pages wide. */
    virtual bool DepthToMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                               std::uint32_t zpsm, std::uint32_t cols, std::uint32_t rows,
                               std::uint32_t first_row = 0) = 0;
    virtual void DepthFromMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                                 std::uint32_t zpsm) = 0;
    /* A texture decoded from GPU memory, as the host's software GS decodes
       one (GsTextureDecode); replaces what the key named. */
    virtual void TextureFromMemory(std::uint64_t key, const GsTextureDecode& decode) = 0;
    /* The GPU's CLUT - the 256 RGBA entries textures decoded on the GPU with
       no CLUT of their own read - loaded from GPU memory, in order with what
       is recorded: a CLUT the GPU drew never waits to come back. */
    virtual void ClutFromMemory(const GsClutLoad& load) = 0;
    /* Rows of the GPU's CLUT (16 entries each; bit r of `rows` for row r)
       from the host's copy, `clut` being all 256 entries. */
    virtual void ClutRows(const std::uint32_t* clut, std::uint32_t rows) = 0;
    /* A depth buffer's Z (the GS's, unsigned), rows top-down; false if there
       is none of that size. */
    virtual bool ReadDepth(std::uint32_t zbp, std::uint32_t width, std::uint32_t height,
                           std::vector<std::uint32_t>* z) = 0;
    /* A texture's texels, RGBA, rows top-down; false if there is none. */
    virtual bool ReadTexture(std::uint64_t key, std::vector<std::uint32_t>* pixels, std::uint32_t* width,
                             std::uint32_t* height) = 0;
    /* A target to show as it is on the GPU (Display::PresentImage): what has
       been drawn is handed to the GPU first, and the target's image (its
       64-bit handle), its layout (a VkImageLayout) and size are given; false
       if there is none or it has not been drawn. */
    /* For the TargetFromMemory / TargetRowsFromMemory calls that follow: an
       image drawn larger (the internal resolution) keeps its own colour
       where memory differs from it in alpha alone - another view's write
       into its alpha (Burnout 3's depth into the frame's alpha, through a
       16-bit view) leaves the picture as fine as it was drawn, rather than
       as memory holds it (a GS pixel a square). Where the colour differs,
       memory's, as ever. */
    virtual void ReloadKeepsColour(bool keep) { (void)keep; }
    /* For the ToMemory and RowsFromMemory calls that follow: columns
       [first, end) of the rectangle alone (0, ~0u: all of it, as before) -
       a few pages of an image moved, not its whole rows. A renderer that
       cannot leaves it out: the whole rows, as before. */
    virtual void Columns(std::uint32_t first, std::uint32_t end) { (void)first, (void)end; }
    /* The target (fbp, fbw, psm), its rows [0, rows), made from target (src_fbp, src_fbw, src_psm)
       over the same memory, as memory would have it, without memory: each
       of its pixels from the source's pixels that lay its bytes out (CT32,
       CT16, CT16S, at the PS2's resolution). False if the renderer cannot -
       the caller goes through memory, as before. Pixels whose bytes the
       source does not hold are left as they are. */
    /* Whether ConvertImage / ConvertFromDepth can be done at all (their
       pipelines made): what the threaded renderer predicts their answers by. */
    virtual bool Converts() const { return false; }
    virtual bool ConvertsDepth() const { return false; }
    virtual bool ConvertImage(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t rows,
                              std::uint32_t src_fbp, std::uint32_t src_fbw, std::uint32_t src_psm) {
        (void)fbp, (void)fbw, (void)psm, (void)rows, (void)src_fbp, (void)src_fbw, (void)src_psm;
        return false;
    }
    /* The same from the depth buffer at ZBP of this size (Z24, Z32, laid out
       FBW pages wide): a Z16 view of it, say. A Z24 word's top byte, not the
       depth buffer's, as GPU memory has it. */
    virtual bool ConvertFromDepth(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t rows,
                                  std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t zfbw,
                                  std::uint32_t zpsm) {
        (void)fbp, (void)fbw, (void)psm, (void)rows, (void)zbp, (void)width, (void)height, (void)zfbw, (void)zpsm;
        return false;
    }
    /* A depth buffer's Z16 texels drawn into a 16-bit view of a 32-bit
       target in one pass over the target (DepthIntoViewParams): what
       nearest-sampled, decal sprites of a PSMZ16 texture over the depth
       buffer's memory, a whole texel a pixel, would draw into the view - the
       view and the Z16 texture never made. False if the renderer cannot (the
       caller draws them as before); DrawsDepthIntoView whether it can at all. */
    virtual bool DrawsDepthIntoView() const { return false; }
    virtual bool DepthIntoView(const DepthIntoViewParams& params) {
        (void)params;
        return false;
    }
    /* TargetFromMemory / DepthFromMemory for rows [first_row, first_row +
       rows) alone (the rest of the image as it is). */
    virtual void TargetRowsFromMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t width,
                                      std::uint32_t height, std::uint32_t first_row, std::uint32_t rows) {
        (void)first_row, (void)rows;
        TargetFromMemory(fbp, fbw, psm, width, height);
    }
    virtual void DepthRowsFromMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                                     std::uint32_t zpsm, std::uint32_t first_row, std::uint32_t rows) {
        (void)first_row, (void)rows;
        DepthFromMemory(zbp, width, height, fbw, zpsm);
    }
    virtual bool DisplayTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint64_t* image,
                               int* layout, std::uint32_t* width, std::uint32_t* height) {
        (void)fbp, (void)fbw, (void)psm, (void)image, (void)layout, (void)width, (void)height;
        return false;
    }
    /* A target's pixels, RGBA in memory, rows top-down; false if there is none. */
    virtual bool ReadTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm,
                            std::vector<std::uint32_t>* pixels, std::uint32_t* width,
                            std::uint32_t* height) = 0;
    /* The same at the internal resolution, every pixel (a capture). */
    virtual bool ReadTargetFull(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm,
                                std::vector<std::uint32_t>* pixels, std::uint32_t* width, std::uint32_t* height) {
        return ReadTarget(fbp, fbw, psm, pixels, width, height);
    }
};
std::unique_ptr<GsRenderer> CreateGsRenderer(Gpu& gpu, std::string* error);
/* Where the renderer keeps what it learns between runs: a folder (the PC
   settings'), for its pipeline cache (gpu_pipeline_cache.bin) and the
   pipelines the game has drawn with (gpu_pipelines.txt) - made again at the
   start, from the cache, so none is built in the middle of a race. Set
   before CreateGsRenderer; "" (a scripted run) keeps nothing. */
void SetPipelineCacheDir(const std::string& dir);
/* Pipelines a game is known to draw with (keys as gpu_pipelines.txt has
   them), made by the renderer's worker from the start as the last runs'
   are - a player's first run then makes none in the middle of a race. Keys
   of an older engine are only made in vain. Set before CreateGsRenderer. */
void SetPipelineHints(const std::uint32_t* keys, std::size_t count);
/* `inner` on a thread of its own (threaded_renderer.cpp): what only hands
   work over is queued and made there in order; what answers or reads waits
   for the queue to empty first. One caller thread. */
std::unique_ptr<GsRenderer> MakeThreadedGsRenderer(std::unique_ptr<GsRenderer> inner);

/* How often the CPU has waited for the GPU to finish what was submitted, and
   for how long in all: "N submissions, S s waiting". */
std::string SubmitStats();
/* What went into the command buffers since the last call: barriers, layout
   transitions, compute dispatches, draws, rendering passes, feedback copies. */
std::string RendererCounters();
/* Since the last call, lines each starting with `prefix`: what ended the
   rendering passes (another target, a copy of the target, the DATE setup,
   other work - by kind, and how much of that work touched nothing the pass
   used), why batches read their target in the shader, and the passes' area. */
std::string RendererBreakdown(const char* prefix);
/* The GPU's time on the frames' command buffers, in all (nanoseconds; 0
   where the GPU keeps no time). */
std::uint64_t GpuBusyNanoseconds();
/* What the GPU's driver holds for the app, in all heaps (bytes;
   VK_EXT_memory_budget's heap usage - MoltenVK's is Metal's allocated size).
   0 where the device cannot say. */
std::uint64_t GpuMemoryInUse();
/* PS2HOST_GPU_PROFILE: the GPU's time since the last call, by what it did. */
std::string GpuProfileReport();
/* The profile's GPU time by pass (what each drew into), the ten largest. */
std::string GpuPassReport();
/* Whether the GPU's work from the next command buffer on is profiled (where
   a profile pool was made: PS2HOST_GPU_PROFILE, or a tablet's default). */
void SetGpuProfiling(bool on);
bool GpuProfiling();
/* A trial on the device itself (PS2GPU_TRIAL; a tablet's default): one of
   the renderer's ways, on or off from the next batch on, so that the
   bridge's report can switch it interval by interval and sum the GPU's time
   a picture each way - a measurement on the GPU that matters, from an
   ordinary game. Its name ("" for none: the way stays on). */
const char* TrialName();
void SetTrialOn(bool on);
bool TrialOn();

}  // namespace ps2gpu

#endif  /* PS2GPU_GS_RENDERER_H */
