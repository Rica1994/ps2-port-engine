#ifndef PS2HOST_ASSET_CACHE_H
#define PS2HOST_ASSET_CACHE_H

/* The asset cache: textures cooked once and kept, as a modern engine's
   derived-data cache keeps its cooked assets.

   A texture is known by its content (asset ids: gpu_bridge.cpp's
   TextureAsset - a hash of exactly the texels its image is decoded from,
   its CLUT, and how they expand), not by where the game put it in GS memory.
   The first time a content is seen the bridge decodes it as before and hands
   its RGBA to Cook(); a worker thread encodes it in the device's format
   (codec::Format: ASTC on Apple's GPUs, BC7 or RGBA on a PC), checks the
   result looks the same (codec::Quality), and appends it to the store - one
   file, records of id, format, size and data, checksummed, read back at the
   next start. From then on, in this run and every later one, the texture is
   uploaded cooked instead of decoded: a quarter of the memory and of the
   GPU's reading.

   Loading is asynchronous too: Find() on a texture in the store but not in
   memory queues a read and answers Loading; the bridge draws it decoded
   meanwhile and swaps the cooked image in when it is there (like a streaming
   system's fallback, then the full asset). */

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "ps2host/texture_codec.h"

namespace ps2host::assets {

struct Options {
    std::filesystem::path directory;            // the store's folder (platform::CacheDirectory()/"assets")
    codec::Format format = codec::kRgba8;       // what textures are cooked to
    int effort = 10;                            // the encoder's (codec::Encode)
    std::uint64_t disk_budget = 1ull << 30;     // the store stops growing past this
    std::uint64_t memory_budget = 128ull << 20; // cooked images kept in memory, least recently used out
    bool simulate = false;                      // a lossy format decoded back to RGBA before drawing
    bool cook = true;                           // false: only what the store has is used
    std::uint32_t min_texels = 256;             // smaller textures are not worth a record
    bool synchronous = false;                   // cooked and read on the caller's thread: a scripted
                                                // run's pictures the same every time (PS2HOST_ASSET_SYNC)
};

/* A cooked image as the renderer takes it (GsRenderer::UploadCooked). */
struct Cooked {
    codec::Format format = codec::kRgba8;
    std::uint32_t width = 0, height = 0;
    /* Its alpha stored stretched so that this, its largest, is 255 (0: as it
       is): the GS's opaque is 0x80 or 0x7F, and a block-compressed format
       keeps the ends of its range exactly, not the middle. The renderer
       scales it back as it reads. */
    std::uint32_t alpha_top = 0;
    std::vector<std::uint8_t> data;
};

enum class State {
    Unknown,            // never seen: Cook() it
    Loading,            // in the store, being read
    Cooking,            // queued or being encoded
    Ready,              // here (Find's result)
    Rejected,           // not worth it or not close enough: drawn decoded
};

/* The store opened and its workers started; Enabled() after. */
void Open(const Options& options);
bool Enabled();
const Options& Settings();

/* The cooked image of `id` when it is in memory (state Ready). Otherwise
   what is happening to it - a store read queued now if it is there. */
std::shared_ptr<const Cooked> Find(std::uint64_t id, State* state);

/* `rgba` (width x height, as the renderer's decode lays it out) cooked under
   `id`, on the worker. Ignored when the id is known already. */
void Cook(std::uint64_t id, std::vector<std::uint32_t> rgba, std::uint32_t width, std::uint32_t height);

/* Since the last call: "assets: N cooked (A MB to B MB, X rejected, Y ms),
   N read from the store, N used cooked; the store N textures, N MB". */
std::string Report();

/* What is kept beside the store written now: the order textures were asked
   for, which the next run prefetches by (cheap when nothing is new). */
void Save();

/* The workers stopped, what they had written kept (at exit; tests). */
void Close();

}  // namespace ps2host::assets

#endif  /* PS2HOST_ASSET_CACHE_H */
