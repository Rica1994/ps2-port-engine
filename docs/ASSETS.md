# Assets: what the port keeps from run to run

A modern engine never builds an asset at the moment it is drawn. Its
pipeline imports and cooks assets ahead of time into the GPU's own formats,
keeps them in a cache keyed by what they were made from, streams them in by
need with a fallback until they arrive, and ships its pipelines compiled.
The port cannot cook the game's assets offline - they live on the player's
disc, in the PS2's formats, and reach the GS as bytes the game's own code
moves into its 4 MB every frame - but it can do the rest, at run time, the
first time each one is seen, and keep the result. That is this layer.

```
disc ─► EE RAM ─► GS memory (the game re-sends its textures every frame)
                      │
                      ▼  the bridge (gpu_bridge.cpp), a texture's blocks changed
              content id ── known? ──► the asset cache (asset_cache.cpp)
                │                         │ in memory: upload the cooked image
                │                         │ in the store: read it (worker), swap in when there
                │                         │ unknown: cook it (worker) - and meanwhile
                ▼                         ▼
          decoded on the GPU as before (decode.comp): the fallback, never a stall
```

## The pieces

**Content ids** (`gpu_bridge.cpp`, `UseAsset`, `KnowAsset`). A texture is
known by what its image is made of - the texels it decodes from, its CLUT,
TEXA, CLAMP's region modes, its mipmap levels - not by where the game put it.
The id is a hash (XXH3) of the image as the GPU renderer decodes it, which the
GS model makes on the CPU for it (`Gs::DecodeImage`: the software GS's own
texel reads; the host tests check it against the GPU's decode for every
format and for mipmap levels). That decode happens once per content per run:
the bridge remembers the id by what the image is made from wherever it is
(`SourceKey`: the texels and CLUT hashed as a texture pack names them, which
is cheap, and CLAMP) - the race streams its textures through a few places of
GS memory, so by address the same texture looked new every time.

**What is worth knowing.** Only the textures a game keeps using are made
known, so that neither a movie nor anything else seen once costs a CPU
decode or a record: a texture decoded from the CPU's memory and still drawn,
unchanged, a second later (most of a race's); or a content the game sends
again and again (its streamed textures) once it has come back six times over
at least a second. A movie's 64x64 tiles go through a few places in turn,
each shown for two frames; an attract loop shows them again a minute later -
before these rules one run's movies made 120,000 records and 1.9 GB of store.

**Cooking** (`texture_codec.cpp`). A worker thread (an iPad's efficiency
cores) encodes the image in the device's format - ASTC 4x4 with Arm's
astcenc on Apple's GPUs, BC7 (bc7enc) or RGBA on a PC - decodes it back and
judges it (`codec::Quality`): colour PSNR at least 38 dB, alpha within 8 (of
the GS's 128), no texel's alpha going from none to some or back (a cut-out's
edge), and in a texture that reaches 0x80 none crossing it (PABE, the alpha
test's usual reference). What fails is remembered as "drawn decoded" and
never tried again. Textures under 256 texels are not cooked.

The GS's alpha is the hard part: opaque is 0x80 or 0x7F, the middle of a
byte, which block compression does not keep exactly. Before encoding, the
alpha is stretched so that the texture's largest is 255 (`Cooked::alpha_top`;
the renderer scales it back as it reads: `gs.frag`'s `Cooked`): opaque and
none at the ends of the range, where ASTC and BC7 are exact. Even so, about
a third of a race's textures pass the judgement (their alpha is mostly
smooth ramps, which 8 bits a texel cannot keep within 8): the rest are drawn
decoded, as before. Measured on Burnout 3's scripted race (lavapipe, ASTC
simulated): a third of the texture cache's bytes cooked (216 of 635 MB as
RGBA), 17% of the race's textured batches drawn cooked (83% in the menus);
the worst of 29 pictures 42.9 dB from the uncooked build, no difference to
be seen. With `rgba8` 92% of the batches are drawn from the cache and every
picture is byte-identical.

**The store** (`asset_cache.cpp`). One file a format in the cache folder
(`platform::CacheDirectory()/assets`: the app's Library/Caches on iOS,
`cache` beside the executable elsewhere): a header naming the format and
the encoder's version and effort, then records of id, format, size and
checksummed data, appended as they are cooked. Opened at the start, its
records' headers are read into an index; a record a run left half-written
is cut off. A store of another encoder or effort is put aside (`.old`),
not read. It stops growing at a budget (768 MB on a tablet, 2 GB on a PC).

**Streaming** (`asset_cache.cpp`'s loader). A texture in the store but not
in memory is read by a worker; until then it is drawn decoded, and the
bridge swaps the cooked image in at the next draw that uses it
(`AwaitAsset`). Cooked images stay in memory up to a budget (96 MB on a
tablet), least recently used out.

**Prefetch** (`.order` beside the store). The order the run first asked for
each texture is kept; the next run, a texture asked for brings the 48 that
followed it last time from the store into memory - a race's textures are
read while the race starts, as an engine streams a level's bundle.

**Pipelines** (`gs_renderer.cpp`, `SetPipelineCacheDir`; docs/GPU.md). The
driver's pipeline cache and the keys a run drew with are kept beside the PC
settings (on a tablet, in the caches folder with the store:
`gpu/gpu_pipeline_cache.bin`, `gpu/gpu_pipelines.txt`); a worker thread makes
the listed ones at the next start, before any batch asks (an engine's PSO
precaching), and a new kind of draw is drawn with the whole shader until its
own pipeline is made. On MoltenVK each is a Metal compile of tens of
milliseconds that would otherwise land on the GS thread mid-race.

## Switches

| | |
|---|---|
| `PS2HOST_ASSETS` / PC SETTINGS `texture_cooking` | `astc`, `bc7`, `rgba8` or `off`. Default: `astc` on a tablet, `off` on a PC (a desktop GPU reads RGBA as fast; nothing to gain). Scripted runs (the virtual clock) leave it off unless `PS2HOST_ASSETS` asks. |
| `PS2HOST_ASSET_SIMULATE=1` | a format the GPU lacks drawn decoded back to RGBA: what ASTC looks like, on any machine |
| `PS2HOST_ASSET_SYNC=1` | cooked and read on the GS thread: the same pictures every time (scripted tests) |
| `PS2HOST_ASSET_EFFORT` | the encoder's effort, 0-100 (10: astcenc's "fast") |
| `PS2HOST_CACHE_DIR` | another cache folder |

`rgba8` cooks without loss: a scripted run with it (`PS2HOST_ASSETS=rgba8
PS2HOST_ASSET_SYNC=1`) must give the same pictures as one without - the test
of the ids, the CPU decode and the swap.

## In the log

Every five seconds in the `[perf]` block (a tablet's, or `PS2HOST_PERF=1`):

```
[perf]   pipelines: 212 of 230 listed built ahead of need (190 of them used), 3 asked for before they were ready; ...
[perf]   assets (ASTC 4x4): 120 cooked (30.1 MB to 7.5 MB, 4 not close enough, 900 ms), ... read from the store ...
```

## What this does not do

It does not change what the game draws or when: the game still decides what
is in GS memory each frame, and every texture the GPU drew itself (render
targets read as textures) is decoded from GPU memory as before. Geometry is
not touched - VU1 still makes it every frame on the CPU. Those would be the
next layers: VU1's programs as GPU vertex shaders over meshes kept on the
GPU, then RenderWare's PS2 driver replaced.
