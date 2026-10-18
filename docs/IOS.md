# A port on an iPad

The same port, built for iPadOS: the game's translated code and the engine's
host compiled for arm64, the GS drawn with Vulkan through **MoltenVK** (Vulkan
on Metal, carried inside the app), SDL3 for the window, the controller and the
sound. Nothing is emulated and nothing is compiled at run time, so Apple's rule
against JIT compilers does not apply.

What it takes:

* A Mac with **Xcode** (App Store) and **Homebrew** (https://brew.sh).
* An Apple ID signed in to Xcode (Xcode > Settings > Accounts). A free one
  works: the app then has to be installed again from Xcode every 7 days. A paid
  developer account ($99 a year) makes it a year.
* The iPad in **Developer Mode** (Settings > Privacy & Security > Developer
  Mode; it appears after the iPad was first connected to Xcode), iPadOS 15 or
  later.
* Optionally a **game controller** paired with the iPad (DualShock 4,
  DualSense, Xbox, MFi). Without one the game plays with on-screen controls
  (below).
* Your own disc image, as for the PC port.

## 1. The game's code, from your disc

On the Mac, in the game repository (cloned with `--recursive`), as for the PC
port - this needs only Python 3:

```sh
python3 engine/tools/isotool.py extract "Burnout 3 - Takedown.iso" SLUS_210.50 -o games/burnout3/extracted/
python3 engine/tools/analyze.py --game games/burnout3
python3 engine/tools/derive.py  --game games/burnout3 --emit --csv
python3 engine/tools/lift.py    --game games/burnout3 --structure --emit --csv
python3 engine/tools/port.py    --game games/burnout3 generate
```

The game's code in C++ stays on your machine (`games/burnout3/src/`,
`build/`), never committed.

Optional, for speed: the VU microprograms compiled ahead of time and the code
the game copies at run time (README's "Building it"). Both are found by
running the port; on a Mac `port.py --game games/burnout3` builds and runs it
natively (Clang, the same as on Linux) for that, after which `generate` puts
what it found into the iPad's build.

## 2. The Xcode project

```sh
engine/tools/ios_setup.sh --game games/burnout3 --team <TEAM ID> --bundle-id com.<you>.burnout3
```

`--team` is your team's ID: Xcode > Settings > Accounts > your Apple ID > the
team (or leave it out and choose the team in Xcode's *Signing & Capabilities*
afterwards). The bundle ID is any reverse-domain name of your own.

The script installs CMake, Ninja and glslang with Homebrew if they are
missing, downloads MoltenVK's iOS framework from its releases, and writes
`build-ios/burnout3/burnout3_port.xcodeproj`.

## 3. On the iPad

1. Open the project in Xcode, choose the **burnout3** scheme and your iPad
   (connected by cable, or on the same network once paired), and use
   **Product > Run**. For full speed, first set the scheme's run settings,
   once (**Product > Scheme > Edit Scheme > Run**; the scheme is Xcode's own,
   `cmake` leaves it alone): **Info > Build Configuration** to **Release**
   and **Debug executable** off; **Diagnostics > Metal API Validation** off;
   **Options > GPU Frame Capture** to **Disabled**. The two Metal layers
   otherwise sit under every Metal call MoltenVK makes - thousands a
   picture, on the thread that draws - and a run with them on measures them
   as much as the game (a third of that thread's time in a race).
2. The first time, the iPad refuses an app from an unknown developer: allow it
   in **Settings > General > VPN & Device Management**.
3. The app says it needs the disc image. In the **Files** app, go to **On My
   iPad > Burnout 3** and copy your `.iso` there (from AirDrop, iCloud Drive,
   a USB drive...). Start the game again: it finds the image, checks it is the
   right disc and remembers it.

The saves (`memcard/`), the settings (`pc_settings.ini`) and the log of the
last run (`burnout3.log`, for a bug report) are in the same folder.

### A benchmark from Xcode

With `PS2HOST_SCRIPT` in the scheme's **Arguments > Environment Variables**
(the line of a `port/runs/*.args` file, say `race.args`'s), the app plays
that scripted run by itself on the iPad's own disc image - the same frames
every time - and quits at its end; its log is its own (`burnout3-<date>-
<time>.log`, beside `burnout3.log`), with a `[perf]` block every five seconds:
pictures a second, each thread's time by section, the GPU's busy time, the
renderer's counts, memory and the thermal state, and every ten seconds where
the GS thread was found (`[sample]`). Untick the variable to play again.
Other variables read there: `PS2GPU_REORDER=0` (the renderer's reordered
passes, on by default on an iPad: `gpu/src/gs_renderer.cpp`), `PS2GPU_TILE=1`
(the tile-based GPU's other ways, `gpu/src/gpu_internal.h`), MoltenVK's own (`MVK_CONFIG_*`), and the host's
(`docs/PORTING.md`). Each of the GPU work cuts of 18 Oct has its own, to
put the old way back for a comparison: `PS2HOST_CONVERT=0` (images made
from a twin over the same memory without going through it -
`host/src/gpu_bridge.cpp`'s ConvertFromTwin), `PS2HOST_Z16_VIEW=0` (a
depth buffer read as PSMZ16 decoded as a texture, not sampled as a view of
it), `PS2HOST_SAMPLE_PAGES=1` (a sampled image's pages brought up to date,
not their whole page rows: fewer pixels, more passes), `PS2GPU_DISCARD_ALWAYS=1` (every pipeline able to
discard, as before: `gs.frag`'s kMayDiscard - pipelines that cannot discard
let Apple's GPU skip hidden pixels; on by default only with MoltenVK).

### Reports and VU programs, without carrying files (a PC and a Mac)

For development with a PC (the game's code generated there, from the disc)
and a Mac (the iPad builds), the app sends its own reports over the local
network, and the Mac takes the PC's results with one command:

1. **On the PC**, in the game repository, leave this running:
   ```
   python engine/tools/ipad_reports.py --game games/burnout3 --out ../ipad-reports
   ```
   It prints the PC's address (allow Python through the firewall, private
   networks). Each run's log arrives in `--out` as it runs (every fifteen
   seconds, when the app leaves the front, and at its end), named after the
   run (`burnout3-20261016-151448-scripted.log`). `vu_programs_new.txt` - the
   VU programs the app started, in normal play and in scripted runs - is
   merged into `build/vu_programs.txt` and compiled ahead at once.
2. **On the Mac**, before each build, in the game repository:
   ```sh
   engine/tools/ipad_update.sh --pc http://<the PC's address>:8765    # the first time; then without --pc
   ```
   It pulls the sources, takes the compiled VU programs from the PC when they
   changed, and brings the Xcode project up to date - with the PC's address
   built into the app (CMake's `PS2_REPORT_URL`). Then Run.
3. The first run asks whether the app may use the local network: allow it.
   `PS2HOST_REPORT=http://host:port` in the scheme overrides the address, `0`
   turns the reports off.

The logs and the programs come from your disc: they stay on your machines.

### Speed, pictures and frame skipping

The corner shows **% SPEED** - the game's steps against a PS2's 60 a second:
below 100% the whole game runs in slow motion - and **PICTURES/S**, the steps
the GPU drew in. A port can leave a step's drawing out while the iPad lags
behind (Burnout 3: `port/frame_skip.cpp`, on by default on a tablet;
`frame_skip` in pc_settings.ini, `PS2HOST_FRAMESKIP=0/1`, or `--frameskip=0/1`
in a `PS2HOST_SCRIPT` line): the game keeps its speed and shows fewer
pictures. Since 18 Oct a picture every N steps, N from what a picture costs
the GPU (its timestamps): one in three for ~43 ms, one in two under ~29 ms;
the log's `[frame skip]` line every ten seconds says which and the cost
(`PS2HOST_FRAMESKIP_MODE=lag` for the lag alone, as before). `--skip-cycle=0xT:0xA,0xB,...` (and `--skip-seconds=N`) in a script
line measures what each drawing function costs (`host/src/host.cpp`).

### Leaving a run

**Select + Start** held two seconds quits the app (its log sent first). The
Home button or the app switcher pauses the game and stops the controller's
rumble until the app is in front again. Xcode's Stop button ends it at once.

## Touch controls

A DualShock 2 drawn over the picture, while no controller is in use:

* **the left stick**: put a finger down in the lower left of the screen and
  move it - where it came down is the stick's centre;
* **Cross, Circle, Square, Triangle** in the lower right (a finger slides from
  one to the next, as a thumb rolls on a pad), **L1** and **R1** above the
  stick and the buttons, **START** at the top;
* anywhere else a tap is the PC port's mouse click, in the menus.

Pressing a button on a controller puts them away; the next touch brings them
back. The controller's buttons are the PC port's (its README: Controls), and
PC SETTINGS > CONTROLS works the same - leave MODERN CONTROLS off to play by
touch: they accelerate with R2, which the touch controls do not have (Cross
accelerates without them).

## How the port differs on an iPad

* **Threads**: the guest threads run on fibers switched by the engine itself
  (`host/src/platform.cpp`, arm64 assembly) instead of Windows' fibers.
* **The VUs** run the plain C++ model (`ps2rt/vu_math.h` without SSE); AVX2 is
  the PC build's.
* **Vulkan**: MoltenVK is loaded from the app's `Frameworks/`
  (`gpu/src/vulkan_setup.cpp`), as a portability implementation; dynamic
  rendering comes from Vulkan 1.3 or its extension. Logic operations (FBMSK's
  partly masked channels) are not in Metal: the renderer's fallback is used.
  BC-compressed texture packs need an M1 or later iPad.
* **Background**: in the background the game waits (iPadOS stops an app's GPU
  work there) and goes on when it is in front again.
* **Sound** plays with the ringer switch on silent, as games do.

## Performance

The PC build expects a desktop processor with AVX2. On an iPad the EE's code
runs as compiled arm64; the VUs without SIMD are slower than on a PC. An iPad
with an M-series chip or an A14 or later should hold the game's 60 frames a
second; older A-series iPads (A12, A13) may slow down in busy races. Keep the
resolution at ORIGINAL or 2X and anti-aliasing off in PC SETTINGS on those.
