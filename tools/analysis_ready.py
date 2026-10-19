#!/usr/bin/env python3
"""The game's executable and its analysis (and the overlays' table), made where missing.

A build that lifts code from the player's executable at build time (Burnout
3's mods: port/tools/lift_mods.py) needs the executable (extracted/) and the
analysis database (build/<name>.json) - which a machine given only the
generated sources (the Mac of docs/IOS.md) does not have. This takes the
executable from the disc and runs the analysis (about a minute), once:

    python engine/tools/analysis_ready.py --game games/burnout3 [--iso <disc>]

The disc: --iso, else game.json's "iso", else the one .iso beside the game
repository or in it. Nothing is done when both are there already.

Then the overlays' dispatch table, when the generated folder (the PC's)
was made before the engine linked one, and the functions the game's list has
lifted that sources lifted by an older engine lack (lift_missing).
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp import game as game_config  # noqa: E402


def find_disc(game, asked: str | None) -> str | None:
    if asked:
        return asked if os.path.isfile(asked) else None
    named = game.path(game.iso) if game.iso else ""
    if named and os.path.isfile(named):
        return named
    # The repository's root is two above the game's directory (games/<name>).
    root = os.path.abspath(os.path.join(game.directory, "..", ".."))
    for folder in (root, os.path.dirname(root)):
        discs = sorted(glob.glob(os.path.join(folder, "*.iso")) + glob.glob(os.path.join(folder, "*.ISO")))
        if len(discs) == 1:
            return discs[0]
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", required=True, help="the game's directory (with game.json)")
    parser.add_argument("--iso", help="the disc image")
    args = parser.parse_args(argv)
    game = game_config.load(args.game)
    if getattr(game, "base", None) is not None:
        print("an overlay's analysis is made from its game's; nothing done here")
        return 0
    if not os.path.isfile(game.elf_path):
        disc = find_disc(game, args.iso)
        if not disc:
            print("the game's executable (%s) is not there and no disc was found: pass --iso <the disc image>"
                  % game.executable, file=sys.stderr)
            return 1
        print("the executable %s from %s" % (game.executable, os.path.basename(disc)))
        subprocess.run([sys.executable, os.path.join(HERE, "isotool.py"), "extract", disc, game.executable,
                        "-o", game.path(game.extracted), "-q"], check=True)
    if not os.path.isfile(game.database_path):
        print("the analysis (%s), about a minute..." % os.path.relpath(game.database_path, game.directory))
        subprocess.run([sys.executable, os.path.join(HERE, "analyze.py"), "--game", game.directory, "-q"],
                       check=True)
    # The overlays' table (dispatch_overlays.cpp), which an engine since
    # v0.8 links and generated sources made before it lack. The rest of the
    # generated folder is the PC's, made with the analysis the lifted code
    # came from: left as it is.
    out = game.path(os.path.join(game.build, "port", "generated"))
    table = os.path.join(out, "dispatch_overlays.cpp")
    if os.path.isdir(out) and not os.path.isfile(table):
        import port            # engine/tools/port.py
        with open(table, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(port.overlays_source(game, out))
        print("the overlays' table written (%s)" % os.path.relpath(table, game.directory))
    lift_missing(game)
    return 0


def lift_missing(game) -> None:
    """Functions the lifted sources here call, or the game's list
    (config/functions.csv) has in src/lifted, that they lack - sources lifted
    by an older engine, which refused what the newer one lifts unverified,
    and functions the PC's runs never reached (its port.py lifts what a run
    stops at for the next build; the Mac takes the PC's sources as they
    are) - lifted now into src/lifted/lifted_missing.cpp and put in the
    dispatch table. Without them a run stops where the game first calls one:
    Burnout 3's 0x002FC820 (from a sound interrupt since v0.12's SPU2), its
    0x0024B0C0 (a countdown at boot that a slow first start - pipelines
    built again after an engine update - runs out)."""
    import csv
    import re

    lifted_dir = game.path(os.path.join(os.path.dirname(game.generated), "lifted"))
    generated_dir = game.path(game.generated)
    table = game.path(os.path.join(game.build, "port", "generated", "dispatch.cpp"))
    listed = game.path(game.status)
    if not (os.path.isdir(lifted_dir) and os.path.isfile(table) and os.path.isfile(listed)):
        return
    own = os.path.join(lifted_dir, "lifted_missing.cpp")
    defined = set()
    called = set()
    sources = [os.path.join(lifted_dir, n) for n in os.listdir(lifted_dir) if n.endswith(".cpp")]
    if os.path.isdir(generated_dir):
        sources += [os.path.join(generated_dir, n) for n in os.listdir(generated_dir) if n.endswith(".cpp")]
    discovered = os.path.join(os.path.dirname(table), "discovered.cpp")
    if os.path.isfile(discovered):
        sources.append(discovered)
    for path in sources:
        if path == own:
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        defined.update(int(m.group(1), 16) for m in
                       re.finditer(r"^void F_([0-9A-F]{8})\(lifted::Context& c\) \{", text, re.M))
        called.update(int(m.group(1), 16) for m in re.finditer(r"(?:Call|TailCall)\(c, 0x([0-9A-F]{8})u", text))
    with open(table, encoding="utf-8") as handle:
        text = handle.read()
    entries = {int(m.group(1), 16): m.group(2) for m in
               re.finditer(r"^    \{0x([0-9A-F]{8})u, (\w+), ", text, re.M)}
    with open(listed, newline="", encoding="utf-8") as handle:
        wanted = {int(row["address"], 16) for row in csv.DictReader(handle) if row["source"] == "src/lifted/"}
    # Called but not in the table, or in it with nothing to run - and what
    # an earlier run lifted (the table names them now).
    wanted |= {a for a in called if entries.get(a, "nullptr") == "nullptr"}
    if os.path.isfile(own):
        with open(own, encoding="utf-8", errors="replace") as handle:
            wanted |= {int(m.group(1), 16) for m in
                       re.finditer(r"^void F_([0-9A-F]{8})\(lifted::Context& c\) \{", handle.read(), re.M)}
    missing = sorted(a for a in wanted if a not in defined)
    if not missing:
        return
    from ps2decomp import lift
    from lift import _other_sections   # engine/tools/lift.py
    database = game.load_database()
    functions = {f["addr"]: f for f in database["functions"]}
    elf = game.load_elf()
    section = elf.exec_sections[0]
    lift.use_sections(_other_sections(elf, section))
    lift.use_function_starts(sorted(functions))
    image = elf.section_data(section)
    out = ["/* Lifted by engine/tools/analysis_ready.py: functions these sources",
           " * call, or the game's list has in src/lifted, that they lacked.",
           " * NOT VERIFIED. Generated - do not edit. */", "",
           '#include "ps2rt/lifted.h"', "", "using namespace generated;", "using namespace lifted;", ""]
    done = []
    bodies = {}
    queue = list(missing)
    seen = set(queue)
    while queue:
        address = queue.pop(0)
        f = functions.get(address)
        if not f or f["end"] <= address or not (section.addr <= address and f["end"] <= section.addr + len(image)):
            continue                    # (not a function this analysis knows in the code: left as it is)
        try:
            body = lift.lift_function(image, section.addr, address, f["end"], f.get("chunks", ()))
        except lift.Refused as refused:
            print("0x%08X: not lifted (%s)" % (address, refused))
            continue
        bodies[address] = body
        done.append(address)
        # What it calls in turn, lacking too.
        for m in re.finditer(r"(?:Call|TailCall)\(c, 0x([0-9A-F]{8})u", body):
            callee = int(m.group(1), 16)
            if callee not in seen and callee not in defined and entries.get(callee, "nullptr") == "nullptr":
                seen.add(callee)
                queue.append(callee)
    for address in sorted(bodies):         # (in address order: the same text each run)
        out += ["/* NOT VERIFIED: lifted where the sources lacked it */", bodies[address]]
    source = "\n".join(out) + "\n"
    old = ""
    if os.path.isfile(own):
        with open(own, encoding="utf-8") as handle:
            old = handle.read()
    if source != old:           # the same text: not touched, nothing built again
        with open(own, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(source)
    patched = False
    for address in done:
        entry = "{0x%08Xu, nullptr, nullptr, " % address
        name = "F_%08X" % address
        if entry in text:
            text = text.replace(entry, "{0x%08Xu, %s, nullptr, " % (address, name), 1)
        elif address not in entries:
            # A new entry, in address order (the table is searched by halves).
            size = functions[address]["end"] - address
            line = "    {0x%08Xu, %s, nullptr, %uu},\n" % (address, name, size)
            later = [a for a in entries if a > address]
            if later:
                at = text.index("    {0x%08Xu, " % min(later))
            else:
                at = text.index("};", text.index("kGuestFunctions[] = {"))
            text = text[:at] + line + text[at:]
            entries[address] = name
            count = re.search(r"kGuestFunctionCount = (\d+);", text)
            if count:
                text = text[:count.start(1)] + str(int(count.group(1)) + 1) + text[count.end(1):]
        else:
            continue
        declaration = "void %s(lifted::Context& c);\n" % name
        if declaration not in text:
            first = text.index("void F_")
            text = text[:first] + declaration + text[first:]
        patched = True
    if patched:
        with open(table, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    if source != old or patched:
        print("lifted %d function(s) the sources lacked: %s" % (len(done), ", ".join("0x%08X" % a for a in sorted(done))))


if __name__ == "__main__":
    raise SystemExit(main())
