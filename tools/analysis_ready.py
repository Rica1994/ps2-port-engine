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
    """Functions the game's list (config/functions.csv) has in src/lifted that
    the lifted sources here lack - sources lifted by an older engine, which
    refused what the newer one lifts unverified - lifted now into
    src/lifted/lifted_missing.cpp and put in the dispatch table. Without them
    a run stops where the game first calls one (Burnout 3's 0x002FC820, from
    a sound interrupt since v0.12's SPU2)."""
    import csv
    import re

    lifted_dir = game.path(os.path.join(os.path.dirname(game.generated), "lifted"))
    table = game.path(os.path.join(game.build, "port", "generated", "dispatch.cpp"))
    listed = game.path(game.status)
    if not (os.path.isdir(lifted_dir) and os.path.isfile(table) and os.path.isfile(listed)):
        return
    own = os.path.join(lifted_dir, "lifted_missing.cpp")
    defined = set()
    for name in os.listdir(lifted_dir):
        if name.endswith(".cpp") and os.path.join(lifted_dir, name) != own:
            with open(os.path.join(lifted_dir, name), encoding="utf-8", errors="replace") as handle:
                defined.update(int(m.group(1), 16) for m in
                               re.finditer(r"^void F_([0-9A-F]{8})\(lifted::Context& c\) \{", handle.read(), re.M))
    with open(listed, newline="", encoding="utf-8") as handle:
        wanted = sorted(int(row["address"], 16) for row in csv.DictReader(handle)
                        if row["source"] == "src/lifted/")
    missing = [a for a in wanted if a not in defined]
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
    out = ["/* Lifted by engine/tools/analysis_ready.py: functions the game's list",
           " * has in src/lifted that these sources lacked. NOT VERIFIED.",
           " * Generated - do not edit. */", "",
           '#include "ps2rt/lifted.h"', "", "using namespace generated;", "using namespace lifted;", ""]
    done = []
    for address in missing:
        f = functions.get(address)
        if not f or f["end"] <= address:
            print("0x%08X: not a function in this analysis; left out" % address)
            continue
        out += ["/* NOT VERIFIED: lifted where the sources lacked it */",
                lift.lift_function(image, section.addr, address, f["end"], f.get("chunks", ()))]
        done.append(address)
    source = "\n".join(out) + "\n"
    old = ""
    if os.path.isfile(own):
        with open(own, encoding="utf-8") as handle:
            old = handle.read()
    if source != old:           # the same text: not touched, nothing built again
        with open(own, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(source)
    with open(table, encoding="utf-8") as handle:
        text = handle.read()
    patched = False
    for address in done:
        entry = "{0x%08Xu, nullptr, nullptr, " % address
        if entry in text:
            text = text.replace(entry, "{0x%08Xu, F_%08X, nullptr, " % (address, address), 1)
            declaration = "void F_%08X(lifted::Context& c);\n" % address
            first = text.index("void F_")
            text = text[:first] + declaration + text[first:]
            patched = True
    if patched:
        with open(table, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    if source != old or patched:
        print("lifted %d function(s) the sources lacked: %s" % (len(done), ", ".join("0x%08X" % a for a in done)))


if __name__ == "__main__":
    raise SystemExit(main())
