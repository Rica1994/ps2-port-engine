#!/bin/bash
# The Mac's step between iPad builds, from the game repository's root:
#
#   engine/tools/ipad_update.sh [--pc http://192.168.1.20:8765] [--game games/burnout3]
#
# 1. the sources: git pull, and the engine submodule to what the game names;
# 2. the VU programs compiled ahead on the PC (engine/tools/ipad_reports.py
#    there), when they changed: into the game's build/port/generated;
# 3. the Xcode project brought up to date (new files, the PC's address for
#    the app's reports: CMake's PS2_REPORT_URL).
#
# Then Run in Xcode. --pc is remembered (build-ios/report_url); the first time
# it is the address engine/tools/ipad_reports.py prints on the PC.
set -euo pipefail
GAME="games/burnout3"
PC=""
while [ $# -gt 0 ]; do
    case "$1" in
        --pc) PC="$2"; shift 2 ;;
        --game) GAME="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -f "$GAME/game.json" ] || { echo "no $GAME/game.json: run this from the game repository's root" >&2; exit 2; }
NAME="$(basename "$GAME")"
mkdir -p build-ios
if [ -n "$PC" ]; then
    case "$PC" in http://*) ;; *) PC="http://$PC" ;; esac
    case "$PC" in *:[0-9]*) ;; *) PC="$PC:8765" ;; esac
    echo "$PC" > build-ios/report_url
elif [ -f build-ios/report_url ]; then
    PC="$(cat build-ios/report_url)"
fi

echo "== the sources"
git pull --ff-only
git submodule update --init --recursive
git -C engine log -1 --format='engine at %h: %s'

GENERATED="$GAME/build/port/generated"
if [ -n "$PC" ]; then
    echo "== the VU programs from $PC"
    if REMOTE="$(curl -fsS --max-time 5 "$PC/vu/stamp")"; then
        LOCAL="$(cat "$GENERATED/.vu_stamp" 2>/dev/null || echo none)"
        if [ "$REMOTE" != "$LOCAL" ]; then
            TMP="$(mktemp -d)"
            curl -fsS --max-time 60 -o "$TMP/vu.zip" "$PC/vu.zip"
            rm -f "$GENERATED"/vu_blocks_*.cpp "$GENERATED"/vu_programs.cpp
            unzip -q -o "$TMP/vu.zip" -d "$GENERATED"
            rm -rf "$TMP"
            echo "new: $(ls "$GENERATED"/vu_*.cpp | wc -l | tr -d ' ') files"
        else
            echo "unchanged"
        fi
    else
        echo "the PC does not answer (is engine/tools/ipad_reports.py running there?) - the programs stay as they are"
    fi
else
    echo "(no --pc yet: the VU programs stay as they are, and the app sends no reports)"
fi

echo "== the Xcode project"
[ -d "build-ios/$NAME" ] || { echo "no build-ios/$NAME: run engine/tools/ios_setup.sh first" >&2; exit 1; }
cmake "build-ios/$NAME" -DPS2_REPORT_URL="$PC" > /dev/null
echo
echo "Done. Run in Xcode (Product > Run)."
