#!/bin/bash
# A port's iPad build, on a Mac: the tools, MoltenVK, and the Xcode project.
#
#   engine/tools/ios_setup.sh --game games/<name> --team <TEAM ID> --bundle-id com.<you>.<name>
#
# Run from the game repository's root, after the game's code was generated
# from your disc (docs/IOS.md: isotool, analyze, derive, lift, port.py
# generate). Writes build-ios/<name>.xcodeproj; open it in Xcode, choose your
# iPad, and Run. Needs Xcode (from the App Store) and Homebrew (brew.sh).
set -euo pipefail

GAME=""
TEAM=""
BUNDLE=""
MOLTENVK_VERSION="v1.4.2"
DEPLOYMENT="15.0"
while [ $# -gt 0 ]; do
    case "$1" in
        --game) GAME="$2"; shift 2 ;;
        --team) TEAM="$2"; shift 2 ;;
        --bundle-id) BUNDLE="$2"; shift 2 ;;
        --moltenvk) MOLTENVK_VERSION="$2"; shift 2 ;;
        --ios) DEPLOYMENT="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$GAME" ] || { echo "--game games/<name> is needed" >&2; exit 2; }
[ -f "$GAME/game.json" ] || { echo "no $GAME/game.json: run this from the game repository's root" >&2; exit 2; }
NAME="$(basename "$GAME")"
ROOT="$(pwd)"
DEPS="$ROOT/build-ios/deps"
mkdir -p "$DEPS"

echo "== the tools"
xcode-select -p >/dev/null 2>&1 || { echo "Xcode is needed: install it from the App Store, open it once, then: xcode-select --install" >&2; exit 1; }
command -v brew >/dev/null || { echo "Homebrew is needed: https://brew.sh" >&2; exit 1; }
for tool in cmake ninja glslang python3; do
    command -v "${tool/glslang/glslangValidator}" >/dev/null || brew install "$tool"
done

echo "== MoltenVK $MOLTENVK_VERSION (Vulkan on Metal, carried in the app)"
MVK="$DEPS/MoltenVK/MoltenVK/dynamic/MoltenVK.xcframework"
if [ ! -d "$MVK" ]; then
    curl -fL -o "$DEPS/MoltenVK-ios.tar" \
        "https://github.com/KhronosGroup/MoltenVK/releases/download/$MOLTENVK_VERSION/MoltenVK-ios.tar"
    tar -xf "$DEPS/MoltenVK-ios.tar" -C "$DEPS"
fi
[ -d "$MVK/ios-arm64/MoltenVK.framework" ] || { echo "no MoltenVK.framework in $MVK" >&2; exit 1; }

GENERATED="$ROOT/$GAME/build/port/generated"
[ -f "$GENERATED/dispatch.cpp" ] || { echo "no $GENERATED/dispatch.cpp: run port.py --game $GAME generate first" >&2; exit 1; }

echo "== the Xcode project"
cmake -S "$GAME/port" -B "build-ios/$NAME" -G Xcode \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT" \
    -DPS2_GENERATED_DIR="$GENERATED" \
    -DPS2_MOLTENVK="$MVK" \
    -DPS2_IOS_TEAM="$TEAM" \
    -DPS2_IOS_BUNDLE_ID="${BUNDLE:-com.example.$NAME}" \
    -DSDL_TESTS=OFF

echo
echo "Done. Open this in Xcode:"
ls -d "build-ios/$NAME"/*.xcodeproj
echo "Choose the '$NAME' scheme and your iPad, then Product > Run."
echo "(Release is faster: Product > Scheme > Edit Scheme > Run > Build Configuration > Release.)"
