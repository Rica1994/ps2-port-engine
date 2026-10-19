#!/usr/bin/env python3
"""The developer's PC's end of a tablet build's reports (engine/host/src/report.cpp).

Run it from the game repository's root on the PC that builds the game's code,
and leave it running:

    python engine/tools/ipad_reports.py --game games/burnout3 --out ../ipad-reports

The iPad build (its address set by engine/tools/ipad_update.sh on the Mac)
sends, as it runs, every fifteen seconds and when it leaves the front:

* each run's log, kept in --out under the run's name
  (burnout3-20261016-151448-scripted.log), the newest copy replacing the last;
* vu_programs_new.txt, the VU programs it started. What is new in it is added
  to the game's build/vu_programs.txt and compiled ahead at once
  (engine/tools/vurecomp.py into build/port/generated) - the next iPad build
  has them.

The Mac takes the compiled programs from here (ipad_update.sh):

    GET /vu/stamp   what the compiled programs were made from (a hash)
    GET /vu.zip     build/port/generated/vu_*.cpp

Everything stays on the PC and the local network: the logs and the programs
come from the player's own disc and are never committed.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import os
import re
import socket
import subprocess
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
LOCK = threading.Lock()


def say(text: str) -> None:
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), text), flush=True)


def lines_of(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\r\n") for line in handle if line.strip()]


def stamp_of(manifest: str) -> str:
    """The programs and the compiler that made them: a new vurecomp.py (an
    engine update) compiles them again, as new programs do."""
    if not os.path.exists(manifest):
        return "none"
    digest = hashlib.sha1()
    for path in (manifest, os.path.join(HERE, "vurecomp.py")):
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


class Receiver:
    def __init__(self, game: str, out: str):
        self.game = game
        self.out = out
        self.manifest = os.path.join(game, "build", "vu_programs.txt")
        self.generated = os.path.join(game, "build", "port", "generated")
        os.makedirs(out, exist_ok=True)

    def stamp(self) -> str:
        path = os.path.join(self.generated, ".vu_stamp")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                return handle.read().strip()
        return stamp_of(self.manifest)

    def save(self, name: str, body: bytes) -> None:
        path = os.path.join(self.out, name)
        part = path + ".part"
        with open(part, "wb") as handle:
            handle.write(body)
        os.replace(part, path)
        if name == "vu_programs_new.txt":
            self.merge(body)
        else:
            say("%s: %d KB" % (name, len(body) // 1024))

    def merge(self, body: bytes) -> None:
        """What is new in the tablet's recording, added to the manifest and compiled."""
        old = lines_of(self.manifest)
        known = set(old)
        new = []
        for line in body.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("vu") and line not in known:
                known.add(line)
                new.append(line)
        if not new:
            return
        os.makedirs(os.path.dirname(self.manifest), exist_ok=True)
        with open(self.manifest, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(old + new) + "\n")
        say("vu_programs_new.txt: %d new program line(s); compiling ahead..." % len(new))
        self.compile()

    def compile(self) -> None:
        os.makedirs(self.generated, exist_ok=True)
        result = subprocess.run([sys.executable, os.path.join(HERE, "vurecomp.py"), "--manifest", self.manifest,
                                 "--out-dir", self.generated], capture_output=True, text=True)
        if result.returncode != 0:
            say("vurecomp failed:\n" + result.stdout + result.stderr)
            return
        with open(os.path.join(self.generated, ".vu_stamp"), "w", encoding="utf-8") as handle:
            handle.write(stamp_of(self.manifest))
        say(result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "compiled")
        say("the Mac takes them with: engine/tools/ipad_update.sh")

    def vu_zip(self) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(os.listdir(self.generated)):
                if name.startswith("vu_") and name.endswith(".cpp"):
                    archive.write(os.path.join(self.generated, name), name)
            archive.writestr(".vu_stamp", self.stamp())
        return buffer.getvalue()


def handler_for(receiver: Receiver):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def reply(self, code: int, body: bytes, kind: str = "text/plain; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            match = re.match(r"^/upload/(.+)$", self.path)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else b""
            if not match or not NAME.match(match.group(1)) or length > 256 << 20:
                self.reply(400, b"bad request\n")
                return
            with LOCK:
                try:
                    receiver.save(match.group(1), body)
                except OSError as error:
                    self.reply(500, (str(error) + "\n").encode())
                    return
            self.reply(200, b"ok\n")

        def do_GET(self) -> None:
            with LOCK:
                if self.path == "/vu/stamp":
                    self.reply(200, (receiver.stamp() + "\n").encode())
                elif self.path == "/vu.zip":
                    self.reply(200, receiver.vu_zip(), "application/zip")
                elif self.path == "/":
                    names = sorted(os.listdir(receiver.out))
                    self.reply(200, ("reports in %s:\n%s\n" % (os.path.abspath(receiver.out), "\n".join(names))).encode())
                else:
                    self.reply(404, b"not found\n")

        def log_message(self, *args) -> None:       # quiet: say() tells what matters
            pass

    return Handler


def addresses() -> list:
    """This PC's addresses on the local network (the one with a route out first)."""
    found = []
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("10.255.255.255", 1))           # no packet is sent
        found.append(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            if info[4][0] not in found and not info[4][0].startswith("127."):
                found.append(info[4][0])
    except OSError:
        pass
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default="games/burnout3", help="the game's folder (games/<name>)")
    parser.add_argument("--out", default=os.path.join("..", "ipad-reports"), help="where the logs go")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.game, "game.json")):
        sys.exit("no %s/game.json: run this from the game repository's root" % args.game)
    receiver = Receiver(args.game, args.out)
    if os.path.exists(receiver.manifest) and receiver.stamp() != stamp_of(receiver.manifest):
        say("the VU programs were compiled by another vurecomp (an engine update): compiling them again...")
        receiver.compile()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler_for(receiver))
    say("receiving the iPad's reports into %s" % os.path.abspath(args.out))
    for address in addresses():
        say("this PC: http://%s:%d   (on the Mac, once: engine/tools/ipad_update.sh --pc http://%s:%d)"
            % (address, args.port, address, args.port))
    say("leave this window open; Ctrl+C stops it")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
