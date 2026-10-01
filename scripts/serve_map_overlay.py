#!/usr/bin/env python3
"""Serve the interactive map overlay on http://127.0.0.1:8765/"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "outputs" / "week2_cnn_bbox" / "map_overlay"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if not (DIR / "index.html").exists():
        raise SystemExit(f"Missing {DIR / 'index.html'} — run scripts/build_map_overlay.py first")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DIR))
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving {DIR}")
        print(f"Open http://{args.host}:{args.port}/")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
