#!/usr/bin/env python3
"""Assemble the auto-update manifest (latest.json) for cargo-packager-updater.

It scans the dist/ directory for signed installer artifacts (each <file>.sig
holds the detached signature) and maps them to platform keys the in-app updater
queries:

    linux-x86_64    -> *.AppImage
    windows-x86_64  -> *-setup.exe
    darwin-x86_64   -> *.app.tar.gz   (universal build, same file)
    darwin-aarch64  -> *.app.tar.gz   (universal build, same file)

Artifacts without a matching .sig are skipped (signing is optional until the
update key is configured). If nothing is signed, exits non-zero so the workflow
can report that the manifest was skipped.
"""
import argparse
import datetime
import glob
import json
import os
import sys


def find_one(dist, pattern):
    matches = sorted(glob.glob(os.path.join(dist, pattern)))
    return matches[0] if matches else None


def signed(path):
    """Return (url_filename, signature_text) if a .sig exists, else None."""
    if not path:
        return None
    sig = path + ".sig"
    if not os.path.exists(sig):
        return None
    with open(sig, "r", encoding="utf-8") as fh:
        return os.path.basename(path), fh.read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--dist", required=True)
    ap.add_argument("--base-url", required=True, help="Release download base URL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--notes", default="See the release notes on GitHub.")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    candidates = {
        "linux-x86_64": find_one(args.dist, "*.AppImage"),
        "windows-x86_64": find_one(args.dist, "*-setup.exe"),
        "darwin-x86_64": find_one(args.dist, "*.app.tar.gz"),
        "darwin-aarch64": find_one(args.dist, "*.app.tar.gz"),
    }

    platforms = {}
    for key, path in candidates.items():
        s = signed(path)
        if s:
            name, signature = s
            platforms[key] = {"signature": signature, "url": f"{base}/{name}"}

    if not platforms:
        print("No signed artifacts found; not writing latest.json", file=sys.stderr)
        sys.exit(1)

    manifest = {
        "version": args.version,
        "notes": args.notes,
        "pub_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platforms": platforms,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote {args.out} with platforms: {', '.join(platforms)}")


if __name__ == "__main__":
    main()
