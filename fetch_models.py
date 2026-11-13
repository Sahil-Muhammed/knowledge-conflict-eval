#!/usr/bin/env python3
"""Download the GGUFs listed in models.json from Hugging Face (skips files already complete)."""
import argparse
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Length"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--models", nargs="+", help="subset of models.json names (default: all)")
    args = ap.parse_args()

    models = json.loads((ROOT / "models.json").read_text())
    out = Path(args.models_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.models or models:
        m = models[name]
        url = f"https://huggingface.co/{m['hf']}/resolve/main/{m['file']}"
        dest = out / m["file"]
        size = remote_size(url)
        if dest.exists() and dest.stat().st_size == size:
            print(f"{name}: already present")
            continue
        print(f"{name}: downloading {size / 2**30:.2f} GiB")
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 22)
        if tmp.stat().st_size != size:
            raise RuntimeError(f"{name}: incomplete download")
        tmp.rename(dest)


if __name__ == "__main__":
    main()
