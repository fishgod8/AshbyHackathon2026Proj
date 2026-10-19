"""Fetch BACH microscopy photos (originals, byte for byte) for Mayumi's split, by range request from Zenodo.

Needed for patch-level work: the UI data folder keeps only downscaled display JPEGs, and resizing or JPEG
compression shifts the embeddings. Resumable; only the photos (never the whole-slide images).

    python -m src.bach_raw --out data/bach_raw
"""

from __future__ import annotations

import argparse
import io
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.bach import BACH_ZIP, MAYUMI_BRANCH, TEAM_REPO
from src.make_subset import HttpRangeFile, fetch_member


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/bach_raw"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    raw = subprocess.run(["git", "-C", str(TEAM_REPO), "show", f"{MAYUMI_BRANCH}:bach/image_embeddings.csv"],
                         check=True, capture_output=True).stdout
    split = pd.read_csv(io.BytesIO(raw), usecols=["image_id", "label", "split"])
    zf = zipfile.ZipFile(HttpRangeFile(BACH_ZIP))
    members = {Path(i.filename).name: i for i in zf.infolist() if "/Photos/" in i.filename and i.filename.endswith(".tif")}
    todo = [r for r in split.itertuples() if not (args.out / r.split / r.image_id).exists()]
    print(f"{len(split)} photos in Mayumi's split, {len(todo)} to fetch", flush=True)

    def fetch(r):
        dest = args.out / r.split / r.image_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(fetch_member(BACH_ZIP, members[r.image_id]))
        tmp.rename(dest)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, _ in enumerate(pool.map(fetch, todo), start=1):
            if n % 50 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} photos", flush=True)
    split.to_csv(args.out / "split.csv", index=False)


if __name__ == "__main__":
    main()
