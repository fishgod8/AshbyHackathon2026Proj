"""Sample a small, class-balanced tile subset from the NCT-CRC-HE Zenodo zips.

Zenodo serves HTTP range requests, so we read the zip's central directory
remotely and download only the sampled tiles instead of the full 11.7 GB archive.
Stdlib only. Re-running skips tiles that already exist (resumable).

Example:
    python -m src.make_subset --dataset train --per-class 200
    python -m src.make_subset --dataset val --per-class 100
"""

from __future__ import annotations

import argparse
import http.client
import io
import random
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ZENODO = "https://zenodo.org/api/records/1214456/files"
DATASETS = {
    "train": f"{ZENODO}/NCT-CRC-HE-100K.zip/content",
    "val": f"{ZENODO}/CRC-VAL-HE-7K.zip/content",
}
CLASSES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg")


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file backed by HTTP range requests, with read-ahead."""

    def __init__(self, url: str, readahead: int = 1 << 20) -> None:
        self.url = url
        self.readahead = readahead
        self.pos = 0
        self.buf_start = 0
        self.buf = b""
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as resp:
            self.size = int(resp.headers["Content-Range"].split("/")[-1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req) as resp:
            return resp.read()

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        end = min(self.pos + n, self.size)
        if end <= self.pos:
            return b""
        buf_end = self.buf_start + len(self.buf)
        if not (self.buf_start <= self.pos and end <= buf_end):
            fetch_end = min(max(end, self.pos + self.readahead), self.size)
            self.buf = self._fetch(self.pos, fetch_end)
            self.buf_start = self.pos
        lo = self.pos - self.buf_start
        out = self.buf[lo : lo + (end - self.pos)]
        self.pos += len(out)
        return out


def list_tiles(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    """Group image entries by class, using the parent folder name as the label."""
    by_class: dict[str, list[str]] = defaultdict(list)
    for name in zf.namelist():
        if not name.lower().endswith(IMAGE_SUFFIXES):
            continue
        label = Path(name).parent.name
        if label in CLASSES:
            by_class[label].append(name)
    return by_class


def sample(by_class: dict[str, list[str]], per_class: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    picked: list[str] = []
    for label in CLASSES:
        names = sorted(by_class.get(label, []))
        picked += rng.sample(names, min(per_class, len(names)))
    return picked


def fetch_range(url: str, start: int, end: int, retries: int = 8) -> bytes:
    """GET bytes [start, end), backing off on Zenodo rate limits (HTTP 429) and network hiccups."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:  # before URLError: HTTPError is a subclass of it
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            wait = float(e.headers.get("Retry-After") or 2 ** attempt)
            reset = e.headers.get("X-RateLimit-Reset")
            if e.code == 429 and reset:
                wait = max(wait, float(reset) - time.time() + 1)
            time.sleep(min(max(wait, 1), 120))
        except (TimeoutError, ConnectionError, urllib.error.URLError, http.client.IncompleteRead):
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 30))  # stalled or dropped connection: try again
    raise RuntimeError("unreachable")


def fetch_member(url: str, info: zipfile.ZipInfo) -> bytes:
    """Download and decompress one zip entry with a single range request."""
    # Local header is 30 bytes + name + extra; extra length can differ from the
    # central directory copy, so over-fetch a little and parse the real lengths.
    start = info.header_offset
    end = start + 30 + len(info.orig_filename.encode()) + 1024 + info.compress_size
    raw = fetch_range(url, start, end)
    if raw[:4] != b"PK\x03\x04":
        raise ValueError(f"bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    lo = 30 + name_len + extra_len
    payload = raw[lo : lo + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        data = payload
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(payload, -15)
    else:
        raise ValueError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(data) != info.CRC:
        raise ValueError(f"CRC mismatch for {info.filename}")
    return data


def save_member(url: str, info: zipfile.ZipInfo, out: Path) -> None:
    dest = out / Path(info.filename).parent.name / Path(info.filename).name
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(fetch_member(url, info))
    tmp.rename(dest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=DATASETS, default="train")
    p.add_argument("--per-class", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="default: data/<dataset>_subset")
    p.add_argument("--workers", type=int, default=4, help="keep low; Zenodo rate-limits")
    args = p.parse_args()

    url = DATASETS[args.dataset]
    out = args.out or Path("data") / f"{args.dataset}_subset"
    zf = zipfile.ZipFile(HttpRangeFile(url))
    by_class = list_tiles(zf)
    print("tiles per class in archive:", {k: len(v) for k, v in sorted(by_class.items())})

    infos = [zf.getinfo(name) for name in sample(by_class, args.per_class, args.seed)]
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(save_member, url, info, out): info.filename for info in infos}
        for done, fut in enumerate(as_completed(futures), start=1):
            try:
                fut.result()
            except Exception as e:
                failed.append(futures[fut])
                print(f"failed {futures[fut]}: {e}", file=sys.stderr)
            if done % 100 == 0 or done == len(infos):
                print(f"{done}/{len(infos)} tiles -> {out}", flush=True)
    if failed:
        sys.exit(f"{len(failed)} tiles failed; rerun the same command to resume.")


if __name__ == "__main__":
    main()
