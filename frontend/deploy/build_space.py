"""Assemble a self-contained Hugging Face Space (Gradio SDK) from the repo.

Output folder (default deploy/space_build/):
  app.py               entry point Spaces runs; starts workbench.py with deployment flags
  workbench.py         copy of ui/app.py
  src/                 analyze.py, stubs.py, live.py (integration layer + live Lookup)
  data/                cases.json, meta.json, evidence.json, literature.json, images/, examples/
                       (never decisions.json: sign-offs are per visitor session on the Space)
  gallery/             reference tiles as JPEG + their QuiltNet embeddings (relative paths)
  requirements.txt     CPU torch + open_clip + sklearn (gradio comes from the Space SDK)
  README.md            Space config header + data attribution

Nothing secret is copied: no .env, no tokens, no decisions.

Example:
    python3 deploy/build_space.py --data ui/data_semi --gallery embeddings/quiltnet_train_subset.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
DATA_FILES = ["cases.json", "meta.json", "evidence.json", "literature.json", "ihc.json"]
SRC_FILES = ["analyze.py", "stubs.py", "live.py"]

APP_PY = '''"""Hugging Face Space entry point: run the workbench with deployment settings."""
import runpy
import sys

# Session-only sign-offs (each visitor gets their own queue) and live Lookup over the bundled gallery.
sys.argv = ["workbench.py", "--data", "data", "--live", "gallery/quiltnet_gallery.csv", "--session-decisions"]
runpy.run_path("workbench.py", run_name="__main__")
'''

REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
open_clip_torch>=2.24
scikit-learn>=1.3
pandas
numpy
Pillow>=10.0
"""

README = """---
title: Pathology Assistant
emoji: 🔬
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.11"
app_file: app.py
pinned: false
short_description: AI pre-read and reference lookup for H&E tiles
---

# Pathology Assistant

AI pre-read and reference lookup for H&E tissue tiles (2026 Ashby Prize Hackathon, Cancer Center at Illinois).

- **Queue / Review**: tiles the AI is unsure about go to a review queue; confident tiles are cleared automatically.
- **Lookup**: compare an ambiguous tile with the closest confirmed reference tiles of each candidate class.
- **References**: PubMed papers for each differential, AI-screened; quotes are verbatim and checked against each abstract.
- **Evidence**: how often "confident" tiles were wrong.

Research demo with placeholder (stub) classifier, uncertainty and retrieval modules. Not for clinical use.

## Data and models
- Tiles: NCT-CRC-HE-100K and CRC-VAL-HE-7K, Kather JN, Halama N, Marx A (2018), [Zenodo 1214456](https://zenodo.org/records/1214456), CC BY 4.0.
- Embeddings: QuiltNet-B-32 ([wisdomik/QuiltNet-B-32](https://huggingface.co/wisdomik/QuiltNet-B-32)).
- Literature metadata: PubMed via NCBI E-utilities.
"""


def copy_data(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)
    for folder in ["images", "examples"]:
        if (src / folder).is_dir():
            shutil.copytree(src / folder, dest / folder)


def build_gallery(gallery_csv: Path, dest: Path, quality: int) -> int:
    """Reference tiles -> JPEG under tiles/<CLASS>/, CSV rewritten with paths relative to it."""
    tiles = dest / "tiles"
    rows_out = []
    with gallery_csv.open() as f:
        for row in csv.DictReader(f):
            src = Path(row["image_path"])
            rel = Path("tiles") / src.parent.name / f"{src.stem}.jpg"
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.open(src).convert("RGB").save(dest / rel, quality=quality)
            rows_out.append({"image_path": rel.as_posix(), "embedding": row["embedding"]})
    with (dest / "quiltnet_gallery.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "embedding"])
        writer.writeheader()
        writer.writerows(rows_out)
    return len(rows_out)


def folder_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO / "ui/data_semi")
    ap.add_argument("--gallery", type=Path, default=REPO / "embeddings/quiltnet_train_subset.csv")
    ap.add_argument("--out", type=Path, default=REPO / "deploy/space_build")
    ap.add_argument("--gradio-version", default=None, help="default: the version installed in ui/.venv")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    shutil.copy2(REPO / "ui/app.py", args.out / "workbench.py")
    (args.out / "app.py").write_text(APP_PY)
    (args.out / "src").mkdir()
    for name in SRC_FILES:
        shutil.copy2(REPO / "src" / name, args.out / "src" / name)
    copy_data(args.data, args.out / "data")
    n = build_gallery(args.gallery, args.out / "gallery", args.jpeg_quality)

    version = args.gradio_version
    if version is None:
        from importlib.metadata import version as pkg_version
        version = pkg_version("gradio")
    (args.out / "requirements.txt").write_text(REQUIREMENTS)
    (args.out / "README.md").write_text(README.format(gradio_version=version))

    leaked = [p for p in args.out.rglob("*") if p.name in (".env", "decisions.json") or "token" in p.name.lower()]
    if leaked:
        raise SystemExit(f"refusing: sensitive files in the bundle: {leaked}")
    print(f"built {args.out}: {n} gallery tiles, gradio {version}, {folder_mb(args.out):.1f} MB")


if __name__ == "__main__":
    main()
