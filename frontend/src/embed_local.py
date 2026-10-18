"""Compute tile embeddings locally / on Colab with the official hackathon tooling.

Thin driver around the official `run_embeddings.py` (ncsa/ashby-hackathon-2026)
so local QuiltNet/PLIP embeddings use exactly the same loading, preprocessing,
L2 normalization and CSV format (`image_path, embedding`) as the CONCH/MUSK
embeddings computed on Delta. Output is resumable: rerun to continue.

Example:
    python -m src.embed_local --input-dir data/train_subset --models quiltnet plip
    python -m src.embed_local --input-dir data/val_subset --models quiltnet --out-dir /content/drive/MyDrive/ashby/embeddings
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Short names -> spec strings understood by the official models.resolve_adapter.
MODEL_SPECS = {
    "quiltnet": "openclip:wisdomik/QuiltNet-B-32",
    "plip": "hfclip:vinid/plip",
    "conch": "conch",
    "musk": "musk",
}
DEFAULT_REPO = Path("external/ashby-hackathon-2026")


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def output_path(out_dir: Path, model: str, input_dir: Path) -> Path:
    return out_dir / f"{model}_{input_dir.resolve().name}.csv"


def run_model(
    model: str, input_dir: Path, out_dir: Path, repo: Path, device: str, batch_size: int
) -> Path:
    out = output_path(out_dir, model, input_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str((repo / "run_embeddings.py").resolve()),
        "--model", MODEL_SPECS[model],
        "--input-dir", str(input_dir.resolve()),
        "--output", str(out.resolve()),
        "--device", device,
        "--batch-size", str(batch_size),
        # The official flush rewrites the whole CSV; checkpoint less often for speed.
        "--checkpoint-every", "500",
    ]
    print(f"[{model}] -> {out}", flush=True)
    subprocess.run(cmd, check=True, cwd=repo)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True, help="tile folder, one subfolder per class")
    p.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=["quiltnet"])
    p.add_argument("--out-dir", type=Path, default=Path("embeddings"))
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="clone of ncsa/ashby-hackathon-2026")
    p.add_argument("--device", default=None, help="default: cuda > mps > cpu")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    if not (args.repo / "run_embeddings.py").exists():
        sys.exit(
            f"{args.repo}/run_embeddings.py not found. Clone it first:\n"
            f"  git clone --depth 1 https://github.com/ncsa/ashby-hackathon-2026 {args.repo}"
        )
    device = args.device or default_device()
    for model in args.models:
        run_model(model, args.input_dir, args.out_dir, args.repo, device, args.batch_size)


if __name__ == "__main__":
    main()
