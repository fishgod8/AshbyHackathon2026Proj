"""Extract a reproducible MUSK embedding sample from CRC-VAL-HE-7K.

The selected images are for workflow development only. This script does not
measure or report model accuracy and the sample is not an independent test set.
"""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import numpy as np
import PIL
import timm
import torch
import torchvision
from PIL import Image
from timm.models import create_model
from torchvision import transforms
from tqdm import tqdm


CLASS_NAMES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TIFF_SUFFIXES = {".tif", ".tiff"}
RANDOM_SEED = 42
EMBEDDING_DIMENSION = 1024
MODEL_NAME = "musk_large_patch16_384"
MODEL_CHECKPOINT = "hf_hub:xiangjx/musk"
MUSK_REPOSITORY = Path(r"C:\Users\Jiayi\MUSK")
RESUME_EMBEDDINGS = ".resume_embeddings.npy"
RESUME_JOURNAL = ".resume_completed.jsonl"
RESUME_CONFIG = ".resume_config.json"


def sample_images(data_dir: Path, per_class: int) -> list[dict[str, str]]:
    """Sample sorted image paths reproducibly from each expected class folder."""
    if per_class < 1:
        raise ValueError("per-class must be at least 1")
    if not data_dir.is_dir():
        raise ValueError(f"dataset directory does not exist: {data_dir}")

    actual_classes = tuple(
        sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    )
    missing_classes = sorted(set(CLASS_NAMES) - set(actual_classes))
    extra_classes = sorted(set(actual_classes) - set(CLASS_NAMES))
    if missing_classes or extra_classes:
        details = []
        if missing_classes:
            details.append("missing: " + ", ".join(missing_classes))
        if extra_classes:
            details.append("unexpected: " + ", ".join(extra_classes))
        raise ValueError("expected the nine CRC class folders (" + "; ".join(details) + ")")

    rng = random.Random(RANDOM_SEED)
    selected = []
    for label in CLASS_NAMES:
        class_dir = data_dir / label
        candidates = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(candidates) < per_class:
            raise ValueError(
                f"class {label} has only {len(candidates)} supported images; "
                f"cannot sample {per_class}"
            )
        for image_path in rng.sample(candidates, per_class):
            selected.append(
                {
                    "image_id": image_path.stem,
                    "image_path": image_path.relative_to(data_dir).as_posix(),
                    "label": label,
                }
            )
    return selected


def list_all_images(data_dir: Path) -> list[dict[str, str]]:
    """List all TIFF images in fixed class/path order for a full extraction."""
    if not data_dir.is_dir():
        raise ValueError(f"dataset directory does not exist: {data_dir}")
    actual_classes = {path.name for path in data_dir.iterdir() if path.is_dir()}
    if actual_classes != set(CLASS_NAMES):
        raise ValueError(
            "dataset must contain exactly these class folders: "
            + ", ".join(CLASS_NAMES)
        )

    images = []
    for label in CLASS_NAMES:
        class_dir = data_dir / label
        for image_path in sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
        ):
            images.append(
                {
                    "image_id": image_path.stem,
                    "image_path": image_path.relative_to(data_dir).as_posix(),
                    "label": label,
                }
            )
    if not images:
        raise ValueError(f"no TIFF images found under {data_dir}")
    image_ids = [sample["image_id"] for sample in images]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("image_id values are not unique; refusing ambiguous metadata")
    return images


def save_manifest(samples: list[dict[str, str]], output_dir: Path) -> None:
    """Persist the exact sampling order before extraction starts."""
    manifest_path = output_dir / "sample_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=("row_index", "image_id", "image_path", "label")
        )
        writer.writeheader()
        for row_index, sample in enumerate(samples):
            writer.writerow({"row_index": row_index, **sample})


def load_model_and_transform() -> tuple[Any, Any, dict[str, str], Path]:
    """Load the existing MUSK model once, without allowing a model download."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this extraction but is unavailable")
    if not MUSK_REPOSITORY.is_dir():
        raise RuntimeError(f"MUSK repository not found: {MUSK_REPOSITORY}")

    model_cache = Path.home() / ".cache" / "model.safetensors"
    if not model_cache.is_file():
        raise RuntimeError(
            "local MUSK checkpoint is missing; refusing to download it: "
            f"{model_cache}"
        )

    sys.path.insert(0, str(MUSK_REPOSITORY))
    from musk import modeling, utils

    model = create_model(MODEL_NAME)
    utils.load_model_and_may_interpolate(
        MODEL_CHECKPOINT,
        model,
        "model|module",
        "",
        local_dir=str(model_cache.parent),
    )
    model = model.to(device="cuda", dtype=torch.float16)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(384, interpolation=3),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "musk": version("musk"),
        "pillow": PIL.__version__,
        "numpy": np.__version__,
    }
    return model, transform, versions, model_cache


def sha256_file(path: Path) -> str:
    """Calculate a stable digest for the cached checkpoint file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_config(
    output_dir: Path,
    data_dir: Path,
    samples: list[dict[str, str]],
    versions: dict[str, str],
    model_cache: Path,
    full_dataset: bool,
) -> dict[str, Any]:
    """Write extraction settings and return the resume-compatibility config."""
    class_counts = {
        label: sum(sample["label"] == label for sample in samples)
        for label in CLASS_NAMES
    }
    manifest_digest = hashlib.sha256()
    for sample in samples:
        image_path = data_dir / Path(sample["image_path"])
        manifest_digest.update(
            json.dumps(
                {
                    **sample,
                    "size_bytes": image_path.stat().st_size,
                    "mtime_ns": image_path.stat().st_mtime_ns,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        manifest_digest.update(b"\n")
    config: dict[str, Any] = {
        "purpose": (
            "CRC-VAL-HE-7K is external validation data. These embeddings are "
            "features only; no classifier is trained and no accuracy is reported. "
            "If used for training or tuning, reserve other data for evaluation; "
            "the full dataset must not then be called an independent test set."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "CRC-VAL-HE-7K",
            "root": str(data_dir.resolve()),
            "source_role": "external validation data",
            "image_count": len(samples),
            "class_counts": class_counts,
            "ordered_manifest_sha256": manifest_digest.hexdigest(),
            "image_extensions": sorted(TIFF_SUFFIXES) if full_dataset else "sample mode",
            "image_order": "CLASS_NAMES order, then lexicographically sorted relative path",
        },
        "model": MODEL_NAME,
        "checkpoint": MODEL_CHECKPOINT,
        "checkpoint_cache": str(model_cache),
        "checkpoint_sha256": sha256_file(model_cache),
        "preprocessing": {
            "resize": 384,
            "interpolation_argument": 3,
            "center_crop": 384,
            "to_tensor": True,
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5],
            "image_mode": "RGB",
        },
        "extraction": {
            "with_head": False,
            "out_norm": True,
            "ms_aug": False,
            "return_global": True,
            "device": "cuda",
            "dtype": "float16",
            "batch_size": 1,
            "output_dtype": "float32",
            "embedding_dimension": EMBEDDING_DIMENSION,
        },
        "sampling": {
            "mode": "all_images" if full_dataset else "per_class_sample",
            "seed": RANDOM_SEED if not full_dataset else None,
            "per_class": None if full_dataset else class_counts,
            "classes": list(CLASS_NAMES),
            "order": "CLASS_NAMES order, then sorted relative image path",
        },
        "software_versions": versions,
    }
    config.pop("created_utc")
    return config


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as json_file:
        json.dump(value, json_file, sort_keys=True)
        json_file.flush()
        os.fsync(json_file.fileno())
    os.replace(temporary_path, path)


def prepare_resume_state(
    output_dir: Path, samples: list[dict[str, str]], config: dict[str, Any], resume: bool
) -> tuple[np.ndarray, set[int], Path, Path]:
    """Open fixed-row checkpoint storage and validate its config/journal on resume."""
    embeddings_path = output_dir / RESUME_EMBEDDINGS
    journal_path = output_dir / RESUME_JOURNAL
    config_path = output_dir / RESUME_CONFIG
    final_paths = (
        output_dir / "embeddings.npy",
        output_dir / "metadata.csv",
    )
    final_exists = [path.exists() for path in final_paths]
    if all(final_exists):
        raise FileExistsError(
            "final output already exists; refusing to overwrite: "
            + ", ".join(str(path) for path in final_paths)
        )

    checkpoint_paths = (embeddings_path, journal_path, config_path)
    checkpoint_exists = [path.exists() for path in checkpoint_paths]
    if any(checkpoint_exists) and not all(checkpoint_exists):
        raise ValueError(
            "incomplete resume checkpoint files; preserve them for inspection "
            "or move them aside before starting a fresh extraction"
        )
    if all(checkpoint_exists) and not resume:
        raise ValueError(
            "resume checkpoint already exists; rerun with --resume to continue"
        )
    if any(final_exists) and not all(checkpoint_exists):
        raise ValueError(
            "a partial final output exists without a complete resume checkpoint; "
            "preserve it for inspection before proceeding"
        )

    signature = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_shape = (len(samples), EMBEDDING_DIMENSION)
    if all(checkpoint_exists):
        with config_path.open("r", encoding="utf-8") as json_file:
            saved_config = json.load(json_file)
        if saved_config.get("signature") != signature:
            raise ValueError(
                "resume configuration does not match current data/model/settings; "
                "refusing to reuse checkpoint"
            )
        embeddings = np.load(embeddings_path, mmap_mode="r+", allow_pickle=False)
        if embeddings.shape != expected_shape or embeddings.dtype != np.float32:
            raise ValueError(
                f"resume embedding store has incompatible shape/dtype: "
                f"{embeddings.shape}/{embeddings.dtype}"
            )
        completed: set[int] = set()
        with journal_path.open("r", encoding="utf-8") as journal:
            for line_number, line in enumerate(journal, start=1):
                record = json.loads(line)
                row_index = record.get("row_index")
                if not isinstance(row_index, int) or not 0 <= row_index < len(samples):
                    raise ValueError(f"invalid row_index in resume journal line {line_number}")
                if row_index in completed:
                    raise ValueError(f"duplicate row_index in resume journal: {row_index}")
                if any(record.get(key) != samples[row_index][key] for key in ("image_id", "image_path", "label")):
                    raise ValueError(
                        f"resume journal row {row_index} does not match current manifest"
                    )
                if not np.isfinite(embeddings[row_index]).all():
                    raise ValueError(
                        f"completed resume row {row_index} has non-finite features"
                    )
                completed.add(row_index)
        return embeddings, completed, journal_path, embeddings_path

    embeddings = np.lib.format.open_memmap(
        embeddings_path, mode="w+", dtype=np.float32, shape=expected_shape
    )
    embeddings[:] = np.nan
    embeddings.flush()
    with journal_path.open("w", encoding="utf-8", newline=""):
        pass
    atomic_write_json(config_path, {"signature": signature, "config": config})
    return embeddings, set(), journal_path, embeddings_path


def append_resume_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as journal:
        journal.write(json.dumps(record, sort_keys=True) + "\n")
        journal.flush()
        os.fsync(journal.fileno())


def write_failures(output_dir: Path, failures: list[dict[str, object]]) -> None:
    with (output_dir / "failures.csv").open("w", encoding="utf-8", newline="") as csv_file:
        fields = ("row_index", "image_id", "image_path", "label", "error")
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)


def write_delivery_readme(output_dir: Path, image_count: int) -> None:
    content = f"""# CRC-VAL-HE-7K MUSK Features

These are MUSK image embeddings extracted from the local `CRC-VAL-HE-7K` dataset.
The dataset is originally external validation data. These feature files do not
train a classifier and do not report model accuracy. If this dataset is used for
classifier training or tuning, reserve separate data for later evaluation; do
not describe the full dataset as an independent test set. Patient identifiers
were not inferred from image filenames.

## Files and Alignment

- `embeddings.npy`: float32 array with shape `({image_count}, 1024)`.
- `metadata.csv`: row index, image ID, path relative to the dataset root, and class label.
- `extraction_config.json`: model/checkpoint fingerprint, preprocessing, options, versions, and class counts.
- `failures.csv`: failed-image report; zero data rows means no failures.

Embedding row `i` corresponds exactly to metadata row with `row_index == i`.
The final output is published only when all images have finite feature values.

## Loading

```python
import csv
import numpy as np

embeddings = np.load("features/crc_val_musk/embeddings.npy")
with open("features/crc_val_musk/metadata.csv", newline="", encoding="utf-8") as f:
    metadata = list(csv.DictReader(f))
assert embeddings.shape[0] == len(metadata)
assert all(int(row["row_index"]) == i for i, row in enumerate(metadata))
```

This extraction is a feature handoff only. No automatic data split or classifier
training is performed.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def extract_embeddings(
    samples: list[dict[str, str]],
    data_dir: Path,
    output_dir: Path,
    *,
    full_dataset: bool,
    resume: bool,
) -> None:
    """Extract rows into resumable fixed indices, then publish aligned deliverables."""
    started_at = time.perf_counter()
    model, transform, versions, model_cache = load_model_and_transform()
    config = write_config(
        output_dir, data_dir, samples, versions, model_cache, full_dataset
    )
    embeddings, completed, journal_path, embeddings_path = prepare_resume_state(
        output_dir, samples, config, resume
    )

    failures = []
    pending = (
        (row_index, sample)
        for row_index, sample in enumerate(samples)
        if row_index not in completed
    )
    for row_index, sample in tqdm(
        pending,
        total=len(samples),
        initial=len(completed),
        desc="Extracting MUSK embeddings",
        unit="image",
    ):
        image_path = data_dir / Path(sample["image_path"])
        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(
                device="cuda", dtype=torch.float16
            )
            with torch.inference_mode():
                embedding = model(
                    image=image_tensor,
                    with_head=False,
                    out_norm=True,
                    ms_aug=False,
                    return_global=True,
                )[0]

            embedding_row = embedding[0].float().cpu().numpy()
            if embedding_row.shape != (EMBEDDING_DIMENSION,):
                raise ValueError(
                    f"expected embedding shape ({EMBEDDING_DIMENSION},), "
                    f"got {embedding_row.shape}"
                )
            if not np.isfinite(embedding_row).all():
                raise ValueError("embedding contains non-finite values")
            embeddings[row_index] = embedding_row
            embeddings.flush()
            append_resume_record(
                journal_path,
                {"row_index": row_index, **sample},
            )
            completed.add(row_index)
        except Exception as error:
            failures.append({"row_index": row_index, **sample, "error": str(error)})

    elapsed = time.perf_counter() - started_at
    print(f"Extraction elapsed time: {elapsed:.1f} seconds")
    if failures:
        write_failures(output_dir, failures)
        print(
            f"Run incomplete: {len(failures)} image(s) failed; "
            "final embeddings.npy and metadata.csv were not published.",
            file=sys.stderr,
        )
        for failure in failures:
            print(
                f"  row {failure['row_index']} ({failure['image_path']}): "
                f"{failure['error']}",
                file=sys.stderr,
            )
        raise RuntimeError(
            "incomplete extraction; fix or inspect failed images and rerun "
            "with --resume"
        )

    expected_shape = (len(samples), EMBEDDING_DIMENSION)
    if embeddings.shape != expected_shape or len(completed) != len(samples):
        raise RuntimeError(
            "internal alignment check failed: "
            f"embeddings={embeddings.shape}, completed_rows={len(completed)}"
        )
    if not np.isfinite(embeddings).all():
        raise RuntimeError("final embeddings contain non-finite values")
    image_ids = [sample["image_id"] for sample in samples]
    if len(image_ids) != len(set(image_ids)):
        raise RuntimeError("duplicate image IDs detected before final write")

    temporary_embeddings = output_dir / ".embeddings.npy.tmp"
    with temporary_embeddings.open("wb") as binary_file:
        np.save(binary_file, np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
        binary_file.flush()
        os.fsync(binary_file.fileno())
    metadata_path = output_dir / "metadata.csv"
    temporary_metadata = output_dir / ".metadata.csv.tmp"
    with temporary_metadata.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=("row_index", "image_id", "image_path", "label")
        )
        writer.writeheader()
        for row_index, sample in enumerate(samples):
            writer.writerow({"row_index": row_index, **sample})
        csv_file.flush()
        os.fsync(csv_file.fileno())

    os.replace(temporary_embeddings, output_dir / "embeddings.npy")
    os.replace(temporary_metadata, metadata_path)
    write_failures(output_dir, [])
    write_delivery_readme(output_dir, len(samples))
    final_config = {"created_utc": datetime.now(timezone.utc).isoformat(), **config}
    with (output_dir / "extraction_config.json").open(
        "w", encoding="utf-8"
    ) as json_file:
        json.dump(final_config, json_file, indent=2)
        json_file.write("\n")
    embeddings.flush()
    if isinstance(embeddings, np.memmap):
        embeddings._mmap.close()
    for checkpoint_path in (
        embeddings_path,
        journal_path,
        output_dir / RESUME_CONFIG,
    ):
        checkpoint_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a reproducible MUSK development sample embedding set."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("CRC-VAL-HE-7K"),
        help="dataset root containing the nine class folders",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/musk_crc_demo"),
        help="directory for manifest, embeddings, metadata, and config",
    )
    parser.add_argument(
        "--per-class", type=int,
        help="sample this many images per class (default mode; default count: 20)",
    )
    parser.add_argument(
        "--all-images", action="store_true",
        help="extract every .tif/.tiff under all nine classes in fixed order",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="persist completed rows and resume only with an identical config",
    )
    arguments = parser.parse_args()

    try:
        if arguments.all_images and arguments.per_class is not None:
            raise ValueError("--all-images and --per-class cannot be used together")
        full_dataset = arguments.all_images
        per_class = 20 if arguments.per_class is None else arguments.per_class
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        samples = (
            list_all_images(arguments.data_dir)
            if full_dataset
            else sample_images(arguments.data_dir, per_class)
        )
        save_manifest(samples, arguments.output_dir)
        print(
            f"Saved deterministic sample manifest: "
            f"{arguments.output_dir / 'sample_manifest.csv'} ({len(samples)} images)"
        )
        extract_embeddings(
            samples,
            arguments.data_dir,
            arguments.output_dir,
            full_dataset=full_dataset,
            resume=arguments.resume,
        )
    except KeyboardInterrupt:
        print(
            "Interrupted. Completed rows are checkpointed; rerun the same command "
            "with --resume to continue.",
            file=sys.stderr,
        )
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Saved embeddings and metadata to {arguments.output_dir}")
    print(
        "CRC-VAL-HE-7K is external validation data. If used for training/tuning, "
        "reserve separate data for evaluation; do not call the full dataset "
        "an independent test set. No classifier was trained."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())