"""Run a small MUSK zero-shot classification demo on one image per CRC class.

These nine images are only a workflow check, not a formal evaluation set.
Similarity values are cosine similarities, not probabilities.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, TypedDict

import PIL
import torch
import torchvision
from PIL import Image
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model
from torchvision import transforms
from tqdm import tqdm
from transformers import XLMRobertaTokenizer


CLASS_NAMES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
CLASS_DESCRIPTIONS = {
    "ADI": "histopathology image of adipose tissue",
    "BACK": "histopathology image of background tissue",
    "DEB": "histopathology image of tissue debris",
    "LYM": "histopathology image of lymphocytes",
    "MUC": "histopathology image of mucus",
    "MUS": "histopathology image of muscle tissue",
    "NORM": "histopathology image of normal colon mucosa",
    "STR": "histopathology image of colorectal tumor stroma",
    "TUM": "histopathology image of colorectal adenocarcinoma tissue",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MODEL_NAME = "musk_large_patch16_384"
MODEL_CHECKPOINT = "hf_hub:xiangjx/musk"
MUSK_REPOSITORY = Path(r"C:\Users\Jiayi\MUSK")
IMAGE_SIZE = 384
TEXT_MAX_LENGTH = 100
EMBEDDING_DIMENSION = 1024


class ImageSample(TypedDict):
    image_id: str
    true_label: str
    image_path: str


class PredictionResult(TypedDict):
    image_id: str
    true_label: str
    predicted_label: str


def _cosine_similarity_columns() -> tuple[str, ...]:
    return tuple(f"cosine_similarity_{label}" for label in CLASS_NAMES)


def select_one_image_per_class(data_dir: Path) -> list[ImageSample]:
    """Select the first supported image by sorted path from each class folder."""
    if not data_dir.is_dir():
        raise ValueError(f"dataset directory does not exist: {data_dir}")

    samples = []
    for label in CLASS_NAMES:
        class_dir = data_dir / label
        if not class_dir.is_dir():
            raise ValueError(f"class directory does not exist: {class_dir}")
        candidates = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not candidates:
            raise ValueError(
                f"no supported images found in class directory: {class_dir}"
            )
        image_path = candidates[0]
        samples.append(
            {
                "image_id": image_path.stem,
                "true_label": label,
                "image_path": image_path.relative_to(data_dir).as_posix(),
            }
        )
    return samples


def load_model_and_transform() -> tuple[Any, Any, Any, Any, dict[str, str]]:
    """Load the cached MUSK model once, with official zero-shot preprocessing."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo but is unavailable")
    if not MUSK_REPOSITORY.is_dir():
        raise RuntimeError(f"MUSK repository not found: {MUSK_REPOSITORY}")

    model_cache = Path.home() / ".cache" / "model.safetensors"
    if not model_cache.is_file():
        raise RuntimeError(
            "local MUSK checkpoint is missing; refusing to download it: "
            f"{model_cache}"
        )
    tokenizer_path = MUSK_REPOSITORY / "musk" / "models" / "tokenizer.spm"
    if not tokenizer_path.is_file():
        raise RuntimeError(f"MUSK tokenizer not found: {tokenizer_path}")

    sys.path.insert(0, str(MUSK_REPOSITORY))
    # Import modeling to register the MUSK architecture with timm.
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

    tokenizer = XLMRobertaTokenizer(str(tokenizer_path))
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE, interpolation=3, antialias=True),
            transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN,
                std=IMAGENET_INCEPTION_STD,
            ),
        ]
    )
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "musk": version("musk"),
        "transformers": version("transformers"),
        "pillow": PIL.__version__,
    }
    return model, tokenizer, transform, utils, versions


def encode_class_descriptions(
    model: Any, tokenizer: Any, musk_utils: Any
) -> torch.Tensor:
    """Encode candidate descriptions using MUSK's official XLM tokenizer path."""
    token_rows = []
    padding_rows = []
    for label in CLASS_NAMES:
        text = CLASS_DESCRIPTIONS[label]
        token_ids, padding_mask = musk_utils.xlm_tokenizer(
            text, tokenizer, max_len=TEXT_MAX_LENGTH
        )
        token_rows.append(torch.tensor(token_ids, dtype=torch.long).unsqueeze(0))
        padding_rows.append(torch.tensor(padding_mask, dtype=torch.long).unsqueeze(0))

    text_ids = torch.cat(token_rows, dim=0).to(device="cuda")
    padding_masks = torch.cat(padding_rows, dim=0).to(device="cuda")
    with torch.inference_mode():
        text_output = model(
            text_description=text_ids,
            padding_mask=padding_masks,
            with_head=True,
            out_norm=True,
            ms_aug=False,
            return_global=True,
        )[1]
    if text_output is None or text_output.shape != (
        len(CLASS_NAMES),
        EMBEDDING_DIMENSION,
    ):
        shape = None if text_output is None else tuple(text_output.shape)
        raise RuntimeError(f"unexpected text embedding shape: {shape}")
    return text_output


def classify_samples(
    model: Any,
    transform: Any,
    text_embeddings: torch.Tensor,
    samples: list[ImageSample],
    data_dir: Path,
) -> list[dict[str, object]]:
    """Classify each sample by cosine similarity to the nine text embeddings."""
    results = []
    for sample in tqdm(samples, desc="MUSK zero-shot demo", unit="image"):
        image_path = data_dir / Path(sample["image_path"])
        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(
                device="cuda", dtype=torch.float16
            )
        except Exception as error:
            raise RuntimeError(f"failed to read image {image_path}: {error}") from error

        with torch.inference_mode():
            image_output = model(
                image=image_tensor,
                with_head=True,
                out_norm=True,
                ms_aug=False,
                return_global=True,
            )[0]
            if image_output is None or image_output.shape != (
                1,
                EMBEDDING_DIMENSION,
            ):
                shape = None if image_output is None else tuple(image_output.shape)
                raise RuntimeError(
                    f"unexpected image embedding shape for {image_path}: {shape}"
                )
            cosine_similarities = image_output @ text_embeddings.T

        similarity_values = cosine_similarities[0].float().cpu().tolist()
        best_index = max(
            range(len(CLASS_NAMES)), key=similarity_values.__getitem__
        )
        results.append(
            {
                "image_id": sample["image_id"],
                "true_label": sample["true_label"],
                "predicted_label": CLASS_NAMES[best_index],
                **{
                    f"cosine_similarity_{label}": similarity_values[index]
                    for index, label in enumerate(CLASS_NAMES)
                },
            }
        )
    return results


def write_outputs(
    output_dir: Path,
    results: list[dict[str, object]],
    versions: dict[str, str],
    data_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        "image_id",
        "true_label",
        "predicted_label",
        *_cosine_similarity_columns(),
    )
    with (output_dir / "zero_shot_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    config = {
        "purpose": (
            "Nine-image workflow check only; not a formal evaluation set; "
            "do not interpret these examples as model performance."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(data_dir),
        "model": MODEL_NAME,
        "checkpoint": MODEL_CHECKPOINT,
        "checkpoint_cache": str(Path.home() / ".cache" / "model.safetensors"),
        "tokenizer": str(MUSK_REPOSITORY / "musk" / "models" / "tokenizer.spm"),
        "image_preprocessing": {
            "resize": IMAGE_SIZE,
            "interpolation_argument": 3,
            "antialias": True,
            "center_crop": [IMAGE_SIZE, IMAGE_SIZE],
            "normalize_mean": list(IMAGENET_INCEPTION_MEAN),
            "normalize_std": list(IMAGENET_INCEPTION_STD),
            "image_mode": "RGB",
        },
        "text_tokenization": {
            "tokenizer_class": "XLMRobertaTokenizer",
            "max_len": TEXT_MAX_LENGTH,
            "text_template": "{class description}",
        },
        "class_descriptions": CLASS_DESCRIPTIONS,
        "inference": {
            "with_head": True,
            "out_norm": True,
            "ms_aug": False,
            "return_global": True,
            "device": "cuda",
            "dtype": "float16",
            "image_batch_size": 1,
            "similarity": (
                "cosine similarity: dot product of L2-normalized image and "
                "text embeddings"
            ),
            "softmax_or_probability": False,
            "threshold_applied": False,
        },
        "sampling": {
            "images_per_class": 1,
            "selection": (
                "first supported image by sorted relative path in each class folder"
            ),
            "classes": list(CLASS_NAMES),
        },
        "software_versions": versions,
    }
    with (output_dir / "zero_shot_config.json").open(
        "w", encoding="utf-8"
    ) as json_file:
        json.dump(config, json_file, indent=2)
        json_file.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run MUSK zero-shot image classification on one image per class."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("CRC-VAL-HE-7K"),
        help="dataset root containing the nine class folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/musk_zero_shot_demo"),
        help="directory for demo CSV and inference config",
    )
    arguments = parser.parse_args()

    try:
        samples = select_one_image_per_class(arguments.data_dir)
        model, tokenizer, transform, musk_utils, versions = load_model_and_transform()
        text_embeddings = encode_class_descriptions(model, tokenizer, musk_utils)
        results = classify_samples(
            model, transform, text_embeddings, samples, arguments.data_dir
        )
        write_outputs(arguments.output_dir, results, versions, arguments.data_dir)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Saved nine-image zero-shot demo to {arguments.output_dir}")
    print("Cosine similarities are not probabilities; no threshold was applied.")
    print("These nine images are only a workflow check, not a formal evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())