"""Slide view: cut a whole-slide image (display only, never used for training) or a large photo into patches,
score every patch with the same chain as a photo, and keep a grid for a heatmap and click-to-inspect.

Per patch: QuiltNet embedding -> Mayumi's single-patch classifier (<data>/patch_classifier.pkl, trained on 224 px
patches with labels inherited from their photos; her photo-level classifier is the fallback) -> Ashley's checks
(confidence threshold; distance to the 5th nearest train patch).
Patch size is physical: a BACH photo patch is 224 px at 0.42 um/px (94 um), so a slide at 0.5 um/px is read in
188 px squares and resized to 224 by the model's preprocessing.

    python -m src.slides wsi --svs data/bach_wsi/test8.svs --data ui/data_bach     # offline, minutes per slide
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.bach import DEFAULT_THRESHOLD, MayumiClassifier, QuiltEmbedder
from src.patches import PatchLibrary, embed_crops
from src.stain import lab_stats, rgb_to_lab, lab_to_rgb

REPO = Path(__file__).resolve().parent.parent
PHOTO_MPP = 0.42                    # BACH microscopy photos, um per pixel
PATCH_UM = 224 * PHOTO_MPP          # one patch covers the same tissue on photos and slides
THUMB_WIDTH = 1600                  # overview image shown in the UI
ABNORMAL = ("InSitu", "Invasive")   # summed for the 'likely abnormal' layer (BACH classes)


def patch_classifier(data_dir: Path) -> MayumiClassifier:
    """Mayumi's single-patch model when present (10/18), else her photo-level model applied to single patches."""
    path = data_dir / "patch_classifier.pkl"
    return MayumiClassifier(path if path.exists() else data_dir / "classifier.pkl")


def rescore(data_dir: Path, threshold: float) -> None:
    """New classifier, same embeddings: update every slide's grid without re-reading the slide."""
    clf, lib = patch_classifier(data_dir), PatchLibrary(data_dir)
    for folder in sorted((data_dir / "slides").glob("*/")):
        z = np.load(folder / "grid.npz")
        grid = {k: z[k] for k in z.files}
        s = score(grid["emb"].astype(np.float32), clf, lib, threshold)
        grid.update({k: v for k, v in s.items() if isinstance(v, np.ndarray)})
        np.savez_compressed(folder / "grid.npz", **grid)
        meta = json.loads((folder / "meta.json").read_text())
        meta["classifier"] = "single-patch" if (data_dir / "patch_classifier.pkl").exists() else "photo-level"
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{folder.name}: likely abnormal {np.mean(s['abnormal'] >= 0.5):.0%}, low confidence {np.mean(s['low_conf']):.0%}, "
              f"unfamiliar {np.mean(s['unfamiliar']):.0%} ({meta['classifier']} classifier)", flush=True)


def score(emb: np.ndarray, clf: MayumiClassifier, lib: PatchLibrary, threshold: float) -> dict:
    """Mayumi's classifier and Ashley's two checks for many patch embeddings at once."""
    probs = clf.model.predict_proba(emb)
    conf = probs.max(axis=1)
    R = lib.gate.index.normalized_embeddings  # Ashley's reference: train patches, L2-normalized
    q = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    k = lib.gate.config["k"]
    dist = np.empty(len(q))
    for i in range(0, len(q), 2048):  # distance to the k-th nearest reference patch, as reference_support.py
        sims = q[i:i + 2048] @ R.T
        dist[i:i + 2048] = 1 - np.sort(sims, axis=1)[:, -k]
    limit = lib.gate.config["threshold"]
    abnormal = probs[:, [clf.classes.index(c) for c in ABNORMAL if c in clf.classes]].sum(axis=1)
    return {"probs": probs.astype(np.float32), "conf": conf.astype(np.float32), "abnormal": abnormal.astype(np.float32),
            "low_conf": conf < threshold, "unfamiliar": dist > limit, "distance": dist.astype(np.float32),
            "classes": np.array(clf.classes), "limit": limit, "threshold": threshold}


def tissue_mask(thumb: Image.Image) -> np.ndarray:
    """Tissue = saturated and not white (Otsu on HSV saturation, H&E is colored, the glass is not)."""
    hsv = np.asarray(thumb.convert("HSV"), dtype=np.float32)
    sat, val = hsv[..., 1], hsv[..., 2]
    hist, edges = np.histogram(sat, bins=256, range=(0, 255))
    w = np.cumsum(hist); mu = np.cumsum(hist * edges[:-1]); total_w, total_mu = w[-1], mu[-1]
    between = (total_mu * w - mu * total_w) ** 2 / np.maximum(w * (total_w - w), 1)
    t = edges[int(np.argmax(between))]
    return (sat > max(t, 20)) & (val > 40)


class SlideColor:
    """Per-slide Reinhard: one LAB mean/std shift from this slide's tissue to the BACH photo patches.

    Applied only before embedding (scanner colors differ from the microscope camera); the UI shows original colors.
    """

    def __init__(self, src_mean, src_std, ref_mean, ref_std) -> None:
        self.src_mean, self.src_std = np.asarray(src_mean), np.asarray(src_std)
        self.ref_mean, self.ref_std = np.asarray(ref_mean), np.asarray(ref_std)

    def __call__(self, im: Image.Image) -> Image.Image:
        lab = rgb_to_lab(np.asarray(im.convert("RGB")))
        return Image.fromarray(lab_to_rgb((lab - self.src_mean) / self.src_std * self.ref_std + self.ref_mean))


def photo_color_reference(data_dir: Path, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """LAB statistics of training photo patches (the color the models learned from)."""
    crops = sorted((data_dir / "patches" / "crops").glob("*.jpg"))
    pick = [crops[i] for i in np.random.default_rng(0).choice(len(crops), min(n, len(crops)), replace=False)]
    return lab_stats([Image.open(c) for c in pick])


def process_wsi(svs: Path, data_dir: Path, embedder: QuiltEmbedder, threshold: float, batch: int = 128,
                out_root: Path | None = None, normalize: bool = False) -> Path:
    """normalize=False by default: on annotated slide A01, Reinhard normalization lowered the cancer-vs-rest AUC
    of both classifiers (single-patch model 0.66 -> 0.59), so slides are embedded with their scanned colors."""
    import openslide

    slide = openslide.OpenSlide(str(svs))
    W, H = slide.dimensions
    mpp = float(slide.properties.get("openslide.mpp-x", 0.5))
    tile = round(PATCH_UM / mpp)
    thumb = slide.get_thumbnail((THUMB_WIDTH, round(THUMB_WIDTH * H / W))).convert("RGB")
    sx, sy = thumb.width / W, thumb.height / H
    mask = tissue_mask(thumb)
    xs, ys = [], []
    for y in range(0, H - tile + 1, tile):
        for x in range(0, W - tile + 1, tile):
            cell = mask[int(y * sy):max(int((y + tile) * sy), int(y * sy) + 1), int(x * sx):max(int((x + tile) * sx), int(x * sx) + 1)]
            if cell.mean() > 0.5:
                xs.append(x); ys.append(y)
    print(f"{svs.name}: {W}x{H} at {mpp} um/px, tile {tile}px, {len(xs)} tissue patches", flush=True)

    rng = np.random.default_rng(0)
    sample = rng.choice(len(xs), min(400, len(xs)), replace=False)
    src_mean, src_std = lab_stats([slide.read_region((xs[i], ys[i]), 0, (tile, tile)) for i in sample])
    ref_mean, ref_std = photo_color_reference(data_dir)
    color = SlideColor(src_mean, src_std, ref_mean, ref_std) if normalize else (lambda im: im.convert("RGB"))

    t0, embs = time.time(), []
    for i in range(0, len(xs), batch):
        crops = [color(slide.read_region((x, y), 0, (tile, tile))) for x, y in zip(xs[i:i + batch], ys[i:i + batch])]
        embs.append(embed_crops(embedder, crops, batch=batch))
        if (i // batch) % 20 == 0:
            print(f"  {i + len(crops)}/{len(xs)} patches, {time.time() - t0:.0f}s", flush=True)
    emb = np.concatenate(embs)
    lib = PatchLibrary(data_dir)
    s = score(emb, patch_classifier(data_dir), lib, threshold)

    out = (out_root or data_dir / "slides") / svs.stem
    out.mkdir(parents=True, exist_ok=True)
    thumb.save(out / "overview.jpg", quality=90)
    link = out / "slide.svs"  # the app reads the clicked patch from the slide itself (hard link: no extra disk)
    if not link.exists():
        os.link(svs, link)
    np.savez_compressed(out / "grid.npz", x=np.array(xs), y=np.array(ys), emb=emb.astype(np.float16),
                        **{k: v for k, v in s.items() if isinstance(v, np.ndarray)})
    meta = {"name": svs.stem, "kind": "slide", "source": svs.name, "width": W, "height": H, "mpp": mpp, "tile": tile,
            "n": len(xs), "limit": s["limit"], "threshold": threshold, "seconds": round(time.time() - t0),
            "color": {"normalized": normalize, "src_mean": src_mean.tolist(), "src_std": src_std.tolist(),
                      "ref_mean": ref_mean.tolist(), "ref_std": ref_std.tolist()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  done in {meta['seconds']}s: likely abnormal (>=0.5) {np.mean(s['abnormal'] >= 0.5):.0%}, "
          f"low confidence {np.mean(s['low_conf']):.0%}, unfamiliar {np.mean(s['unfamiliar']):.0%}", flush=True)
    return out


def photo_grid(img: Image.Image, stride: int = 112) -> tuple[list[int], list[int]]:
    """Overlapping 224 px patches over a photo (native scale, as Mayumi's patches)."""
    W, H = img.size
    xs = list(range(0, W - 224 + 1, stride)) or [0]
    ys = list(range(0, H - 224 + 1, stride)) or [0]
    return [x for y in ys for x in xs], [y for y in ys for x in xs]


def analyze_photo(img: Image.Image, embedder: QuiltEmbedder, clf: MayumiClassifier, lib: PatchLibrary,
                  threshold: float) -> dict:
    """Live: a 2048x1536 photo gives about 200 overlapping patches (a few seconds on CPU)."""
    img = img.convert("RGB")
    xs, ys = photo_grid(img)
    emb = embed_crops(embedder, [img.crop((x, y, x + 224, y + 224)) for x, y in zip(xs, ys)])
    return {"x": np.array(xs), "y": np.array(ys), "emb": emb, "tile": 224, **score(emb, clf, lib, threshold)}


# ---- Heatmap and click mapping (shared by the app) ----

ABNORMAL_RGB = (13, 148, 136)    # teal: stands out on pink/purple H&E
LOW_CONF_RGB = (217, 119, 6)     # amber: the app's review color
UNFAMILIAR_RGB = (100, 116, 139)  # slate: the AI gives no class here


def cells(grid: dict, scale: float, stride: int | None = None) -> np.ndarray:
    """Display-space box of each patch. Overlapping photo patches draw only their central stride x stride cell."""
    x, y, t = grid["x"].astype(float), grid["y"].astype(float), float(grid["tile"])
    if stride:
        off = (t - stride) / 2
        x, y, t = x + off, y + off, float(stride)
    return np.stack([x, y, x + t, y + t], axis=1) * scale


WIDE_UNFAMILIAR = 0.8  # share of tissue patches above which unfamiliarity is a property of the whole slide


def wide_unfamiliar(grid: dict) -> bool:
    return float(np.mean(grid["unfamiliar"])) > WIDE_UNFAMILIAR


ANNOTATION_RGB = {"Benign": (37, 99, 235), "InSitu": (234, 179, 8), "Invasive": (220, 38, 38)}  # expert layer
TOP_SHARE = 0.2  # slides: color the most suspicious 20% of the tissue (validated on annotated slides)


SMOOTH = 5  # slides: average 'abnormal' over 5 x 5 neighboring patches (~0.5 mm). Fixed on A01/A08/A09 before
            # checking it on unseen annotated slides; it raised the cancer-vs-rest AUC on all three.


def smooth(grid: dict, size: int = SMOOTH) -> np.ndarray:
    """Mean 'abnormal' score over the size x size patch neighborhood (tissue patches only), for a slide grid whose
    patches tile the slide without overlap."""
    from scipy import ndimage

    x, y = grid["x"], grid["y"]
    step = int(grid["tile"])
    col, row = (x - x.min()) // step, (y - y.min()) // step
    total = np.zeros((row.max() + 1, col.max() + 1))
    count = np.zeros_like(total)
    total[row, col], count[row, col] = grid["abnormal"], 1
    mean = ndimage.uniform_filter(total, size, mode="constant") / np.maximum(
        ndimage.uniform_filter(count, size, mode="constant"), 1e-9)
    return mean[row, col]


def suspicion_rank(grid: dict) -> np.ndarray:
    """Percentile of each patch's suspicion within its own slide (0 = least, 1 = most suspicious): the smoothed
    score when the grid has one, else 'abnormal'."""
    order = np.argsort(grid.get("suspicion", grid["abnormal"]), kind="stable")
    rank = np.empty(len(order))
    rank[order] = np.arange(len(order)) / max(len(order) - 1, 1)
    return rank


def render(base: Image.Image, grid: dict, layer: str, scale: float, stride: int | None = None,
           highlight: int | None = None, annotation: Image.Image | None = None) -> Image.Image:
    """Base image with one translucent cell per patch.
    'abnormal': teal; on slides (grid has 'rank') the most suspicious 20%, darker = more suspicious; on photos the
    patches scored >= 0.5 where the tissue is familiar. 'unsure': amber for low confidence, slate for unfamiliar
    tissue. 'annotation': the expert-annotation overlay as drawn."""
    if layer == "annotation" and annotation is not None:
        out = Image.alpha_composite(base.convert("RGBA"), annotation.convert("RGBA"))
        return _marker(out, cells(grid, scale, stride), highlight, base.width).convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rank = grid.get("rank")
    boxes = cells(grid, scale, stride)
    # When nearly all of it is unfamiliar, that is said once for the whole slide (app banner), not cell by cell.
    unfamiliar = np.zeros(len(boxes), bool) if wide_unfamiliar(grid) else grid["unfamiliar"]
    low = grid["low_conf"]
    for i, b in enumerate(boxes):
        if layer == "abnormal":
            if rank is not None:
                if rank[i] >= 1 - TOP_SHARE:
                    draw.rectangle(tuple(b), fill=ABNORMAL_RGB + (int(70 + 150 * (rank[i] - 1 + TOP_SHARE) / TOP_SHARE),))
            elif unfamiliar[i]:
                draw.rectangle(tuple(b), fill=UNFAMILIAR_RGB + (70,))
            elif grid["abnormal"][i] >= 0.5:
                draw.rectangle(tuple(b), fill=ABNORMAL_RGB + (int(90 + 120 * (grid["abnormal"][i] - 0.5) / 0.5),))
        elif unfamiliar[i]:
            draw.rectangle(tuple(b), fill=UNFAMILIAR_RGB + (110,))
        elif low[i]:
            draw.rectangle(tuple(b), fill=LOW_CONF_RGB + (115,))
    out = Image.alpha_composite(base.convert("RGBA"), overlay)
    return _marker(out, boxes, highlight, base.width).convert("RGB")


def _marker(out: Image.Image, boxes: np.ndarray, highlight: int | None, width: int) -> Image.Image:
    if highlight is not None:  # the image is shown at ~40% of its width: size the marker for that
        b = boxes[highlight]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        k = width / 640  # marker sizes are in on-screen pixels
        r = max((b[2] - b[0]) / 2 + 3 * k, 12 * k)
        d = ImageDraw.Draw(out)
        d.rectangle((cx - r, cy - r, cx + r, cy + r), outline=(15, 23, 42, 255), width=max(round(2.5 * k), 2))
        inner = r - 2.5 * k
        d.rectangle((cx - inner, cy - inner, cx + inner, cy + inner), outline=(255, 255, 255, 255), width=max(round(k), 1))
    return out


def patch_at(grid: dict, x: float, y: float) -> int | None:
    """Index of the patch under a point in full-resolution coordinates (the last one drawn wins on overlap)."""
    t = grid["tile"]
    hit = np.flatnonzero((grid["x"] <= x) & (x < grid["x"] + t) & (grid["y"] <= y) & (y < grid["y"] + t))
    if not len(hit):
        return None
    cx, cy = grid["x"][hit] + t / 2, grid["y"][hit] + t / 2  # overlapping photo patches: the most centered one
    return int(hit[np.argmin((cx - x) ** 2 + (cy - y) ** 2)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wsi")
    w.add_argument("--svs", type=Path, nargs="+", required=True)
    w.add_argument("--data", type=Path, default=REPO / "ui/data_bach")
    w.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    w.add_argument("--device", default="mps")
    r = sub.add_parser("rescore", help="re-score stored slide grids with the current classifier (no slide reading)")
    r.add_argument("--data", type=Path, default=REPO / "ui/data_bach")
    r.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()
    if args.cmd == "rescore":
        rescore(args.data, args.threshold)
        return
    embedder = QuiltEmbedder(device=args.device)
    for svs in args.svs:
        process_wsi(svs, args.data, embedder, args.threshold)


if __name__ == "__main__":
    main()
