"""Check the slide heatmap against expert annotations (BACH training slides A01-A10). Evaluation only:
nothing is trained on these slides.

Each patch takes the label of the annotated region under its center (Aperio ImageScope polygons; invasive drawn
over in situ over benign). Tissue outside every region is 'unannotated' and counted as not cancer, which is how
BACH defines its slide labels, though an annotator may have missed small lesions.

    python -m src.slide_eval --slides A01 A08 A09
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score

from src.bach import DEFAULT_THRESHOLD, MayumiClassifier, QuiltEmbedder
from src.patches import PatchLibrary
from src.slides import ANNOTATION_RGB, process_wsi, score, smooth

REPO = Path(__file__).resolve().parent.parent
LABELS = {"benign": "Benign", "carcinoma in situ": "InSitu", "in situ carcinoma": "InSitu", "invasive carcinoma": "Invasive"}
CODES = {"Unannotated": 0, "Benign": 1, "InSitu": 2, "Invasive": 3}
SCALE = 16  # annotation mask at 1/16 of the slide


def regions(xml: Path) -> list[tuple[str, np.ndarray]]:
    out = []
    for r in ET.parse(xml).getroot().iter("Region"):
        a = r.find("Attributes/Attribute")
        raw = (a.get("Value") if a is not None and a.get("Value") else r.get("Text") or "").strip().lower()
        if raw in LABELS:
            pts = np.array([(float(v.get("X")), float(v.get("Y"))) for v in r.iter("Vertex")])
            if len(pts) >= 3:
                out.append((LABELS[raw], pts))
    return out


def label_mask(xml: Path, width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width // SCALE + 1, height // SCALE + 1), 0)
    draw = ImageDraw.Draw(mask)
    for name in ("Benign", "InSitu", "Invasive"):  # later classes win where regions overlap
        for label, pts in regions(xml):
            if label == name:
                draw.polygon([tuple(p) for p in pts / SCALE], fill=CODES[name])
    return np.asarray(mask)


def evaluate(folder: Path, xml: Path, classifiers: dict, lib: PatchLibrary, threshold: float) -> dict:
    meta = json.loads((folder / "meta.json").read_text())
    z = np.load(folder / "grid.npz")
    t = meta["tile"]
    mask = label_mask(xml, meta["width"], meta["height"])
    cx = np.clip(((z["x"] + t / 2) / SCALE).astype(int), 0, mask.shape[1] - 1)
    cy = np.clip(((z["y"] + t / 2) / SCALE).astype(int), 0, mask.shape[0] - 1)
    truth = mask[cy, cx]
    cancer = truth >= CODES["InSitu"]
    res = {"slide": folder.name, "patches": int(len(truth)),
           "truth": {k: int((truth == v).sum()) for k, v in CODES.items()}}
    both = 0 < cancer.sum() < len(cancer)
    for name, clf in classifiers.items():
        s = score(z["emb"].astype(np.float32), clf, lib, threshold)
        flag = s["abnormal"] >= 0.5
        sm = smooth({"x": z["x"], "y": z["y"], "tile": t, "abnormal": s["abnormal"]})
        top = np.argsort(-sm)[: int(0.2 * len(sm))]
        res[name] = {
            "auc_cancer": round(float(roc_auc_score(cancer, s["abnormal"])), 3) if both else None,
            "auc_cancer_smoothed": round(float(roc_auc_score(cancer, sm)), 3) if both else None,
            "top20_smoothed_cancer_share": round(float(cancer[top].mean()), 3),
            "cancer_found": round(float(flag[cancer].mean()), 3),               # sensitivity
            "flagged_that_are_cancer": round(float(cancer[flag].mean()), 3) if flag.any() else None,  # precision
            "noncancer_flagged": round(float(flag[~cancer].mean()), 3),         # false-positive rate
            "flagged_share": round(float(flag.mean()), 3),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slides", nargs="+", default=["A01", "A08", "A09"])
    ap.add_argument("--raw", type=Path, default=REPO / "data/bach_wsi_annotated")
    ap.add_argument("--data", type=Path, default=REPO / "ui/data_bach")
    ap.add_argument("--out", type=Path, default=REPO / "data/slide_eval")
    ap.add_argument("--patch-model", type=Path, default=REPO / "data/mayumi_patch_model/bach_single_patch_classifier.pkl")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--normalize", action="store_true", help="Reinhard color normalization before embedding")
    ap.add_argument("--export", action="store_true", help="also copy the slides into the UI data folder")
    ap.add_argument("--validation", type=Path, nargs=2, metavar=("TUNED", "HELD_OUT"),
                    help="only write <data>/slides/validation.json from two results.json files, then stop")
    args = ap.parse_args()
    if args.validation:
        for row in write_validation(*args.validation, args.data):
            print(row)
        return
    lib = PatchLibrary(args.data)
    classifiers = {"photo_model": MayumiClassifier(args.data / "classifier.pkl"),
                   "patch_model": MayumiClassifier(args.patch_model)}
    embedder = None
    results = []
    for name in args.slides:
        folder = args.out / name
        if not (folder / "grid.npz").exists():  # embed once; scoring is cheap
            embedder = embedder or QuiltEmbedder(device=args.device)
            process_wsi(args.raw / f"{name}.svs", args.data, embedder, DEFAULT_THRESHOLD, out_root=args.out,
                        normalize=args.normalize)
        r = evaluate(folder, args.raw / f"{name}.xml", classifiers, lib, DEFAULT_THRESHOLD)
        results.append(r)
        print(json.dumps(r), flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    if args.export:
        for name in args.slides:
            v = export_to_ui(name, args.out, args.raw, args.data, classifiers["patch_model"], lib, DEFAULT_THRESHOLD)
            print(f"exported {name} to {args.data / 'slides' / name}: {v}", flush=True)



# ---- Export annotated slides to the UI (display and evaluation only; never trained on) ----

def export_to_ui(name: str, eval_dir: Path, raw: Path, data_dir: Path, clf: MayumiClassifier, lib: PatchLibrary,
                 threshold: float) -> dict:
    """Copy an evaluated slide into <data>/slides/: grid re-scored with `clf`, an expert-annotation overlay, and the
    per-slide numbers the page quotes (cancer share, AUC, cancer share in the most suspicious 20%)."""
    import os

    src, dst = eval_dir / name, data_dir / "slides" / name
    dst.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text())
    z = np.load(src / "grid.npz")
    grid = {k: z[k] for k in z.files}
    s = score(grid["emb"].astype(np.float32), clf, lib, threshold)
    grid.update({k: v for k, v in s.items() if isinstance(v, np.ndarray)})
    t = meta["tile"]
    grid["suspicion"] = smooth({**grid, "tile": t})
    mask = label_mask(raw / f"{name}.xml", meta["width"], meta["height"])
    truth = mask[np.clip(((grid["y"] + t / 2) / SCALE).astype(int), 0, mask.shape[0] - 1),
                 np.clip(((grid["x"] + t / 2) / SCALE).astype(int), 0, mask.shape[1] - 1)]
    grid["truth"] = truth
    np.savez_compressed(dst / "grid.npz", **grid)

    base = Image.open(src / "overview.jpg").convert("RGB")
    base.save(dst / "overview.jpg", quality=90)
    sc = base.width / meta["width"]
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for label in ("Benign", "InSitu", "Invasive"):
        for lab, pts in regions(raw / f"{name}.xml"):
            if lab == label:
                draw.polygon([tuple(p) for p in pts * sc], fill=ANNOTATION_RGB[label] + (80,),
                             outline=ANNOTATION_RGB[label] + (255,))
    overlay.save(dst / "annotation.png")
    if not (dst / "slide.svs").exists():
        os.link(raw / f"{name}.svs", dst / "slide.svs")

    cancer = truth >= CODES["InSitu"]
    top = np.argsort(-grid["suspicion"])[: int(0.2 * len(truth))]
    meta.update({"classifier": "single-patch", "annotated": True,
                 "validation": {"cancer_share": round(float(cancer.mean()), 3),
                                "auc": round(float(roc_auc_score(cancer, grid["suspicion"])), 3),
                                "auc_unsmoothed": round(float(roc_auc_score(cancer, grid["abnormal"])), 3),
                                "top20_cancer_share": round(float(cancer[top].mean()), 3)}})
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta["validation"]


def write_validation(tuned: Path, held_out: Path, data_dir: Path) -> list[dict]:
    """<data>/slides/validation.json for the Evidence page: one row per annotated slide, from the results.json of
    the slides the settings were chosen on and of the slides checked afterwards without changes."""
    rows = []
    for role, path in (("Chosen on", tuned), ("Unseen check", held_out)):
        for r in json.loads(path.read_text()):
            t, new, old = r["truth"], r["patch_model"], r["photo_model"]
            rows.append({"slide": r["slide"], "role": role, "patches": r["patches"],
                         "cancer_share": round((t["InSitu"] + t["Invasive"]) / r["patches"], 3),
                         "top20_cancer_share": new["top20_smoothed_cancer_share"],
                         "auc": new["auc_cancer_smoothed"], "auc_unsmoothed": new["auc_cancer"],
                         "auc_photo_model": old["auc_cancer"]})
    (data_dir / "slides" / "validation.json").write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    main()
