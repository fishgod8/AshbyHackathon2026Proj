"""Generate mock data so the web UI runs before teammates' code is ready.

Outputs:
  mock_data/images/       fake tissue tiles (colored noise, placeholders only)
  mock_data/cases.json    per-case analysis results, in the team's agreed result format

When real results from Mayumi / Person 2 / Felix are ready, replace cases.json
with one in the same format; app.py does not need to change.
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

# The 9 tissue classes of CRC-100K
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

# A rough H&E tint per class, only so placeholder tiles look distinct
TINTS = {
    "ADI": (245, 235, 240), "BACK": (230, 230, 230), "DEB": (200, 120, 160),
    "LYM": (90, 60, 150), "MUC": (190, 170, 210), "MUS": (220, 110, 150),
    "NORM": (170, 100, 170), "STR": (230, 150, 180), "TUM": (130, 70, 160),
}

OUT = Path(__file__).parent / "mock_data"
IMG_DIR = OUT / "images"
rng = np.random.default_rng(0)
random.seed(0)


def make_tile(label: str, path: Path) -> str:
    base = np.array(TINTS[label], dtype=float)
    noise = rng.normal(0, 25, size=(224, 224, 3))
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return str(path.relative_to(OUT))


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Reference gallery: 5 "confirmed" tiles per class
    reference = []
    for c in CLASSES:
        for i in range(5):
            p = make_tile(c, IMG_DIR / f"ref_{c}_{i}.png")
            reference.append({"image": p, "label": c})

    # 8 test cases: confident, needs review, and model wrong (but flagged)
    scenarios = [
        ("case_01", "TUM", "TUM", 0.96, None),
        ("case_02", "LYM", "LYM", 0.93, None),
        ("case_03", "ADI", "ADI", 0.98, None),
        ("case_04", "STR", "TUM", 0.48, "STR"),   # model is wrong, but flagged
        ("case_05", "MUS", "MUS", 0.55, "STR"),
        ("case_06", "NORM", "NORM", 0.89, None),
        ("case_07", "DEB", "DEB", 0.51, "TUM"),
        ("case_08", "MUC", "MUC", 0.91, None),
    ]

    cases = []
    for case_id, truth, pred, top_p, runner_up in scenarios:
        img = make_tile(truth, IMG_DIR / f"{case_id}.png")

        probs = {pred: top_p}
        if runner_up:
            probs[runner_up] = round((1 - top_p) * 0.8, 3)
        rest = [c for c in CLASSES if c not in probs]
        leftover = 1 - sum(probs.values())
        weights = rng.dirichlet(np.ones(len(rest)))
        for c, w in zip(rest, weights):
            probs[c] = round(float(leftover * w), 3)

        if top_p >= 0.85:
            risk, candidate_set = "confident", [pred]
            reason = "Top class clearly leads; all other classes score low."
        else:
            risk = "review"
            candidate_set = [pred, runner_up]
            reason = f"{pred} and {runner_up} score closely; manual review recommended."

        pool = [r for r in reference if r["label"] in candidate_set] or reference
        picks = random.sample(pool, k=min(5, len(pool)))
        similar = [
            {"image": r["image"], "label": r["label"],
             "score": round(random.uniform(0.80, 0.97), 3)}
            for r in picks
        ]
        similar.sort(key=lambda s: s["score"], reverse=True)

        cases.append({
            "case_id": case_id,
            "image": img,
            "true_label": truth,
            "prediction": pred,
            "probabilities": dict(sorted(probs.items(), key=lambda kv: -kv[1])),
            "risk": risk,
            "candidate_set": candidate_set,
            "reason": reason,
            "similar_cases": similar,
        })

    (OUT / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    print(f"wrote {len(cases)} cases and {len(reference)} reference tiles to {OUT}")


if __name__ == "__main__":
    main()
