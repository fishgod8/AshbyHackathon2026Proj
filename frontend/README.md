# Pathology Assistant: web UI + integration (Bowen)

A Gradio workbench that runs the team's modules end to end on BACH breast histology:

```
H&E image (any size from 224 x 224; BACH photos are 2048 x 1536)
 -> Mayumi: 12 patches of 224 on a 3x4 grid -> QuiltNet-B-32 -> mean -> classifier.pkl -> 4 class probabilities
 -> Ashley: two review checks
      confidence below the threshold (assess_uncertainty; 0.9, chosen on validation)
      unfamiliar image: cosine distance to the 5th nearest training image above the
      95th percentile of validation images (ashley_review gate)
 -> similar confirmed images: stub centered-cosine top-k (Felix's retrieval not delivered yet)
 -> PubMed papers for the top two candidates, AI-screened, quotes verified verbatim
```

Pages: **Lookup** (upload or pick an image: suggestion, why, side-by-side references per
candidate, papers) and **Evidence** (when "confident" can be trusted, review checks compared,
what happens when the input changes). Queue and Review pages exist but are hidden for the demo
(`--show-review`).

## Team modules

Vendored verbatim in `src/team/`, each file headed with its source branch and commit.
Update by re-copying, never edit there.

| Module | Source | Used in |
|---|---|---|
| `ashley_uncertainty.py` | `ashley-uncertainty` @ 07d25b3 | confidence check |
| `ashley_review/` | `main` @ 1fa48f5 | unfamiliar-image check |
| Mayumi's classifier, embeddings, split | `mayumi-model` (read with `git show` at build time) | everything |

`ashley_review` fixes `EMBEDDING_DIMENSION = 1024` (MUSK). QuiltNet gives 512, so `src/bach.py`
sets that module constant instead of editing the file. The gate is rebuilt on Mayumi's split:
reference = her 320 training images, limit = 95th percentile over her 40 validation images.
Files are written in Ashley's formats to `ui/data_bach/support/`.

Felix's retrieval plugs in at `load_gallery()` / `analyze_embedding()` in `src/bach.py`.

## Run it

All commands from this `frontend/` folder. Python 3.10+.

```sh
python -m venv ui/.venv && ui/.venv/bin/pip install -r ui/requirements.txt
git fetch origin mayumi-model                       # the build reads Mayumi's artifacts from this branch

# Build the data folder (about 0.74 GB: 80 BACH photos fetched from Zenodo by range request)
ui/.venv/bin/python -m src.bach build --out ui/data_bach --per-class 10 \
    --test-zip /path/to/ICIAR2018_BACH_Challenge_TestDataset.zip

# Optional: literature (PubMed), then AI screening (needs DEEPSEEK_API_KEY in frontend/.env, never committed)
ui/.venv/bin/python -m src.literature --data ui/data_bach --terms ui/literature_terms_bach.json --context breast
ui/.venv/bin/python -m src.screen_literature --data ui/data_bach

# Local app, or a temporary public link served from this machine
cd ui && .venv/bin/python app.py --data data_bach --live
deploy/share_local.sh                               # prints a *.gradio.live URL
```

Without BACH data: `ui/.venv/bin/python ui/make_mock_data.py` then `cd ui && .venv/bin/python app.py`.
The CRC pipeline (`src/analyze.py`, `src/live.py`, `src/make_subset.py`, `src/embed_local.py`) is the
earlier prototype on CRC-100K with stub models.

## Findings so far (for the team)

- Mayumi's test set (40 images): accuracy 90%. Confidence threshold 0.9, raised from Ashley's provisional
  0.7 (agreed with her): on the 40 validation images it is the lowest value that sends every error to review
  (0.7: 6 of 8). Test: all 4 errors flagged, but 19 of 40 images go to review. Three test errors sit at
  0.85 to 0.88 confidence, so no single confidence threshold flags them without flagging many correct images.
- The unfamiliar-image check rarely fires on BACH test images (1 of 40) and caught none of those errors.
  It does catch changed inputs: on 12 challenge test photos, an ordinary JPEG export changed the
  suggestion for 3 and was flagged for all 12; half resolution changed 5 and was flagged for 11; a
  colon tile (CRC-VAL-HE-7K) got "In situ carcinoma, 100%" and was flagged.
- Lookup examples are stored lossless (WebP) for that reason; the upload hint asks for PNG. An unfamiliar
  upload shows only the warning and what to do next: no class, no comparison images, no papers.
- Raw cosine between these embeddings is always high (0.74 to 0.997; same class 0.919 vs different class
  0.902): every image shares one large component (mean training embedding has length 0.95) and the 12-patch
  mean shrinks the rest. The stub retrieval therefore removes the training mean first (centered cosine: same
  class 0.135 vs different class -0.045). Ranking barely changes (over all 320 training images: top-1 same
  class 75% either way), so ranking quality is for the real retrieval module.
- BACH has no patient IDs, so the split is by image and results may be optimistic.
- Mayumi's `classifier.pkl` was pickled with scikit-learn 1.7.2; loading with another version warns
  (results here match her `evaluation.csv` to 7e-8 with 1.9.1).

## Data and licenses

No data is committed (`ui/data_*` is git-ignored). BACH images: ICIAR 2018 Grand Challenge
(Aresta et al., Medical Image Analysis 2019; Zenodo 3632035), CC BY-NC-ND: credit, non-commercial,
no modification; the app shows whole images with attribution. Only microscopy photos are used,
never the whole-slide images. Literature: PubMed (NCBI). Research demo; not for clinical use.
