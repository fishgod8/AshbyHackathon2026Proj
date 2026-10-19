# Pathology Assistant: web UI + integration (Bowen)

Live: https://bowenx307--pathology-assistant-web.us-east.modal.run (sleeps when idle; the first visit takes
about half a minute to wake).

A Gradio workbench that runs the team's modules end to end on BACH breast histology (and BreakHis, below):

```
H&E image (any size from 224 x 224; BACH photos are 2048 x 1536)
 -> Mayumi: 12 patches of 224 on a 3x4 grid -> QuiltNet-B-32 -> mean -> classifier.pkl -> 4 class probabilities
 -> Ashley: two review checks
      confidence below the threshold (assess_uncertainty; 0.9, chosen on validation)
      unfamiliar image: cosine distance to the 5th nearest training image above the
      95th percentile of validation images (ashley_review gate)
 -> similar confirmed images: stub centered-cosine top-k (Felix's retrieval not delivered yet)
 -> flagged differential: suggested next step (stain or imaging correlation, cited), then general steps
 -> background PubMed papers for the top two candidates (folded), AI-screened, quotes verified verbatim
```

Pages: **Slide view** (BACH, see below), **Lookup** (upload or pick an image: suggestion, why, suggested next step, side-by-side
references per candidate, background papers) and **Evidence** (when "confident" can be trusted, review checks compared,
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

Felix's retrieval plugs in at `load_gallery()` / `analyze_embedding()` in `src/bach.py`. What the UI needs:
a function that takes one 512-dim embedding of a new image (Mayumi's QuiltNet, mean of 12 patches),
searches her `bach/image_embeddings.csv` rows with `split == "train"` (320 images), optionally filtered
to one class, and returns the k most similar as `[{image_id, label, score}]`. The gallery must be in the
same embedding space as the query (same model, same patching), or the scores are meaningless.

## Slide view (pathologist feedback, 10/18)

Pathologists read the whole slide first and zoom in only where needed. The Slide view shows a BACH whole-slide
image with a map of where to look first; clicking a spot shows that patch, where it ranks within the slide, its 5
most similar confirmed training patches and, on annotated slides, what the experts drew there. **Slides are only
shown and evaluated, never trained on**; training stays on patches and photos.

- Model: Mayumi's single-patch classifier (Normal / Benign / InSitu / Invasive from one 224 x 224 QuiltNet patch
  embedding; 92.5% on BACH validation, 87.5% on test, as the mean of 12 patches per photo). Her file
  `bach_single_patch_classifier.pkl` goes to `ui/data_bach/patch_classifier.pkl` (not committed).
- Patches are cut at the photos' physical size (224 px x 0.42 um = 94 um, so 188 px at the slides' 0.5 um/px),
  embedded with QuiltNet in their scanned colors, scored, then averaged over 5 x 5 neighboring patches. The map
  colors the most suspicious 20% of each slide: a rank within the slide, because absolute probabilities do not
  carry over from microscope photos to scans (at a 0.5 cut, most tissue is flagged).
- Checked against expert annotations (`src/slide_eval.py`, BACH annotated training slides, Aperio XML; each patch
  labeled by the region under its center; unannotated tissue counts as not cancer). AUC, cancer vs other tissue:

  | Slide | Role | Cancer share | Cancer in top 20% | AUC | Photo model AUC |
  |---|---|---|---|---|---|
  | A01 | settings chosen | 29% | 55% | 0.75 | 0.56 |
  | A08 | settings chosen | 7% | 16% | 0.73 | 0.56 |
  | A09 | settings chosen | 60% | 95% | 0.85 | 0.76 |
  | A02 | unseen | 3% | 3% | 0.58 | 0.63 |
  | A03 | unseen | 13% | 24% | 0.72 | 0.79 |

  Per-slide color normalization (Reinhard) made it worse (A01 0.66 to 0.59), so it is off. Useful as "where to
  look first", not as a detector; the photo-level model did better on the two unseen slides, so the model for
  slides is still open.

```sh
ui/.venv/bin/python -m src.bach_raw                 # Mayumi's split photos, original TIFFs (6.9 GB)
ui/.venv/bin/python -m src.patches build            # patch library: 4,800 patch embeddings + training crops
ui/.venv/bin/python -m src.slide_eval --slides A01 A08 A09 --export   # needs data/bach_wsi_annotated/*.svs + .xml
ui/.venv/bin/python -m src.slide_eval --slides A02 A03 --out data/slide_eval_holdout
ui/.venv/bin/python -m src.slide_eval --validation data/slide_eval/results.json data/slide_eval_holdout/results.json
```

## BreakHis (second dataset)

`src/breakhis.py build` builds `ui/data_breakhis/` from Mayumi's BreakHis patch embeddings (6 patches per
700 x 460 image, 82 patients, patient-level split): a temporary benign/malignant classifier with her recipe,
Ashley's checks, all 4,967 training images as the gallery. On 14 unseen test patients: accuracy 79.9%; one
fibroadenoma patient is wrong on 119 of 120 images at 98% mean confidence, and neither check catches it. BACH and
BreakHis cannot share a gallery or classifier (a BreakHis-trained classifier gets 48% on BACH). `ui/main.py`
serves both behind a BACH | BreakHis switch.

## Suggested next step (pathologist feedback, 10/18)

A pathologist told us that, when unsure, pathologists order tests rather than read papers. So a flagged
differential now leads with a next step from `ui/next_steps_bach.json`: one test per class pair
(p63 and 34βE12; CK5/6; p63 with SMMHC or calponin; imaging correlation for normal vs benign), then
"look at more of the lesion" and "ask a colleague". Every test cites verbatim sentences that code checked
against the PubMed abstract. It is labeled a draft until a pathologist reviews it:
`docs/next_steps_review.md` is the review sheet. Papers are folded under "Background papers".

## Run it

All commands from this `frontend/` folder. Python 3.10+.

```sh
python -m venv ui/.venv && ui/.venv/bin/pip install -r ui/requirements.txt
git fetch origin mayumi-model                       # the build reads Mayumi's artifacts from this branch

# Build the data folder: 40 test photos + the reference gallery fetched from Zenodo by range request
# (--per-class 80 = all 320 training images, about 3.2 GB to download; --per-class 10 = about 0.74 GB)
ui/.venv/bin/python -m src.bach build --out ui/data_bach --per-class 80 \
    --test-zip /path/to/ICIAR2018_BACH_Challenge_TestDataset.zip

# Optional: literature (PubMed), then AI screening (needs DEEPSEEK_API_KEY in frontend/.env, never committed)
ui/.venv/bin/python -m src.literature --data ui/data_bach --terms ui/literature_terms_bach.json --context breast
ui/.venv/bin/python -m src.screen_literature --data ui/data_bach

# Local app (both datasets), a temporary public link, or the permanent Modal deployment
cd ui && .venv/bin/python main.py --data data_bach data_breakhis --live --session-decisions
deploy/share_local.sh                               # prints a *.gradio.live URL
modal deploy deploy/modal_app.py                    # bakes code, data folders and QuiltNet into the image
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
- Gallery size matters more: with all 320 training images instead of 40, the closest image's centered
  score rises from a median of 0.66 to 0.84, and the closest image has the true class for 30 of 40 test
  images instead of 26.
- BACH has no patient IDs, so the split is by image and results may be optimistic.
- Mayumi's `classifier.pkl` was pickled with scikit-learn 1.7.2; loading with another version warns
  (results here match her `evaluation.csv` to 7e-8 with 1.9.1).

## Data and licenses

No data is committed (`ui/data_*` is git-ignored). BACH images: ICIAR 2018 Grand Challenge
(Aresta et al., Medical Image Analysis 2019; Zenodo 3632035), CC BY-NC-ND: credit, non-commercial,
no modification; the app shows whole images with attribution. Models are trained only on microscopy photos and
their patches; whole-slide images are shown and evaluated, never trained on. BreakHis: Spanhol et al., IEEE TBME
2016, non-commercial research use. Literature: PubMed (NCBI). Research demo; not for clinical use.
