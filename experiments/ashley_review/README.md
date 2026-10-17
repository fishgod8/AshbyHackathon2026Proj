# Ashley Review Split

This is Ashley's personal development split only. It does not change or replace any team split.

- Method: per-class random split of non-demo rows, NumPy `default_rng(seed=42)`; within each class, rounded 20% goes to test, then rounded 20% to validation, and the remainder to reference. Target overall proportions are reference 60%, validation 20%, test 20%.
- The nine images recorded in `outputs/musk_zero_shot_demo/zero_shot_predictions.csv` were identified by exact `image_id`, checked against metadata labels, and assigned to `reference`; none is in `test`.
- No trusted patient or slide grouping field is present. Patient independence cannot be guaranteed; no patient IDs were inferred from filenames.
- This split cannot represent final performance after integration with the team classifier. Revalidate thresholds when this is connected to the team reference library.
- Counts include the nine reserved demo images. Actual split sizes: reference=4296 (60.03%), validation=1430 (19.98%), test=1430 (19.98%).

## Integrity

- Source rows: `7156`; embedding shape/dtype: `(7156, 1024)` / `float32`.
- SHA-256 `embeddings.npy`: `00e4b443033298715bc67c86aae458ca84549a3209350f4d04639e0261261820`.
- SHA-256 `metadata.csv`: `5fae48b6f5f16fde846e4da92ec0ac9e4b849f4350a6d24005608025eb7e455c`.
- SHA-256 `extraction_config.json`: `31c7ecf1b6154c3c05ee21a65b786100bf4a95c50737c92b67f256b2dbe59ab0`.
- SHA-256 `split_manifest.csv`: `6e1ac9303c2b4d3e9f140f71ee6176e3a07b7752d39880366a0014a57265ee90`.

## Counts by Split and Class

| Split | Total | ADI | BACK | DEB | LYM | MUC | MUS | NORM | STR | TUM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 4296 | 788 | 509 | 203 | 380 | 621 | 356 | 445 | 253 | 741 |
| validation | 1430 | 263 | 169 | 68 | 127 | 207 | 118 | 148 | 84 | 246 |
| test | 1430 | 263 | 169 | 68 | 127 | 207 | 118 | 148 | 84 | 246 |
