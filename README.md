# AshbyHackathon2026Proj

## Team Responsibilities

- **Mayumi:** image embeddings and tissue classification (`predict.py`).
- **Ashley:** uncertainty assessment and evaluation (`uncertainty.py`).
- **Felix:** similar-image retrieval (`retrieval.py`).
- **Bowen:** Streamlit UI and integration (`app.py`).

## Interfaces

- `predict_image(image)` is the planned single-image inference interface. It returns `predicted_label`, `class_names`, `probabilities`, and `embedding`. The probability at each position corresponds to the class name at the same position. Real inference is not implemented yet and raises `NotImplementedError`.
- `find_similar(embedding, k=5)` is the planned retrieval interface. It returns records containing `image_path`, `label`, and `similarity`. Real retrieval is not implemented yet and raises `NotImplementedError`.
- `assess_uncertainty(probabilities, threshold)` is implemented in `uncertainty.py`. It returns `(confidence, needs_review)`, where confidence is the largest class probability. It validates the threshold and probabilities, including that their sum is approximately 1.

## Mock Demo

Install Streamlit in the project's selected Python environment if it is not already available, then run:

```powershell
python -m streamlit run app.py
```

The demo accepts an image upload, but its clearly labeled mock mode uses fixed synthetic probabilities and does **not** analyze the uploaded image. The reference-image area is only a placeholder; this repository does not include reference images or real cases. The mock threshold (`0.7`) is provisional and must be selected using validation data before real use.

## Implementation Status

- **Implemented:** input validation and confidence-based review flag in `uncertainty.py`; a synthetic mock demo in `app.py`.
- **Placeholders:** real image inference and embeddings in `predict.py`; real similar-image search in `retrieval.py`; these interfaces raise `NotImplementedError`.
- **Not included:** model weights, datasets, reference cases, or evaluation/validation data.