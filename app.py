"""Minimal Streamlit mock demo for uncertainty assessment."""

import streamlit as st

from uncertainty import assess_uncertainty


st.set_page_config(page_title="Image Uncertainty Demo")
st.title("Image Uncertainty Demo")
st.file_uploader(
    "Upload an image",
    type=("png", "jpg", "jpeg", "webp"),
)

st.subheader("Mock mode: synthetic results only")
st.warning(
    "Mock results do not analyze the uploaded image. They use fixed synthetic "
    "class probabilities and are for interface demonstration only."
)

class_names = ["mock_class_a", "mock_class_b", "mock_class_c"]
probabilities = [0.18, 0.72, 0.10]
provisional_threshold = 0.7
predicted_index = probabilities.index(max(probabilities))
confidence, needs_review = assess_uncertainty(
    probabilities, provisional_threshold
)

st.write(f"Mock predicted label: {class_names[predicted_index]}")
st.write(f"Mock class names: {class_names}")
st.write(f"Mock probabilities: {probabilities}")
st.metric("Mock confidence", f"{confidence:.2f}")
st.write(f"Needs review: {'Yes' if needs_review else 'No'}")
st.caption(
    "The 0.7 threshold is provisional and must be selected using validation data."
)

st.subheader("Reference images")
st.info("Placeholder only. No reference images or real cases are included.")