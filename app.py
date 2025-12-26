import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from model import Classifier

st.set_page_config(page_title="Multi-Model AI App")

st.title("Classification + Detection App")

# Sidebar selection
task = st.sidebar.radio(
    "Choose Task",
    ["Image Classification", "Object Detection"]
)

# Load models once
@st.cache_resource
def load_models():
    clf = Classifier()
    clf.load_state_dict(torch.load("best_model.hcl", map_location="cpu"))
    clf.eval()

    det = YOLO("best.pt")
    return clf, det

classifier, detector = load_models()

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(image, use_container_width=True)

    if task == "Image Classification":
        x = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255
        with torch.no_grad():
            pred = classifier(x)
        st.success(f"Prediction: {pred.argmax().item()}")

    else:
        with st.spinner("Detecting objects..."):
            results = detector(img, conf=0.25)
        st.image(results[0].plot(), use_container_width=True)
