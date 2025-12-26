import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import ChestXrayClassifier
from ultralytics import YOLO

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config(page_title="Chest X-ray AI", layout="centered")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 14               # CHANGE if needed
BACKBONE = "efficientnet"      # MUST MATCH TRAINING
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]

# -----------------------------------
# LOAD MODELS (CACHED)
# -----------------------------------
@st.cache_resource
def load_classifier():
    model = ChestXrayClassifier(
        num_classes=NUM_CLASSES,
        backbone=BACKBONE
    )
    model.load_state_dict(
        torch.load("best_model.hcl", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_detector():
    return YOLO("best.pt")


classifier = load_classifier()
detector = load_detector()

# -----------------------------------
# IMAGE TRANSFORMS (MATCH TRAINING)
# -----------------------------------
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# -----------------------------------
# UI
# -----------------------------------
st.title(" Chest X-ray Disease Detection & Classification")

uploaded_file = st.file_uploader(
    "Upload a Chest X-ray image",
    type=["png", "jpg", "jpeg"]
)

threshold = st.slider(
    "Classification Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -----------------------------------
    # CLASSIFICATION
    # -----------------------------------
    st.subheader(" Classification Results")

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = classifier(img_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    results = []
    for cls, prob in zip(CLASS_NAMES, probs):
        if prob >= threshold:
            results.append((cls, prob))

    if results:
        for cls, prob in results:
            st.success(f"{cls}: {prob:.2f}")
    else:
        st.info("No disease detected above threshold")

    # -----------------------------------
    # DETECTION
    # -----------------------------------
    st.subheader(" Detection Results")

    yolo_results = detector(image)

    annotated_img = yolo_results[0].plot()
    st.image(annotated_img, caption="Detected Regions", use_column_width=True)
