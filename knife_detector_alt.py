import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# Load model (only once)
@st.cache_resource
def load_model():
    return YOLO("./runs/detect/train16/weights/best.pt")

model = load_model()

st.title("Knife Detection with YOLOv11 (Cloud-Compatible)")

# Upload camera image
img_file = st.camera_input("Take a picture with your webcam")

# Confidence threshold
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

if img_file:
    # Read image as OpenCV image
    img = Image.open(img_file)
    img_np = np.array(img)

    # Run detection
    results = model.predict(img_np, conf=conf_threshold, verbose=False)
    annotated_frame = results[0].plot()

    # Convert for display
    st.image(annotated_frame, caption="Detection Result", channels="BGR")
else:
    st.info("Please take a picture using your webcam.")
