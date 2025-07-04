import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np


@st.cache_resource
def load_model():
    return YOLO("./runs/detect/train16/weights/best.pt")


@st.cache_resource
def get_available_cameras():
    def probe():
        available = []
        for i in range(10):  # Test camera indexes 0 to 9
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.read()[0]:
                available.append(i)
            cap.release()
        return available
    return probe()


# Load model
model = load_model()

st.title(" Real-Time Knife Detection with YOLOv11")

# Webcam toggle
run = st.checkbox('Start Camera')

camera_options = get_available_cameras()
camera_index = st.selectbox("Select Camera", options=camera_options, index=0, format_func=lambda x: f"Camera {x}")

# Display confidence slider
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)


# Start webcam when checkbox is checked
FRAME_WINDOW = st.image([])


if run:
    cap = cv2.VideoCapture(camera_index)

    # st.write("Press **Stop Camera** to end.")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame. Exiting...")
            break

        # Run YOLO model
        results = model.predict(frame, conf=conf_threshold, verbose=False)

        # Plot predictions
        annotated_frame = results[0].plot()

        # Convert BGR (OpenCV) to RGB (Streamlit uses RGB)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(annotated_frame)

        # # Break if checkbox unchecked
        # run = st.checkbox('Start Camera', value=True)

    cap.release()
    cv2.destroyAllWindows()
else:
    st.write("Click the checkbox above to start knife detection.")
