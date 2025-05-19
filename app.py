import streamlit as st
from ultralytics import YOLO, RTDETR
import cv2
import tempfile
import os
import base64
import numpy as np
import matplotlib.pyplot as plt


def draw_boxes(image, results):
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


st.set_page_config(page_title="Smart Retail Detector")

st.title("🧠📦 Human + Product Detection & Heatmap")
st.write("Detect people and/or products in video, visualize heatmaps, and download the annotated result.")

# Load models
@st.cache_resource
def load_models():
    human_model = YOLO("human_weights.pt")
    product_model = RTDETR("product_weights.pt")
    return human_model, product_model

human_model, product_model = load_models()

# Model selection
task = st.radio("Select detection task:", ["Human Detection Only", "Product Detection Only", "Both"])

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file:
    st.video(uploaded_file)
    st.write("Processing full video... Please wait ⏳")

    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile_path = tfile.name

    cap = cv2.VideoCapture(tfile_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video path
    output_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    preview_frames = []
    heatmap = np.zeros((height, width), dtype=np.float32)
    total_human_boxes = 0
    total_product_boxes = 0

    frame_index = 0
    with st.spinner("Running detection..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated = rgb_frame.copy()

            # Detection logic
            if task in ["Human Detection Only", "Both"]:
                results_human = human_model(rgb_frame, verbose=False)
                if results_human and results_human[0].boxes:
                    annotated = draw_boxes(annotated, results_human[0].boxes, color=(255, 0, 0), label="Human")
                    for box in results_human[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        heatmap[y1:y2, x1:x2] += 1
                    total_human_boxes += len(results_human[0].boxes)

            if task in ["Product Detection Only", "Both"]:
                results_product = product_model(rgb_frame)
                if results_product and results_product[0].boxes:
                    annotated = draw_boxes(annotated, results_product[0].boxes, color=(0, 255, 0), label="Product")
                    for box in results_product[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        heatmap[y1:y2, x1:x2] += 1
                    total_product_boxes += len(results_product[0].boxes)

            out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            if frame_index < 5:
                preview_frames.append((rgb_frame, annotated))

            frame_index += 1

    cap.release()
    out.release()

    # Normalize heatmap
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    st.subheader("🖼️ Detection Preview (First 5 Frames)")
    for i, (orig, ann) in enumerate(preview_frames):
        st.markdown(f"**Frame {i+1}**")
        col1, col2 = st.columns(2)
        with col1:
            st.image(orig, caption="Original", use_column_width=True)
        with col2:
            st.image(ann, caption="With Detections", use_column_width=True)

    st.subheader("🔥 Heatmap of Detection Density")
    st.image(heatmap_color, caption="Detection Heatmap", use_column_width=True)

    st.subheader("📊 Analytics")
    if task in ["Human Detection Only", "Both"]:
        st.write(f"👤 Total human detections: **{total_human_boxes}**")
    if task in ["Product Detection Only", "Both"]:
        st.write(f"📦 Total product detections: **{total_product_boxes}**")

    st.subheader("📥 Download Annotated Video")
    with open(output_path, "rb") as file:
        video_bytes = file.read()
        b64 = base64.b64encode(video_bytes).decode()
        href = f'<a href="data:video/mp4;base64,{b64}" download="annotated_video.mp4">▶️ Click here to download annotated video</a>'
        st.markdown(href, unsafe_allow_html=True)


def draw_boxes(frame, boxes, color=(0, 255, 0), label=""):
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
