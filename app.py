import streamlit as st
from ultralytics import YOLO, RTDETR
import cv2
import tempfile
import os
import numpy as np

def draw_boxes(image, boxes, color=(0, 255, 0), label=None):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2, cv2.LINE_AA)
    return image

st.set_page_config(page_title="Smart Retail Detector")
st.title("🧠📦 Human + Product Detection & Heatmap")
st.write("Detect people and/or products in video, visualize heatmaps, and download the annotated result.")

@st.cache_resource
def load_models():
    human_model = YOLO("human_weights.pt")
    product_model = RTDETR("product_weights.pt")
    return human_model, product_model

human_model, product_model = load_models()

task = st.radio("Select detection task:", ["Human Detection Only", "Product Detection Only", "Both"])
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file:
    st.video(uploaded_file)
    st.write("Processing full video... Please wait ⏳")

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile_path = tfile.name

    cap = cv2.VideoCapture(tfile_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    if not out.isOpened():
        st.error("⚠️ Failed to open VideoWriter. Check output path and codec.")
    else:
        preview_frames = []
        heatmaps = [np.zeros((height, width), dtype=np.float32) for _ in range(10)]
        total_human_boxes = 0
        total_product_boxes = 0

        frame_index = 0
        progress_bar = st.progress(0)
        with st.spinner("Running detection..."):
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    annotated = rgb_frame.copy()

                    # Human detection
                    if task in ["Human Detection Only", "Both"]:
                        results_human = human_model(rgb_frame, verbose=False)
                        if results_human and len(results_human[0].boxes) > 0:
                            human_boxes = results_human[0].boxes.xyxy.cpu().numpy()
                            annotated = draw_boxes(annotated, human_boxes, color=(255, 0, 0), label="Human")
                            for box in human_boxes:
                                x1, y1, x2, y2 = map(int, box[:4])
                                segment_idx = min(9, int((frame_index / total_frames) * 10))
                                heatmaps[segment_idx][y1:y2, x1:x2] += 1
                            total_human_boxes += len(human_boxes)

                    # Product detection
                    if task in ["Product Detection Only", "Both"]:
                        results_product = product_model(rgb_frame, verbose=False)
                        if results_product and len(results_product[0].boxes) > 0:
                            product_boxes = results_product[0].boxes.data.cpu().numpy()
                            annotated = draw_boxes(annotated, product_boxes, color=(0, 255, 0), label="Product")
                            total_product_boxes += len(product_boxes)

                    out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                    if frame_index < 5:
                        preview_frames.append((rgb_frame, annotated))

                    frame_index += 1
                    if frame_index >= total_frames:
                        break
                    progress_bar.progress(min(frame_index / total_frames, 1.0))
            except Exception as e:
                st.error(f"⚠️ Detection failed: {e}")
            finally:
                cap.release()
                out.release()

        st.success("✅ Detection complete!")

        st.subheader("🔥 Detection Heatmaps (Humans Only)")
        selected_segment = st.slider("Select segment:", 1, 10, 1)
        heatmap_norm = cv2.normalize(heatmaps[selected_segment - 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        st.image(heatmap_color, caption=f"Segment {selected_segment} Heatmap (Humans Only)", use_container_width=True)

        st.subheader("🖼️ Detection Preview (First 5 Frames)")
        for i, (orig, ann) in enumerate(preview_frames):
            st.markdown(f"**Frame {i+1}**")
            col1, col2 = st.columns(2)
            with col1:
                st.image(orig, caption="Original", use_container_width=True)
            with col2:
                st.image(ann, caption="With Detections", use_container_width=True)

        st.subheader("📊 Analytics")
        if task in ["Human Detection Only", "Both"]:
            st.write(f"👤 Total human detections: **{total_human_boxes}**")
        if task in ["Product Detection Only", "Both"]:
            st.write(f"📦 Total product detections: **{total_product_boxes}**")

        st.subheader("📅 Download Annotated Video")
        if os.path.exists(output_path):
            with open(output_path, "rb") as file:
                st.download_button(
                    label="▶️ Click to Download Annotated Video",
                    data=file,
                    file_name="annotated_video.mp4",
                    mime="video/mp4"
                )
        else:
            st.warning("⚠️ Output video not available. Please try reprocessing.")
