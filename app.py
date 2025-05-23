import streamlit as st
from ultralytics import YOLO, RTDETR
import cv2
import tempfile
import os
import numpy as np
from sklearn.cluster import KMeans
import torchvision.transforms as T
from collections import defaultdict

st.set_page_config(page_title="Smart Retail Detector")
st.title("🧠📦 Product Detection, Clustering & Relabeling")
st.write("Upload a video to detect products, group them by visual similarity, and relabel interactively.")

@st.cache_resource
def load_model():
    return RTDETR("product_weights.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file:
    st.video(uploaded_file)
    st.write("Processing video... Please wait ⌛")

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile_path = tfile.name

    cap = cv2.VideoCapture(tfile_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    transform = T.Compose([T.ToPILImage(), T.Resize((64, 64)), T.ToTensor()])

    product_features = []
    product_boxes_by_frame = defaultdict(list)

    def extract_feature(crop):
        return crop.mean(axis=(0, 1))

    frame_index = 0
    progress_bar = st.progress(0)
    with st.spinner("Detecting and extracting features..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb_frame, verbose=False)

            if results and len(results[0].boxes) > 0:
                product_boxes = results[0].boxes.data.cpu().numpy()
                for box in product_boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    crop = rgb_frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    feature = extract_feature(crop)
                    product_features.append(feature)
                    product_boxes_by_frame[frame_index].append((box, feature))

            frame_index += 1
            progress_bar.progress(min(frame_index / total_frames, 1.0))

        cap.release()

    st.success("Detection complete!")

    # Clustering
    if product_features:
        k = min(10, len(product_features))
        kmeans = KMeans(n_clusters=k, random_state=0).fit(product_features)
        cluster_labels = kmeans.labels_

        label_map = {}
        idx = 0
        for frame_idx, boxes in product_boxes_by_frame.items():
            label_map[frame_idx] = []
            for box, feat in boxes:
                label_map[frame_idx].append((box, int(cluster_labels[idx])))
                idx += 1

        # Rename UI
        st.subheader("📝 Rename Product Clusters")
        product_names = {}
        for i in range(k):
            product_names[i] = st.text_input(f"Name for Product {i+1}", value=f"Product {i+1}")

        # Annotate and save video
        cap = cv2.VideoCapture(tfile_path)
        output_path = os.path.join(tempfile.gettempdir(), "final_labeled_output.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_index = 0
        with st.spinner("Generating final video with labels..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame_index in label_map:
                    for box, cluster_id in label_map[frame_index]:
                        x1, y1, x2, y2 = map(int, box[:4])
                        label = product_names.get(cluster_id, f"Product {cluster_id+1}")
                        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(rgb_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

                out.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                frame_index += 1

            cap.release()
            out.release()

        st.success("Final labeled video is ready!")

        with open(output_path, "rb") as file:
            st.download_button(
                label="▶️ Download Final Labeled Video",
                data=file,
                file_name="final_labeled_output.mp4",
                mime="video/mp4"
            )
