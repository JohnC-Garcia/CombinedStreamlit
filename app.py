import streamlit as st
from ultralytics import YOLO, RTDETR
import cv2
import tempfile
import os
import numpy as np
from sklearn.cluster import KMeans
import torchvision.transforms as T
from collections import defaultdict
import random

# Fix for PyTorch/Streamlit compatibility
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

st.set_page_config(page_title="Smart Retail Detector")
st.title("🧠📦 Product Detection, Clustering & Relabeling")
st.write("Upload a video to detect products, group them by visual similarity, and relabel interactively.")

@st.cache_resource
def load_model():
    return RTDETR("product_weights.pt")

model = load_model()

# Generate distinct colors for different product clusters
def generate_colors(n_clusters):
    colors = []
    for i in range(n_clusters):
        # Generate vibrant, distinct colors using a simpler approach
        hue = int((i * 137.5) % 180)  # Keep hue within valid range for OpenCV
        # Create HSV color and convert to RGB
        hsv = np.uint8([[[hue, 255, 255]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        color = tuple(int(c) for c in rgb[0][0])
        colors.append(color)
    return colors

def group_nearby_boxes(boxes_with_labels, distance_threshold=100):
    """Group nearby boxes of the same cluster to reduce clutter"""
    grouped = defaultdict(list)
    
    # Group by cluster_id first
    clusters = defaultdict(list)
    for box, cluster_id in boxes_with_labels:
        clusters[cluster_id].append(box)
    
    # For each cluster, group nearby boxes
    for cluster_id, boxes in clusters.items():
        if len(boxes) <= 1:
            grouped[cluster_id] = boxes
            continue
            
        # Simple grouping by proximity
        used = set()
        groups = []
        
        for i, box1 in enumerate(boxes):
            if i in used:
                continue
                
            group = [box1]
            used.add(i)
            x1_center = (box1[0] + box1[2]) / 2
            y1_center = (box1[1] + box1[3]) / 2
            
            for j, box2 in enumerate(boxes):
                if j in used:
                    continue
                    
                x2_center = (box2[0] + box2[2]) / 2
                y2_center = (box2[1] + box2[3]) / 2
                
                distance = np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
                
                if distance < distance_threshold:
                    group.append(box2)
                    used.add(j)
            
            groups.append(group)
        
        grouped[cluster_id] = groups
    
    return grouped

def create_group_bounding_box(boxes):
    """Create a bounding box that encompasses all boxes in a group"""
    if len(boxes) == 1:
        return boxes[0], 1
    
    x1_min = min(box[0] for box in boxes)
    y1_min = min(box[1] for box in boxes)
    x2_max = max(box[2] for box in boxes)
    y2_max = max(box[3] for box in boxes)
    
    return [x1_min, y1_min, x2_max, y2_max], len(boxes)

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

        # Generate colors for clusters
        cluster_colors = generate_colors(k)

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
        
        # Show color preview for each cluster
        cols = st.columns(min(5, k))
        for i in range(k):
            with cols[i % 5]:
                color_hex = "#{:02x}{:02x}{:02x}".format(*cluster_colors[i])
                st.markdown(f'<div style="background-color: {color_hex}; height: 20px; width: 100%; margin: 5px 0;"></div>', unsafe_allow_html=True)
                product_names[i] = st.text_input(f"Product {i+1}", value=f"Product {i+1}", key=f"product_{i}")

        # Add grouping distance slider
        st.subheader("🎛️ Grouping Settings")
        grouping_distance = st.slider("Grouping distance (pixels)", min_value=50, max_value=300, value=100, 
                                     help="Products within this distance will be grouped together")

        # Annotate and save video
        cap = cv2.VideoCapture(tfile_path)
        output_path = os.path.join(tempfile.gettempdir(), "final_labeled_output.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_index = 0
        with st.spinner("Generating final video with grouped labels..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame_index in label_map:
                    # Group nearby boxes
                    grouped_boxes = group_nearby_boxes(label_map[frame_index], grouping_distance)
                    
                    for cluster_id, box_groups in grouped_boxes.items():
                        color = cluster_colors[cluster_id]
                        product_name = product_names.get(cluster_id, f"Product {cluster_id+1}")
                        
                        if isinstance(box_groups[0], list):  # Multiple groups
                            for group in box_groups:
                                group_box, count = create_group_bounding_box(group)
                                x1, y1, x2, y2 = map(int, group_box)
                                
                                # Draw semi-transparent filled rectangle
                                overlay = rgb_frame.copy()
                                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                                rgb_frame = cv2.addWeighted(rgb_frame, 0.8, overlay, 0.2, 0)
                                
                                # Draw border
                                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 3)
                                
                                # Create label with count
                                label = f"{product_name}" + (f" ({count})" if count > 1 else "")
                                
                                # Calculate text size and position
                                font_scale = 0.7
                                thickness = 2
                                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                                
                                # Position text above the box
                                text_x = x1
                                text_y = max(y1 - 10, text_h + 10)
                                
                                # Draw text background
                                cv2.rectangle(rgb_frame, (text_x - 5, text_y - text_h - 5), 
                                            (text_x + text_w + 5, text_y + baseline + 5), (0, 0, 0), -1)
                                
                                # Draw text
                                cv2.putText(rgb_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                        else:  # Single boxes
                            for box in box_groups:
                                x1, y1, x2, y2 = map(int, box[:4])
                                
                                # Draw semi-transparent filled rectangle
                                overlay = rgb_frame.copy()
                                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                                rgb_frame = cv2.addWeighted(rgb_frame, 0.8, overlay, 0.2, 0)
                                
                                # Draw border
                                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 3)
                                
                                # Calculate text size and position
                                font_scale = 0.7
                                thickness = 2
                                (text_w, text_h), baseline = cv2.getTextSize(product_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                                
                                # Position text above the box
                                text_x = x1
                                text_y = max(y1 - 10, text_h + 10)
                                
                                # Draw text background
                                cv2.rectangle(rgb_frame, (text_x - 5, text_y - text_h - 5), 
                                            (text_x + text_w + 5, text_y + baseline + 5), (0, 0, 0), -1)
                                
                                # Draw text
                                cv2.putText(rgb_frame, product_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                out.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                frame_index += 1

            cap.release()
            out.release()

        st.success("Final labeled video is ready!")

        # Show summary
        st.subheader("📊 Detection Summary")
        total_detections = sum(len(boxes) for boxes in product_boxes_by_frame.values())
        st.write(f"**Total detections:** {total_detections}")
        st.write(f"**Product clusters:** {k}")
        st.write(f"**Frames processed:** {len(product_boxes_by_frame)}")

        with open(output_path, "rb") as file:
            st.download_button(
                label="▶️ Download Final Labeled Video",
                data=file,
                file_name="final_labeled_output.mp4",
                mime="video/mp4"
            )
