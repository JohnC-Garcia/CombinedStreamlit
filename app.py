import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

# Avoid PyTorch import issues by importing ultralytics after Streamlit setup
st.set_page_config(page_title="Smart Retail Detector")
st.title("🧠📦 Product Detection, Clustering & Relabeling")
st.write("Upload a video to detect products, group them by visual similarity, and relabel interactively.")

@st.cache_resource
def load_model():
    try:
        from ultralytics import RTDETR
        return RTDETR("product_weights.pt")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Generate distinct colors for different product clusters
def generate_colors(n_clusters):
    # Predefined distinct colors (RGB) to avoid HSV conversion issues
    base_colors = [
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green  
        (0, 0, 255),     # Blue
        (255, 255, 0),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (255, 128, 0),   # Orange
        (128, 0, 255),   # Purple
        (255, 192, 203), # Pink
        (0, 128, 0),     # Dark Green
        (255, 165, 0),   # Orange Red
        (128, 128, 0),   # Olive
        (0, 128, 128),   # Teal
        (128, 0, 128),   # Purple
        (192, 192, 192), # Silver
    ]
    
    colors = []
    for i in range(n_clusters):
        colors.append(base_colors[i % len(base_colors)])
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
        return boxes[0][:4], 1
    
    x1_min = min(box[0] for box in boxes)
    y1_min = min(box[1] for box in boxes)
    x2_max = max(box[2] for box in boxes)
    y2_max = max(box[3] for box in boxes)
    
    return [x1_min, y1_min, x2_max, y2_max], len(boxes)

def extract_feature(crop):
    """Extract simple color features from crop"""
    if crop.size == 0:
        return np.zeros(3)
    return crop.mean(axis=(0, 1))

model = load_model()

if model is None:
    st.error("Failed to load the model. Please check if 'product_weights.pt' exists.")
    st.stop()

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

    product_features = []
    product_boxes_by_frame = defaultdict(list)

    frame_index = 0
    progress_bar = st.progress(0)
    with st.spinner("Detecting and extracting features..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
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
                        
            except Exception as e:
                st.warning(f"Error processing frame {frame_index}: {e}")
                continue

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

        # Extract sample images for each cluster
        st.subheader("🖼️ Product Cluster Samples")
        st.write("Here are sample images for each detected product cluster:")
        
        cluster_samples = {}
        cap_sample = cv2.VideoCapture(tfile_path)
        
        # Collect sample crops for each cluster
        frame_idx = 0
        samples_collected = {i: [] for i in range(k)}
        max_samples_per_cluster = 3
        
        with st.spinner("Extracting sample images..."):
            while cap_sample.isOpened() and frame_idx < total_frames:
                ret, frame = cap_sample.read()
                if not ret:
                    break
                
                if frame_idx in label_map:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    for box, cluster_id in label_map[frame_idx]:
                        if len(samples_collected[cluster_id]) < max_samples_per_cluster:
                            x1, y1, x2, y2 = map(int, box[:4])
                            crop = rgb_frame[y1:y2, x1:x2]
                            
                            if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                                # Resize crop for display (maintain aspect ratio)
                                h, w = crop.shape[:2]
                                if h > w:
                                    new_h, new_w = 100, int(100 * w / h)
                                else:
                                    new_h, new_w = int(100 * h / w), 100
                                
                                resized_crop = cv2.resize(crop, (new_w, new_h))
                                samples_collected[cluster_id].append(resized_crop)
                
                frame_idx += 1
                
                # Break if we have enough samples for all clusters
                if all(len(samples) >= max_samples_per_cluster for samples in samples_collected.values()):
                    break
        
        cap_sample.release()
        
        # Display samples in a grid
        cols_per_row = 3
        for cluster_id in range(k):
            if samples_collected[cluster_id]:
                st.write(f"**Product Cluster {cluster_id + 1}:**")
                
                # Create columns for sample images
                sample_cols = st.columns(min(len(samples_collected[cluster_id]), cols_per_row))
                
                for idx, sample in enumerate(samples_collected[cluster_id][:cols_per_row]):
                    with sample_cols[idx]:
                        # Convert RGB to display format
                        st.image(sample, caption=f"Sample {idx + 1}", width=120)
                
                st.write("---")  # Separator line
        
        # Rename UI
        st.subheader("📝 Rename Product Clusters")
        st.write("Use the sample images above to identify and rename each product cluster:")
        
        product_names = {}
        
        # Show color preview and rename inputs for each cluster
        cols = st.columns(min(3, k))
        for i in range(k):
            with cols[i % 3]:
                color_hex = "#{:02x}{:02x}{:02x}".format(*cluster_colors[i])
                st.markdown(f'<div style="background-color: {color_hex}; height: 20px; width: 100%; margin: 5px 0; border-radius: 3px;"></div>', unsafe_allow_html=True)
                product_names[i] = st.text_input(f"Product {i+1} Name", value=f"Product {i+1}", key=f"product_{i}")
                
                # Show cluster info
                cluster_count = sum(1 for boxes in label_map.values() for box, cid in boxes if cid == i)
                st.caption(f"📊 Found in {cluster_count} detections")
                st.write("")  # Add some spacing

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
                        
                        # Check if we have grouped boxes or individual boxes
                        if box_groups and len(box_groups) > 0:
                            # Handle case where box_groups might contain groups or individual boxes
                            if isinstance(box_groups[0], list) and len(box_groups[0]) > 1:
                                # Multiple groups of boxes
                                for group in box_groups:
                                    if len(group) > 0:
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
                            else:
                                # Individual boxes
                                for box_item in box_groups:
                                    # Handle both single boxes and grouped boxes
                                    if isinstance(box_item, list) and len(box_item) > 4:
                                        # This is a box array from detection
                                        box = box_item
                                    else:
                                        # This might be a single box
                                        box = box_item
                                    
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
    else:
        st.warning("No products detected in the video. Please check your model or video content.")

    # Cleanup
    try:
        os.unlink(tfile_path)
    except:
        pass
