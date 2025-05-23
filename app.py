import streamlit as st
from ultralytics import YOLO, RTDETR
import cv2
import tempfile
import os
import numpy as np
from sklearn.cluster import KMeans
import torchvision.transforms as T
from collections import defaultdict
import math

st.set_page_config(page_title="Smart Retail Detector")
st.title("🧠📦👥 Product Detection, Customer Tracking & Interaction Analysis")
st.write("Upload a video to detect products, track customers, and analyze customer-product interactions.")

@st.cache_resource
def load_models():
    product_model = RTDETR("product_weights.pt")
    human_model = YOLO("human_weights.pt")
    return product_model, human_model

product_model, human_model = load_models()

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

def calculate_distance(box1, box2):
    """Calculate distance between centers of two bounding boxes"""
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def calculate_overlap(box1, box2):
    """Calculate IoU (Intersection over Union) between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def track_customers(human_detections, max_distance=100):
    """Simple customer tracking based on proximity between frames"""
    tracked_customers = {}
    customer_id_counter = 1
    
    for frame_idx in sorted(human_detections.keys()):
        current_humans = human_detections[frame_idx]
        
        if frame_idx == 0 or not tracked_customers:
            # Initialize tracking for first frame
            for i, human_box in enumerate(current_humans):
                tracked_customers[customer_id_counter] = {
                    'last_seen': frame_idx,
                    'last_box': human_box,
                    'boxes_by_frame': {frame_idx: human_box}
                }
                customer_id_counter += 1
        else:
            # Track humans in subsequent frames
            unmatched_humans = list(range(len(current_humans)))
            
            # Try to match current detections with existing customers
            for customer_id, customer_data in tracked_customers.items():
                if frame_idx - customer_data['last_seen'] > 30:  # Skip if not seen for 30 frames
                    continue
                    
                best_match = None
                best_distance = float('inf')
                
                for i in unmatched_humans:
                    distance = calculate_distance(customer_data['last_box'], current_humans[i])
                    if distance < max_distance and distance < best_distance:
                        best_distance = distance
                        best_match = i
                
                if best_match is not None:
                    # Update customer tracking
                    tracked_customers[customer_id]['last_seen'] = frame_idx
                    tracked_customers[customer_id]['last_box'] = current_humans[best_match]
                    tracked_customers[customer_id]['boxes_by_frame'][frame_idx] = current_humans[best_match]
                    unmatched_humans.remove(best_match)
            
            # Create new customers for unmatched detections
            for i in unmatched_humans:
                tracked_customers[customer_id_counter] = {
                    'last_seen': frame_idx,
                    'last_box': current_humans[i],
                    'boxes_by_frame': {frame_idx: current_humans[i]}
                }
                customer_id_counter += 1
    
    return tracked_customers

def analyze_customer_product_interactions(tracked_customers, product_boxes_by_frame, fps, proximity_threshold=150):
    """Analyze time customers spend near products"""
    interactions = defaultdict(lambda: defaultdict(int))  # customer_id -> product_cluster -> time_frames
    
    for customer_id, customer_data in tracked_customers.items():
        for frame_idx, customer_box in customer_data['boxes_by_frame'].items():
            if frame_idx in product_boxes_by_frame:
                for product_box, product_cluster in product_boxes_by_frame[frame_idx]:
                    distance = calculate_distance(customer_box, product_box[:4])
                    if distance < proximity_threshold:
                        interactions[customer_id][product_cluster] += 1
    
    # Convert frame counts to seconds
    interaction_results = []
    for customer_id, products in interactions.items():
        for product_cluster, frame_count in products.items():
            seconds = frame_count / fps
            if seconds >= 0.5:  # Only show interactions of 0.5 seconds or more
                interaction_results.append({
                    'customer': customer_id,
                    'product': product_cluster,
                    'time': seconds
                })
    
    return sorted(interaction_results, key=lambda x: x['time'], reverse=True)

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
    human_detections = defaultdict(list)

    def extract_feature(crop):
        return crop.mean(axis=(0, 1))

    frame_index = 0
    progress_bar = st.progress(0)
    with st.spinner("Detecting products and humans..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect products
            product_results = product_model(rgb_frame, verbose=False)
            if product_results and len(product_results[0].boxes) > 0:
                product_boxes = product_results[0].boxes.data.cpu().numpy()
                for box in product_boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    crop = rgb_frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    feature = extract_feature(crop)
                    product_features.append(feature)
                    product_boxes_by_frame[frame_index].append((box, feature))
            
            # Detect humans
            human_results = human_model(rgb_frame, verbose=False)
            if human_results and len(human_results[0].boxes) > 0:
                human_boxes = human_results[0].boxes.data.cpu().numpy()
                for box in human_boxes:
                    human_detections[frame_index].append(box[:4])  # Only keep x1, y1, x2, y2

            frame_index += 1
            progress_bar.progress(min(frame_index / total_frames, 1.0))

        cap.release()

    st.success("Detection complete!")

    # Track customers
    st.write("Tracking customers...")
    tracked_customers = track_customers(human_detections)
    st.success(f"Tracked {len(tracked_customers)} unique customers!")

    # Product clustering
    if product_features:
        k = min(10, len(product_features))
        kmeans = KMeans(n_clusters=k, random_state=0).fit(product_features)
        cluster_labels = kmeans.labels_

        # Map features to clusters
        label_map = {}
        idx = 0
        for frame_idx, boxes in product_boxes_by_frame.items():
            label_map[frame_idx] = []
            for box, feat in boxes:
                label_map[frame_idx].append((box, int(cluster_labels[idx])))
                idx += 1

        # Product naming interface
        st.subheader("📝 Rename Product Clusters")
        product_names = {}
        for i in range(k):
            product_names[i] = st.text_input(f"Name for Product {i+1}", value=f"Product {i+1}")

        # Analyze customer-product interactions
        st.subheader("📊 Customer-Product Interaction Analysis")
        interactions = analyze_customer_product_interactions(tracked_customers, label_map, fps)
        
        if interactions:
            st.write("**Top Customer-Product Interactions:**")
            for interaction in interactions[:20]:  # Show top 20 interactions
                customer_id = interaction['customer']
                product_cluster = interaction['product']
                time_spent = interaction['time']
                product_name = product_names.get(product_cluster, f"Product {product_cluster+1}")
                st.write(f"🛒 **Customer {customer_id}** spent **{time_spent:.1f} seconds** near **{product_name}**")
        else:
            st.write("No significant customer-product interactions detected (minimum 0.5 seconds required).")

        # Generate annotated video automatically
        st.subheader("🎬 Generating Final Video")
        cap = cv2.VideoCapture(tfile_path)
        output_path = os.path.join(tempfile.gettempdir(), "final_labeled_output.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_index = 0
        video_progress = st.progress(0)
        with st.spinner("Generating final video with labels..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Draw product boxes
                if frame_index in label_map:
                    for box, cluster_id in label_map[frame_index]:
                        x1, y1, x2, y2 = map(int, box[:4])
                        label = product_names.get(cluster_id, f"Product {cluster_id+1}")
                        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(rgb_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # Draw customer boxes
                for customer_id, customer_data in tracked_customers.items():
                    if frame_index in customer_data['boxes_by_frame']:
                        box = customer_data['boxes_by_frame'][frame_index]
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(rgb_frame, f"Customer {customer_id}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

                out.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                frame_index += 1
                video_progress.progress(min(frame_index / total_frames, 1.0))

            cap.release()
            out.release()

        st.success("Final labeled video is ready!")

        # Load the video file once for download
        with open(output_path, "rb") as video_file:
            video_bytes = video_file.read()
        
        st.download_button(
            label="▶️ Download Final Labeled Video",
            data=video_bytes,
            file_name="final_labeled_output.mp4",
            mime="video/mp4"
        )

    else:
        st.warning("No products detected in the video.")
