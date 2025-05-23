import streamlit as st
from ultralytics import YOLO, RTDETR
import cv2
import tempfile
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

def extract_product_features(image, box):
    """Extract features from a product bounding box for clustering"""
    x1, y1, x2, y2 = map(int, box[:4])
    
    # Crop the product region
    product_crop = image[y1:y2, x1:x2]
    
    if product_crop.size == 0:
        return np.zeros(10)  # Return zero features if crop is empty
    
    # Resize to standard size for consistent feature extraction
    try:
        product_crop = cv2.resize(product_crop, (64, 64))
    except:
        return np.zeros(10)
    
    # Extract simple features
    features = []
    
    # Color histogram features (simplified)
    for channel in range(3):
        hist = cv2.calcHist([product_crop], [channel], None, [8], [0, 256])
        features.extend(hist.flatten())
    
    # Shape features
    features.append(x2 - x1)  # width
    features.append(y2 - y1)  # height
    features.append((x2 - x1) / (y2 - y1 + 1e-6))  # aspect ratio
    
    # Position features (normalized)
    img_h, img_w = image.shape[:2]
    features.append((x1 + x2) / (2 * img_w))  # center_x normalized
    features.append((y1 + y2) / (2 * img_h))  # center_y normalized
    
    return np.array(features[:32])  # Limit to 32 features

def cluster_products(product_data, n_clusters=None):
    """Cluster products based on their features"""
    if len(product_data) == 0:
        return [], None
    
    features = [item['features'] for item in product_data]
    features_array = np.array(features)
    
    # Handle case where all features are zero
    if np.all(features_array == 0):
        return [0] * len(product_data), None
    
    # Standardize features
    scaler = StandardScaler()
    try:
        features_scaled = scaler.fit_transform(features_array)
    except:
        return [0] * len(product_data), None
    
    # Determine optimal number of clusters if not specified
    if n_clusters is None:
        n_clusters = min(5, max(1, len(product_data) // 3))
    
    # Perform clustering
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        return cluster_labels.tolist(), kmeans
    except:
        return [0] * len(product_data), None

def draw_boxes(image, boxes, color=(0, 255, 0), labels=None):
    """Draw bounding boxes with labels on image"""
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        if labels and i < len(labels):
            label = labels[i]
            # Create background for text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                image, 
                (x1, y1 - text_height - baseline - 5), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            cv2.putText(
                image, label, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )
    return image

# Streamlit UI
st.set_page_config(page_title="Smart Product Detector with Clustering", layout="wide")
st.title("🎯📦 Smart Product Detection, Clustering & Labeling")
st.write("Detect products, cluster similar ones, customize labels, and generate annotated video.")

@st.cache_resource
def load_model():
    """Load the product detection model"""
    try:
        product_model = RTDETR("product_weights.pt")
        return product_model
    except:
        st.error("Could not load product_weights.pt. Using YOLOv8n as fallback.")
        return YOLO("yolov8n.pt")

product_model = load_model()

# File upload
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    st.video(uploaded_file)
    
    # Initialize session state for product labels
    if 'product_labels' not in st.session_state:
        st.session_state.product_labels = {}
    if 'clustering_done' not in st.session_state:
        st.session_state.clustering_done = False
    if 'product_data' not in st.session_state:
        st.session_state.product_data = []
    
    # Step 1: Product Detection and Clustering
    if st.button("🔍 Detect & Cluster Products", type="primary"):
        st.write("Processing video for product detection and clustering... ⏳")
        
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile_path = tfile.name
        
        cap = cv2.VideoCapture(tfile_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames for clustering (every nth frame to avoid redundancy)
        sample_interval = max(1, total_frames // 50)  # Sample ~50 frames max
        product_data = []
        frame_samples = []
        
        frame_index = 0
        progress_bar = st.progress(0)
        
        with st.spinner("Extracting product features for clustering..."):
            while cap.isOpened() and frame_index < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_index % sample_interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect products
                    results = product_model(rgb_frame, verbose=False)
                    
                    if results and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.data.cpu().numpy()
                        
                        for box in boxes:
                            # Extract features for clustering
                            features = extract_product_features(rgb_frame, box)
                            
                            product_info = {
                                'frame_index': frame_index,
                                'box': box,
                                'features': features,
                                'frame_sample': rgb_frame.copy()
                            }
                            product_data.append(product_info)
                
                frame_index += 1
                progress_bar.progress(min(frame_index / total_frames, 1.0))
            
            cap.release()
        
        # Perform clustering
        if product_data:
            st.write(f"Found {len(product_data)} product instances. Performing clustering...")
            
            # Determine number of clusters
            n_clusters = st.slider(
                "Number of product clusters:", 
                min_value=1, 
                max_value=min(10, len(product_data)), 
                value=min(5, max(1, len(product_data) // 5))
            )
            
            cluster_labels, kmeans_model = cluster_products(product_data, n_clusters)
            
            # Assign cluster labels to products
            for i, product in enumerate(product_data):
                product['cluster'] = cluster_labels[i]
            
            st.session_state.product_data = product_data
            st.session_state.clustering_done = True
            
            # Initialize default labels
            unique_clusters = list(set(cluster_labels))
            for cluster_id in unique_clusters:
                if cluster_id not in st.session_state.product_labels:
                    st.session_state.product_labels[cluster_id] = f"Product {cluster_id + 1}"
            
            st.success(f"✅ Clustering complete! Found {len(unique_clusters)} product clusters.")
            
        else:
            st.warning("No products detected in the video.")
    
    # Step 2: Label Customization
    if st.session_state.clustering_done and st.session_state.product_data:
        st.subheader("🏷️ Customize Product Labels")
        
        # Group products by cluster for display
        clusters = {}
        for product in st.session_state.product_data:
            cluster_id = product['cluster']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(product)
        
        # Display cluster samples and label inputs
        cols = st.columns(min(3, len(clusters)))
        
        for idx, (cluster_id, products) in enumerate(clusters.items()):
            with cols[idx % len(cols)]:
                st.write(f"**Cluster {cluster_id + 1}** ({len(products)} instances)")
                
                # Show a sample image from this cluster
                sample_product = products[0]
                frame = sample_product['frame_sample']
                box = sample_product['box']
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Crop and display sample
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        st.image(crop, caption=f"Sample from Cluster {cluster_id + 1}", width=150)
                
                # Label input
                current_label = st.session_state.product_labels.get(cluster_id, f"Product {cluster_id + 1}")
                new_label = st.text_input(
                    f"Label for Cluster {cluster_id + 1}:",
                    value=current_label,
                    key=f"label_{cluster_id}"
                )
                st.session_state.product_labels[cluster_id] = new_label
    
    # Step 3: Generate Final Annotated Video
    if st.session_state.clustering_done and st.button("🎬 Generate Labeled Video", type="secondary"):
        st.write("Generating final video with custom labels... ⏳")
        
        # Reload video for final processing
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.getvalue())
        tfile_path = tfile.name
        
        cap = cv2.VideoCapture(tfile_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = os.path.join(tempfile.gettempdir(), "labeled_product_video.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # Create a mapping from features to cluster labels for real-time assignment
        cluster_features = {}
        for product in st.session_state.product_data:
            cluster_id = product['cluster']
            if cluster_id not in cluster_features:
                cluster_features[cluster_id] = []
            cluster_features[cluster_id].append(product['features'])
        
        # Calculate cluster centroids for assignment
        cluster_centroids = {}
        for cluster_id, features_list in cluster_features.items():
            if features_list:
                cluster_centroids[cluster_id] = np.mean(features_list, axis=0)
        
        frame_index = 0
        progress_bar = st.progress(0)
        total_detections = 0
        
        with st.spinner("Processing video with labels..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated = rgb_frame.copy()
                
                # Detect products in current frame
                results = product_model(rgb_frame, verbose=False)
                
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.data.cpu().numpy()
                    labels = []
                    
                    for box in boxes:
                        # Extract features for current detection
                        features = extract_product_features(rgb_frame, box)
                        
                        # Assign to closest cluster
                        best_cluster = 0
                        min_distance = float('inf')
                        
                        for cluster_id, centroid in cluster_centroids.items():
                            distance = np.linalg.norm(features - centroid)
                            if distance < min_distance:
                                min_distance = distance
                                best_cluster = cluster_id
                        
                        # Get custom label for this cluster
                        label = st.session_state.product_labels.get(best_cluster, f"Product {best_cluster + 1}")
                        labels.append(label)
                    
                    # Draw boxes with custom labels
                    annotated = draw_boxes(annotated, boxes, color=(0, 255, 0), labels=labels)
                    total_detections += len(boxes)
                
                out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                
                frame_index += 1
                progress_bar.progress(min(frame_index / total_frames, 1.0))
            
            cap.release()
            out.release()
        
        st.success(f"✅ Video processing complete! Total detections: {total_detections}")
        
        # Display download button
        st.subheader("📥 Download Labeled Video")
        if os.path.exists(output_path):
            with open(output_path, "rb") as file:
                st.download_button(
                    label="📹 Download Labeled Video",
                    data=file,
                    file_name="labeled_product_video.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("Output video not found. Please try again.")
        
        # Show summary
        st.subheader("📊 Detection Summary")
        st.write(f"**Total product detections:** {total_detections}")
        st.write("**Product labels used:**")
        for cluster_id, label in st.session_state.product_labels.items():
            cluster_count = len([p for p in st.session_state.product_data if p['cluster'] == cluster_id])
            st.write(f"- {label}: ~{cluster_count} training instances")

    # Reset button
    if st.button("🔄 Reset All", type="secondary"):
        st.session_state.product_labels = {}
        st.session_state.clustering_done = False
        st.session_state.product_data = []
        st.rerun()
