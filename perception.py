import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
from itertools import combinations
import random
from PIL import Image
import tempfile

# Page config
st.set_page_config(
    page_title="Vanishing Points Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 25px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #ff6b6b;
        transform: scale(1.02);
    }
    .css-1d391kg {
        background-color: #1a1c23;
    }
    h1 {
        text-align: center;
        background: linear-gradient(135deg, #ff4b4b, #ff9a4b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    .info-box {
        background-color: #1e1e24;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #ff4b4b;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
CONFIG = {
    "hough_threshold": 70,
    "min_line_length": 70,
    "max_line_gap": 20,
    "cluster_eps": 0.20,
    "cluster_min_samples": 2,
    "random_seed": 42,
    "use_ransac": True,
    "ransac_iterations": 100,
    "ransac_threshold": 5.0,
}

@st.cache_data
def load_image(uploaded_file):
    """Load image from uploaded file"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    scale = min(1.0, 1200 / w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blur)
    low = int(max(0, 0.66 * v))
    high = int(min(255, 1.33 * v))
    return cv2.Canny(blur, low, high)

def detect_lines(edges, config):
    lines_raw = cv2.HoughLinesP(edges, 1, np.pi/180,
                                 config["hough_threshold"],
                                 config["min_line_length"],
                                 config["max_line_gap"])
    if lines_raw is None:
        return []
    
    filtered = []
    for l in lines_raw:
        x1, y1, x2, y2 = l[0]
        angle = np.degrees(np.arctan2(abs(y2-y1), abs(x2-x1)+1e-6))
        if 2 < angle < 88:
            filtered.append((x1, y1, x2, y2))
    return filtered

def line_intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None
    
    A = np.array([[x2-x1, x3-x4], [y2-y1, y3-y4]])
    B = np.array([x3-x1, y3-y1])
    
    try:
        t, u = np.linalg.solve(A, B)
        return (x1 + t*(x2-x1), y1 + t*(y2-y1))
    except np.linalg.LinAlgError:
        return None

def ransac_vp(lines, initial_vp, max_iterations=100, threshold=5.0):
    best_vp = initial_vp
    best_inliers = []
    
    line_data = []
    for x1, y1, x2, y2 in lines:
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        norm = np.sqrt(dx*dx + dy*dy)
        if norm > 0:
            dx, dy = dx/norm, dy/norm
        line_data.append(((x1, y1), (dx, dy)))
    
    if len(line_data) < 3:
        return initial_vp, []
    
    for _ in range(max_iterations):
        sample = random.sample(line_data, min(2, len(line_data)))
        if len(sample) < 2:
            continue
        
        p1, d1 = sample[0]
        p2, d2 = sample[1]
        
        A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        B = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        
        try:
            t, s = np.linalg.solve(A, B)
            vp_candidate = (p1[0] + t*d1[0], p1[1] + t*d1[1])
        except np.linalg.LinAlgError:
            continue
        
        inliers = []
        for point, direction in line_data:
            to_vp = np.array([vp_candidate[0] - point[0], vp_candidate[1] - point[1]])
            perp = np.array([-direction[1], direction[0]])
            distance = abs(np.dot(to_vp, perp))
            if distance < threshold:
                inliers.append((point, direction))
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_vp = vp_candidate
    
    if len(best_inliers) >= 2:
        A_matrix = []
        B_vector = []
        for point, direction in best_inliers:
            perp = np.array([-direction[1], direction[0]])
            A_matrix.append(perp)
            B_vector.append(np.dot(perp, point))
        
        if len(A_matrix) >= 2:
            A_matrix = np.array(A_matrix)
            B_vector = np.array(B_vector)
            try:
                refined_vp = np.linalg.lstsq(A_matrix, B_vector, rcond=None)[0]
                best_vp = (float(refined_vp[0]), float(refined_vp[1]))
            except:
                pass
    
    return best_vp, best_inliers

def find_vanishing_points(lines, img_shape, config):
    h, w = img_shape[:2]
    diagonal = np.sqrt(h**2 + w**2)
    intersections = []
    line_pairs = []
    n = len(lines)
    
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    
    if n > 50:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2-y1, x2-x1)
            angles.append(angle)
        
        indices = list(range(n))
        sampled_pairs = set()
        
        while len(sampled_pairs) < min(1000, n * (n-1) // 2, 500):
            i = random.choice(indices)
            angle_diff = [abs(angles[i] - angles[j]) for j in range(n)]
            weights = np.array(angle_diff)
            weights = weights / (weights.sum() + 1e-6)
            j = np.random.choice(indices, p=weights)
            
            if i != j:
                pair = tuple(sorted([i, j]))
                if pair not in sampled_pairs:
                    sampled_pairs.add(pair)
                    pt = line_intersection(lines[i], lines[j])
                    if pt and -10*w < pt[0] < 11*w and -10*h < pt[1] < 11*h:
                        intersections.append(pt)
                        line_pairs.append(pair)
    else:
        for i, j in combinations(range(n), 2):
            pt = line_intersection(lines[i], lines[j])
            if pt and -10*w < pt[0] < 11*w and -10*h < pt[1] < 11*h:
                intersections.append(pt)
                line_pairs.append((i, j))
    
    if len(intersections) < 3:
        return []
    
    pts = np.array(intersections)
    adaptive_eps = config["cluster_eps"] * (diagonal / 1000.0)
    adaptive_eps = max(0.1, min(0.5, adaptive_eps))
    pts_norm = pts / diagonal
    
    db = DBSCAN(eps=adaptive_eps, min_samples=config["cluster_min_samples"]).fit(pts_norm)
    
    vps = []
    for lbl in set(db.labels_) - {-1}:
        mask = db.labels_ == lbl
        cluster_pts = pts[mask]
        
        if len(cluster_pts) < 2:
            continue
        
        initial_vp = np.median(cluster_pts, axis=0)
        
        cluster_indices = set()
        for idx, (pair_idx, lbl_val) in enumerate(zip(line_pairs, db.labels_)):
            if lbl_val == lbl:
                cluster_indices.add(pair_idx[0])
                cluster_indices.add(pair_idx[1])
        
        cluster_lines = [lines[idx] for idx in cluster_indices if idx < len(lines)]
        
        if config["use_ransac"] and len(cluster_lines) >= 3:
            refined_vp, inliers = ransac_vp(cluster_lines, initial_vp,
                                           config["ransac_iterations"],
                                           config["ransac_threshold"])
            vp = refined_vp
            confidence = len(inliers)
        else:
            vp = initial_vp
            confidence = len(cluster_pts)
        
        vps.append({
            "point": (float(vp[0]), float(vp[1])),
            "confidence": confidence,
            "num_lines": len(cluster_indices)
        })
    
    vps.sort(key=lambda v: v["confidence"], reverse=True)
    return vps

def split_lines_by_angle(lines, min_angle=15):
    group1 = []
    group2 = []
    
    for x1, y1, x2, y2 in lines:
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1) + 1e-6))
        
        if abs(angle) < min_angle:
            continue
            
        if angle > 0:
            group1.append((x1, y1, x2, y2))
        else:
            group2.append((x1, y1, x2, y2))
    
    return group1, group2

def merge_close_vps(vps, distance_threshold=150):
    if len(vps) <= 2:
        return vps
    
    merged = []
    used = [False] * len(vps)
    
    for i in range(len(vps)):
        if used[i]:
            continue
        
        cluster = [vps[i]]
        used[i] = True
        
        for j in range(i+1, len(vps)):
            if used[j]:
                continue
            
            dist = np.sqrt((vps[i]['point'][0] - vps[j]['point'][0])**2 + 
                          (vps[i]['point'][1] - vps[j]['point'][1])**2)
            
            if dist < distance_threshold:
                cluster.append(vps[j])
                used[j] = True
        
        if len(cluster) == 1:
            merged.append(cluster[0])
        else:
            total_conf = sum(v['confidence'] for v in cluster)
            avg_point = (
                sum(v['point'][0] * v['confidence'] for v in cluster) / total_conf,
                sum(v['point'][1] * v['confidence'] for v in cluster) / total_conf
            )
            merged.append({
                "point": avg_point,
                "confidence": total_conf,
                "num_lines": sum(v.get('num_lines', 0) for v in cluster)
            })
    
    merged.sort(key=lambda v: v["confidence"], reverse=True)
    return merged[:2]

def visualize_vps(img, lines, vps):
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img_rgb)
    
    for x1, y1, x2, y2 in lines:
        ax.plot([x1, x2], [y1, y2], color="yellow", linewidth=0.8, alpha=0.4)
    
    colors = ["#00FF00", "#FF0000"]
    
    for i, vp_info in enumerate(vps[:2]):
        vp = vp_info["point"]
        color = colors[i % len(colors)]
        
        ax.plot(vp[0], vp[1], "X", color=color, markersize=14, 
                markeredgecolor="white", markeredgewidth=2, 
                label=f"VP {i+1} (conf: {vp_info['confidence']})")
        
        label = f"VP{i+1}: ({int(vp[0])}, {int(vp[1])})"
        ax.annotate(label, xy=(vp[0], vp[1]), xytext=(10, 10), 
                    textcoords="offset points",
                    color="white", fontweight="bold", fontsize=10,
                    bbox=dict(facecolor=color, alpha=0.8, pad=2))
        
        for angle in np.linspace(-np.pi/2, np.pi/2, 8):
            dx = np.cos(angle) * max(w, h) * 2
            dy = np.sin(angle) * max(w, h) * 2
            ax.plot([vp[0], vp[0] + dx], [vp[1], vp[1] + dy], 
                   color=color, linewidth=0.5, alpha=0.15)
    
    plt.title(f"Perspective Analysis: {len(vps)} Vanishing Point(s) Detected", fontsize=15, fontweight='bold')
    
    all_x = [v["point"][0] for v in vps[:2]] + [0, w]
    all_y = [v["point"][1] for v in vps[:2]] + [0, h]
    margin = max(w, h) * 0.1
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(max(all_y) + margin, min(all_y) - margin)
    
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    return fig

def process_image(img, config):
    """Main image processing function"""
    with st.spinner('🔄 Processing image...'):
        edges = preprocess(img)
        lines = detect_lines(edges, config)
        
        if len(lines) < 5:
            st.error(f"❌ Only {len(lines)} lines detected. Try adjusting the settings.")
            return None, None, None
        
        # Split lines by angle
        min_angle = config.get("min_angle", 15)
        g1, g2 = split_lines_by_angle(lines, min_angle)
        
        # Find vanishing points
        vps = []
        if len(g1) > 2:
            vps += find_vanishing_points(g1, img.shape, config)
        if len(g2) > 2:
            vps += find_vanishing_points(g2, img.shape, config)
        
        # Merge close points
        vps = merge_close_vps(vps, config.get("merge_distance", 150))
        
        return lines, vps, edges

def main():
    st.title("🎯 Vanishing Points Detector")
    st.markdown("### Detect vanishing points in your images with high accuracy")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/camera--v1.png", width=80)
        st.markdown("## ⚙️ Settings")
        
        # Load default or custom config
        use_advanced = st.checkbox("🔧 Advanced Settings", value=False)
        
        config = CONFIG.copy()
        
        if use_advanced:
            st.markdown("---")
            st.markdown("### 🎨 Detection Settings")
            config["hough_threshold"] = st.slider("Hough Threshold", 30, 150, CONFIG["hough_threshold"])
            config["min_line_length"] = st.slider("Min Line Length", 30, 150, CONFIG["min_line_length"])
            config["max_line_gap"] = st.slider("Max Line Gap", 5, 50, CONFIG["max_line_gap"])
            
            st.markdown("---")
            st.markdown("### 📐 Angle Settings")
            config["min_angle"] = st.slider("Min Angle (ignore horizontal lines)", 5, 30, 15)
            config["merge_distance"] = st.slider("Merge Distance (pixels)", 50, 300, 150)
            
            st.markdown("---")
            st.markdown("### 🎯 Vanishing Point Settings")
            config["cluster_eps"] = st.slider("DBSCAN Epsilon", 0.05, 0.5, CONFIG["cluster_eps"], 0.01)
            config["cluster_min_samples"] = st.slider("Min Samples for Cluster", 1, 10, CONFIG["cluster_min_samples"])
            config["use_ransac"] = st.checkbox("Use RANSAC (improves accuracy)", value=CONFIG["use_ransac"])
            
            if config["use_ransac"]:
                config["ransac_iterations"] = st.slider("RANSAC Iterations", 50, 300, CONFIG["ransac_iterations"])
                config["ransac_threshold"] = st.slider("RANSAC Threshold", 1.0, 15.0, CONFIG["ransac_threshold"], 0.5)
        else:
            st.info("💡 Enable Advanced Settings for full control")
        
        st.markdown("---")
        st.markdown("### 📖 How It Works")
        st.info("""
        **Vanishing points** are points where parallel lines appear to converge in an image.
        
        **Applications:**
        - 📐 Perspective correction
        - 🏗️ Architectural analysis
        - 🎨 3D reconstruction
        - 🚗 Computer vision for autonomous vehicles
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📤 Upload an image", 
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image with perspective (buildings, roads, corridors, etc.)"
        )
    
    with col2:
        st.markdown("### 🖼️ Example Images")
        example = st.selectbox("Try with:", ["Select an example", "Corridor", "Building", "Road"])
        
        if example != "Select an example":
            st.info(f"💡 Upload a photo of a {example} for best results")
    
    if uploaded_file is not None:
        # Load and display image
        img = load_image(uploaded_file)
        if img is None:
            st.error("❌ Failed to load image")
            return
        
        st.markdown("---")
        
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("### 🖼️ Original Image")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Process button
        if st.button("🚀 Start Analysis", use_container_width=True):
            # Process image
            lines, vps, edges = process_image(img, config)
            
            if lines and vps:
                with col_img2:
                    st.markdown("### 📊 Analysis Result")
                    fig = visualize_vps(img, lines, vps)
                    st.pyplot(fig, use_container_width=True)
                
                # Results in columns
                st.markdown("---")
                st.markdown("## 📈 Detailed Results")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.metric("📏 Detected Lines", len(lines))
                
                with col_res2:
                    st.metric("🎯 Vanishing Points", len(vps))
                
                with col_res3:
                    if len(vps) >= 2:
                        st.metric("📐 Perspective Type", "Two-point")
                    elif len(vps) == 1:
                        st.metric("📐 Perspective Type", "One-point")
                    else:
                        st.metric("📐 Perspective Type", "Undefined")
                
                # Display VP details
                if vps:
                    st.markdown("### 🎯 Detected Vanishing Points")
                    
                    vp_cols = st.columns(len(vps))
                    for i, (col, vp) in enumerate(zip(vp_cols, vps)):
                        with col:
                            st.success(f"**Vanishing Point {i+1}**")
                            st.write(f"📍 Location: `({int(vp['point'][0])}, {int(vp['point'][1])})`")
                            st.write(f"⭐ Confidence: `{vp['confidence']}`")
                            st.write(f"📊 Lines count: `{vp.get('num_lines', 'N/A')}`")
                
                # Option to download result - FIXED VERSION
                if len(vps) >= 1:
                    st.markdown("---")
                    st.success("✅ Analysis complete! You can save the result.")
                    
                    # Save to memory instead of temp file
                    from io import BytesIO
                    fig_temp = visualize_vps(img, lines, vps)
                    buf = BytesIO()
                    fig_temp.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                    plt.close(fig_temp)
                    buf.seek(0)
                    
                    st.download_button(
                        label="💾 Download Result",
                        data=buf,
                        file_name="vanishing_points_result.png",
                        mime="image/png",
                        use_container_width=True
                    )
            else:
                st.error("❌ Could not find enough vanishing points. Try adjusting the settings.")
                
                if lines:
                    st.warning(f"✅ {len(lines)} lines were detected but couldn't be clustered into vanishing points")
    
    else:
        # Placeholder when no image uploaded
        st.markdown("---")
        st.info("👈 **Upload an image to start analysis**")
        
        # Show instructions
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 📐 How It Works")
            st.markdown("""
            <div class="info-box">
            <b>🔍 Analysis Pipeline:</b><br>
            1️⃣ Edge detection using Canny algorithm<br>
            2️⃣ Line extraction using Hough Transform<br>
            3️⃣ Calculate line intersections<br>
            4️⃣ Cluster intersections using DBSCAN<br>
            5️⃣ Refine results using RANSAC<br>
            6️⃣ Display final vanishing points
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
            <b>💡 Tips for Best Results:</b><br>
            • Use images with clear parallel lines (buildings, corridors, roads)<br>
            • Ensure good lighting and contrast<br>
            • Avoid images with too many curves or organic shapes<br>
            • Adjust advanced settings if needed
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()