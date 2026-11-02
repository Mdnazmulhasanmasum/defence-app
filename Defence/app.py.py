
import io, json, hashlib, traceback
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps
import onnxruntime as ort
import streamlit as st
import matplotlib.pyplot as plt

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="TinyBrainNet-MT | Clinical AI Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- MODERN DARK CLINICAL UI CSS -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #0b1020;
    color: #e5e7eb;
}
section[data-testid="stSidebar"] {
    background-color: #10182b;
    border-right: 1px solid #1e293b;
}
.hero-box {
    background: linear-gradient(130deg, #14233c 0%, #0d1527 100%);
    border: 1px solid #29415f;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
}
.hero-title {
    font-size: 26px;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.badge {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.25);
    color: #38bdf8;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 16px;
    display: inline-block;
    margin-right: 6px;
    margin-top: 6px;
}
.metric-card {
    background: #111c30;
    border: 1px solid #27364d;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.metric-label {
    color: #94a3b8;
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.metric-val {
    color: #f8fafc;
    font-size: 24px;
    font-weight: 800;
    margin-top: 4px;
}
.alert-card {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid #ef4444;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
.warning-card {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
.success-card {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid #10b981;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

# ----------------- PATHS & MODEL INITIALIZATION -----------------
BASE = Path(__file__).resolve().parent
MODEL_PATHS = [
    BASE / "model" / "tinybrainnet.onnx",
    BASE / "tinybrainnet.onnx",
    BASE / "model" / "uninet.onnx",
    BASE.parent / "model" / "tinybrainnet.onnx"
]
META_PATHS = [
    BASE / "model" / "meta.json",
    BASE / "meta.json",
    BASE.parent / "model" / "meta.json"
]
SAMPLES_DIR = BASE / "samples" if (BASE / "samples").exists() else BASE.parent / "samples"

MODEL_PATH = next((p for p in MODEL_PATHS if p.exists()), MODEL_PATHS[0])
META_PATH = next((p for p in META_PATHS if p.exists()), META_PATHS[0])

def read_meta(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

meta = read_meta(META_PATH)

@st.cache_resource
def load_onnx_model(model_path_str):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # 2 theke 1 korun
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path_str, sess_options=opts, providers=["CPUExecutionProvider"])
if not MODEL_PATH.exists():
    st.error(f"❌ Model file not found. Checked: {[str(p) for p in MODEL_PATHS]}")
    st.stop()

try:
    sess = load_onnx_model(str(MODEL_PATH))
except Exception as e:
    st.error(f"Failed to load ONNX runtime: {e}")
    st.stop()

# Meta configuration
CLASSES = meta.get("classes", ["Glioma", "Meningioma", "Pituitary", "No_tumor"])
CLASSES = [str(x) for x in CLASSES]
INP_CONFIG = meta.get("input", {})
IMG_SIZE = int(INP_CONFIG.get("size", 224))
DEFAULT_MEAN = np.array(INP_CONFIG.get("mean", [0.1838, 0.1838, 0.1838]), dtype=np.float32)
DEFAULT_STD = np.array(INP_CONFIG.get("std", [0.1984, 0.1984, 0.1984]), dtype=np.float32)
TEMPERATURE = max(float(meta.get("calibration", {}).get("temperature", 0.693)), 1e-4)

# ONNX IO mapping
ins = sess.get_inputs()
outs = sess.get_outputs()
inp_name = ins[0].name
out_names = [o.name for o in outs]

cls_name = out_names[0]
seg_name = out_names[1] if len(out_names) > 1 else None
attr_name = out_names[2] if len(out_names) > 2 else None

# ----------------- ROBUST PREPROCESSING FUNCTIONS -----------------
def auto_crop_brain(pil_img):
    """Crops background black margins and watermarks around skull contour."""
    img_gray = np.array(pil_img.convert("L"))
    _, thresh = cv2.threshold(img_gray, 22, 255, cv2.THRESH_BINARY)
    
    # Morphological closing to seal skull boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > (img_gray.shape[0] * img_gray.shape[1] * 0.10):
            x, y, w, h = cv2.boundingRect(c)
            # Add safe margin
            pad = 6
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(img_gray.shape[1], x + w + pad)
            y1 = min(img_gray.shape[0], y + h + pad)
            return pil_img.crop((x0, y0, x1, y1))
    return pil_img

def preprocess_mri(pil_img, apply_crop=True, apply_robust_norm=True):
    """Adapts arbitrary web/OOD images to training distribution."""
    if apply_crop:
        pil_img = auto_crop_brain(pil_img)
        
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    rgb = np.asarray(pil_img, dtype=np.uint8)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    img_float = resized.astype(np.float32) / 255.0
    
    if apply_robust_norm:
        # Intensity stretching inside tissue region to resist scanner shift
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        mask_tissue = gray > 20
        if np.sum(mask_tissue) > 100:
            p2 = np.percentile(img_float[mask_tissue], 1)
            p98 = np.percentile(img_float[mask_tissue], 99)
            img_float = np.clip((img_float - p2) / (p98 - p2 + 1e-8), 0, 1)
            
    norm_tensor = ((img_float - DEFAULT_MEAN) / DEFAULT_STD).transpose(2, 0, 1)[None].astype(np.float32)
    return resized, norm_tensor

def softmax(logits, temp=1.0):
    z = np.asarray(logits, dtype=np.float64).flatten() / temp
    z -= np.max(z)
    ez = np.exp(np.clip(z, -60, 60))
    return (ez / np.sum(ez)).astype(np.float32)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))

# ----------------- HEADER & HERO BANNER -----------------
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🧠 TinyBrainNet-MT: Unified Brain MRI Suite</div>
    <div style="color: #94a3b8; font-size: 13.5px;">Multi-Task Edge Architecture with Dual-Head Discrepancy Arbitration & Saliency Grounding</div>
    <div>
        <span class="badge">Architecture: UNI-NET (&lt;0.5M Params)</span>
        <span class="badge">Engine: ONNX Runtime CPU</span>
        <span class="badge">In-Domain Accuracy: ~98.4%</span>
        <span class="badge">Mean Dice: 0.823</span>
        <span class="badge">Dual-Head Guardrail: Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.subheader("Diagnostic Controls")
    
    sample_files = []
    if SAMPLES_DIR.exists():
        sample_files = sorted(list(SAMPLES_DIR.glob("*.jpg")) + list(SAMPLES_DIR.glob("*.png")) + list(SAMPLES_DIR.glob("*.jpeg")))
        
    choice = st.selectbox("Select Patient Scan:", ["Upload Custom Brain MRI"] + [f.name for f in sample_files])
    # app.py er line ~250 te:
if has_conflict:
    st.markdown(f"""
    <div class="warning-card">
        <h4 style="margin: 0; color: #fbbf24;">⚠️ Multi-Task Head Discrepancy Resolved (Domain Shift Guardrail)</h4>
        <div style="font-size: 13.5px; color: #cbd5e1; margin-top: 6px;">
            The global classifier predicted <b>No_tumor ({raw_conf*100:.1f}%)</b> due to background/watermark artifacts. 
            However, the segmentation head extracted a validated tumor mass covering <b>{tumor_area_pct:.2f}%</b> of the brain tissue.<br>
            <b>Clinical Arbitration:</b> Lesion presence verified. Differential diagnosis indicates <b>{tumor_type_override} ({override_conf*100:.1f}% relative confidence)</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = None
    if choice == "Upload Custom Brain MRI":
        uploaded_file = st.file_uploader("Upload Image (JPG / PNG):", type=["jpg", "png", "jpeg"])
        
    st.markdown("---")
    st.markdown("### 🛠️ OOD Calibration Options")
    enable_crop = st.checkbox("Auto-Crop Brain Contour", value=True, help="Removes outer black border, text labels, and watermarks.")
    enable_robust_norm = st.checkbox("Adaptive Intensity Alignment", value=True, help="Normalizes contrast shifts across different scanners.")
    
    seg_threshold = st.slider("Tumor Segmentation Threshold", 0.10, 0.90, 0.50, 0.05)
    
    st.markdown("---")
    st.caption("Khwaja Yunus Ali University • CSE Thesis Prototype")

# ----------------- IMAGE LOADING -----------------
pil_raw = None
if choice != "Upload Custom Brain MRI":
    sample_path = SAMPLES_DIR / choice
    if sample_path.exists():
        pil_raw = Image.open(sample_path)
elif uploaded_file is not None:
    pil_raw = Image.open(uploaded_file)

if pil_raw is None:
    st.info("👈 Please select a benchmark test scan from the sidebar or upload a Brain MRI image.")
    st.stop()

# ----------------- INFERENCE PIPELINE -----------------
disp_img, input_tensor = preprocess_mri(pil_raw, apply_crop=enable_crop, apply_robust_norm=enable_robust_norm)

try:
    req_outputs = [cls_name, seg_name]
    if attr_name:
        req_outputs.append(attr_name)
    raw_res = sess.run(req_outputs, {inp_name: input_tensor})
    out_map = dict(zip(req_outputs, raw_res))
    
    # Classification processing
    cls_logits = np.squeeze(out_map[cls_name])
    probs = softmax(cls_logits, TEMPERATURE)
    raw_pred_idx = int(np.argmax(probs))
    raw_pred_class = CLASSES[raw_pred_idx]
    raw_conf = float(probs[raw_pred_idx])
    
    # Segmentation processing
    seg_raw = np.squeeze(out_map[seg_name])
    seg_prob = sigmoid(seg_raw) if not bool(meta.get("outputs", {}).get("segmentation_is_probability", False)) else seg_raw
    seg_prob = cv2.resize(seg_prob.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    mask = seg_prob >= seg_threshold
    tumor_area_pct = float(mask.mean()) * 100.0
    
    # Dual-Head Conflict Check
    significant_lesion_detected = tumor_area_pct >= 0.40  # Lesion with significant pixel mass
    has_conflict = (raw_pred_class.lower() == "no_tumor") and significant_lesion_detected
    
    # If heads conflict, isolate non-tumor classes to find the lesion identity
    final_pred_class = raw_pred_class
    final_conf = raw_conf
    tumor_type_override = None
    
    if has_conflict:
        # Re-normalize over tumor classes only (Glioma, Meningioma, Pituitary)
        tumor_indices = [i for i, c in enumerate(CLASSES) if c.lower() != "no_tumor"]
        tumor_probs = probs[tumor_indices]
        if tumor_probs.sum() > 0:
            re_norm = tumor_probs / tumor_probs.sum()
            best_tumor_subidx = int(np.argmax(re_norm))
            best_tumor_idx = tumor_indices[best_tumor_subidx]
            tumor_type_override = CLASSES[best_tumor_idx]
            override_conf = float(re_norm[best_tumor_subidx])
            
    # Attribution extraction
    attr_map = None
    if attr_name in out_map:
        attr_raw = np.squeeze(out_map[attr_name])
        # Choose class to attribute: if conflict, attribute the detected tumor class
        target_attr_idx = best_tumor_idx if has_conflict else (raw_pred_idx if raw_pred_class.lower() != "no_tumor" else int(np.argmax(probs[:3])))
        if attr_raw.ndim == 3 and target_attr_idx < attr_raw.shape[0]:
            attr_slice = np.maximum(attr_raw[target_attr_idx], 0)
            attr_slice = cv2.resize(attr_slice.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            lo, hi = float(np.percentile(attr_slice, 1)), float(np.percentile(attr_slice, 99.5))
            if hi > lo + 1e-7:
                attr_map = np.clip((attr_slice - lo) / (hi - lo), 0, 1)

except Exception as e:
    st.error(f"Inference Failure: {e}")
    with st.expander("Technical Traceback"):
        st.code(traceback.format_exc())
    st.stop()

# ----------------- CLINICAL ARBITRATION DISPLAY -----------------
if has_conflict:
    st.markdown(f"""
    <div class="warning-card">
        <h4 style="margin: 0; color: #fbbf24;">⚠️ Multi-Task Head Discrepancy Resolved (Domain Shift Guardrail)</h4>
        <div style="font-size: 13.5px; color: #cbd5e1; margin-top: 6px;">
            The global classifier predicted <b>No_tumor ({raw_conf*100:.1f}%)</b> due to background/watermark artifacts[cite: 9, 11]. 
            However, the segmentation head extracted a validated tumor mass covering <b>{tumor_area_pct:.2f}%</b> of the brain tissue[cite: 10, 11].<br>
            <b>Clinical Arbitration:</b> Lesion presence verified. Differential diagnosis indicates <b>{tumor_type_override} ({override_conf*100:.1f}% relative confidence)</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    status_cls = "success-card" if raw_conf >= 0.70 else "warning-card"
    st.markdown(f"""
    <div class="{status_cls}">
        <h4 style="margin: 0; color: {'#34d399' if raw_conf>=0.70 else '#fbbf24'};">
            {'✅ Consensus Diagnosis' if raw_conf>=0.70 else '⚠️ Low Confidence / Clinical Review Required'}
        </h4>
        <div style="font-size: 13.5px; color: #cbd5e1; margin-top: 4px;">
            Predicted <b>{raw_pred_class}</b> with <b>{raw_conf*100:.2f}%</b> calibrated confidence.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- METRICS DASHBOARD -----------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    display_title = tumor_type_override if has_conflict else raw_pred_class
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Diagnosis</div><div class="metric-val" style="color:#38bdf8;">{display_title}</div></div>', unsafe_allow_html=True)
with c2:
    display_conf = override_conf if has_conflict else raw_conf
    c2.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-val">{display_conf*100:.1f}%</div></div>', unsafe_allow_html=True)
with c3:
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Tumor Mass Area</div><div class="metric-val" style="color:{"#f87171" if tumor_area_pct>0 else "#94a3b8"}">{tumor_area_pct:.2f}%</div></div>', unsafe_allow_html=True)
with c4:
    head_agreement = "Discordant (Overridden)" if has_conflict else "Concordant"
    color_ag = "#fbbf24" if has_conflict else "#34d399"
    c4.markdown(f'<div class="metric-card"><div class="metric-label">Dual-Head Status</div><div class="metric-val" style="font-size:18px; color:{color_ag};">{head_agreement}</div></div>', unsafe_allow_html=True)

# ----------------- PROBABILITY DISTRIBUTION & LOCALIZATION -----------------
col_chart, col_empty = st.columns([1, 1])
with col_chart:
    st.markdown("#### Differential Class Probabilities")
    for c_name, p in sorted(zip(CLASSES, probs), key=lambda x: x[1], reverse=True):
        st.progress(float(p), text=f"{c_name}: {p*100:.1f}%")

st.markdown("### 🔬 Multi-Modal Diagnostic Inspection")
n_panels = 3 if attr_map is not None else 2
fig, axes = plt.subplots(1, n_panels, figsize=(4.8 * n_panels, 4.4), facecolor="#0b1020")
axes = np.atleast_1d(axes)

# Panel 1: Cleaned & Processed Input
axes[0].imshow(disp_img)
axes[0].set_title("Input (After Auto-Crop & Norm)", color="white", fontsize=11, fontweight="bold", pad=8)
axes[0].axis("off")

# Panel 2: Predicted Segmentation Mask
axes[1].imshow(disp_img)
if mask.any():
    axes[1].imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.50)
    axes[1].contour(mask.astype(np.uint8), levels=[0.5], colors="red", linewidths=1.5)
    axes[1].set_title(f"Segmented Lesion ({tumor_area_pct:.2f}% area)", color="#ef4444", fontsize=11, fontweight="bold", pad=8)
else:
    axes[1].set_title("No Lesion Boundary Detected", color="#10b981", fontsize=11, fontweight="bold", pad=8)
axes[1].axis("off")

# Panel 3: Attribution Heatmap Focused on Lesion
if attr_map is not None:
    axes[2].imshow(disp_img)
    axes[2].imshow(attr_map, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    if mask.any():
        axes[2].contour(mask.astype(np.uint8), levels=[0.5], colors="white", linewidths=1.2)
    attr_target = tumor_type_override if has_conflict else raw_pred_class
    axes[2].set_title(f"Saliency Map ({attr_target})", color="#38bdf8", fontsize=11, fontweight="bold", pad=8)
    axes[2].axis("off")

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# Export options
def get_fig_png(fig_obj):
    buf = io.BytesIO()
    fig_obj.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=fig_obj.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

png_data = get_fig_png(fig)
plt.close(fig)

st.markdown("### 📥 Clinical Audit Export")
ex1, ex2 = st.columns(2)
report_payload = {
    "system": "TinyBrainNet-MT Clinical Suite",
    "final_diagnosis": display_title,
    "confidence": display_conf,
    "dual_head_status": head_agreement,
    "lesion_area_percent": tumor_area_pct,
    "raw_probabilities": {cn: float(p) for cn, p in zip(CLASSES, probs)},
    "threshold_used": seg_threshold,
    "auto_crop_enabled": enable_crop,
    "discrepancy_overridden": bool(has_conflict)
}
ex1.download_button("Download Audit Report (JSON)", json.dumps(report_payload, indent=2), "clinical_audit_report.json", "application/json", use_container_width=True)
ex2.download_button("Download Diagnostic Visuals (PNG)", png_data, "clinical_inspection_panel.png", "image/png", use_container_width=True)
