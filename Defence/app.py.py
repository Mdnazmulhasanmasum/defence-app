import io, json, time, hashlib
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageOps

# ───────────────────────── PAGE ─────────────────────────
st.set_page_config(page_title="TinyBrainNet-MT", page_icon="🧠", layout="wide")

# One accent, one surface colour, one radius. Everything else stays quiet.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--bg:#0d1117;--surface:#151b23;--line:#222b36;--text:#e6edf3;--mute:#8b98a7;
      --accent:#5eead4;--warn:#fbbf24;--bad:#f87171;--ok:#4ade80;--r:12px}
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif}
.stApp{background:var(--bg);color:var(--text)}
header[data-testid="stHeader"]{background:transparent}
footer,#MainMenu{visibility:hidden}
.block-container{max-width:1180px;padding-top:2.2rem;padding-bottom:4rem}
section[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--line)}

h1.t{font-size:1.7rem;font-weight:700;letter-spacing:-.02em;margin:0}
p.s{color:var(--mute);font-size:.92rem;margin:.25rem 0 1.4rem}

/* upload zone */
[data-testid="stFileUploaderDropzone"]{background:var(--surface);border:1.5px dashed #2f3b4a;
  border-radius:var(--r);padding:2.2rem 1rem;transition:border-color .2s,background .2s}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent);background:#17212b}
[data-testid="stFileUploaderDropzone"] button{background:var(--accent);color:#04201c;border:0;
  border-radius:8px;font-weight:600;padding:.5rem 1.1rem}
[data-testid="stFileUploaderDropzone"] small,[data-testid="stFileUploaderDropzone"] span{color:var(--mute)}

/* result pieces */
.banner{border-radius:var(--r);padding:14px 18px;margin:1.1rem 0;border:1px solid var(--line);
  border-left:4px solid var(--c);background:var(--surface)}
.banner b{color:var(--c)}
.banner div{color:var(--mute);font-size:.9rem;margin-top:3px;line-height:1.5}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:14px 16px}
.kpi .l{color:var(--mute);font-size:.78rem}
.kpi .v{font-size:1.5rem;font-weight:700;margin-top:2px;letter-spacing:-.01em}
.bar{margin:0 0 12px}
.bar .row{display:flex;justify-content:space-between;font-size:.88rem;margin-bottom:5px}
.bar .row span:last-child{color:var(--mute);font-variant-numeric:tabular-nums}
.bar .track{height:6px;background:var(--line);border-radius:99px;overflow:hidden}
.bar .fill{height:100%;border-radius:99px;background:#3b4756}
.bar.top .fill{background:var(--accent)}
.note{color:var(--mute);font-size:.78rem;margin-top:2.2rem;line-height:1.6}
button[data-baseweb="tab"]{font-weight:500}
</style>
""", unsafe_allow_html=True)

# ───────────────────────── MODEL ─────────────────────────
BASE = Path(__file__).resolve().parent
MODEL_PATHS = [BASE / "model" / "tinybrainnet.onnx", BASE / "tinybrainnet.onnx",
               BASE / "model" / "uninet.onnx", BASE.parent / "model" / "tinybrainnet.onnx"]
META_PATHS = [BASE / "model" / "meta.json", BASE / "meta.json", BASE.parent / "model" / "meta.json"]
SAMPLES_DIR = BASE / "samples" if (BASE / "samples").exists() else BASE.parent / "samples"

MODEL_PATH = next((p for p in MODEL_PATHS if p.exists()), None)
META_PATH = next((p for p in META_PATHS if p.exists()), None)

if MODEL_PATH is None:
    st.error("Model file not found. Put `tinybrainnet.onnx` inside a `model/` folder next to app.py.")
    st.stop()


def read_meta(p):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p else {}
    except Exception:
        return {}


@st.cache_resource(show_spinner="Loading model…")
def load_session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])


meta = read_meta(META_PATH)
sess = load_session(str(MODEL_PATH))

CLASSES = [str(c) for c in meta.get("classes", ["Glioma", "Meningioma", "Pituitary", "No_tumor"])]
NO_TUMOR = next((i for i, c in enumerate(CLASSES) if c.lower() == "no_tumor"), None)
INP = meta.get("input", {})
IMG = int(INP.get("size", 224))
MEAN = np.array(INP.get("mean", [0.1838] * 3), np.float32)
STD = np.array(INP.get("std", [0.1984] * 3), np.float32)
TEMP = max(float(meta.get("calibration", {}).get("temperature", 0.693)), 1e-4)
SEG_IS_PROB = bool(meta.get("outputs", {}).get("segmentation_is_probability", False))

IN_NAME = sess.get_inputs()[0].name
OUT_NAMES = [o.name for o in sess.get_outputs()][:3]  # cls, seg, attribution (optional)

# ───────────────────────── PIPELINE ─────────────────────────
def auto_crop(pil):
    g = np.array(pil.convert("L"))
    _, th = cv2.threshold(g, 22, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > g.size * 0.10:
            x, y, w, h = cv2.boundingRect(c)
            p = 6
            return pil.crop((max(0, x - p), max(0, y - p), min(g.shape[1], x + w + p), min(g.shape[0], y + h + p)))
    return pil


def preprocess(pil, crop, robust):
    pil = ImageOps.exif_transpose(pil).convert("RGB")  # fix rotation first, then crop
    if crop:
        pil = auto_crop(pil)
    rgb = cv2.resize(np.asarray(pil, np.uint8), (IMG, IMG), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    if robust:
        tissue = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) > 20
        if tissue.sum() > 100:
            lo, hi = np.percentile(x[tissue], [1, 99])
            x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    t = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    return rgb, t


def run_once(t):
    res = sess.run(OUT_NAMES, {IN_NAME: t})
    cls = np.squeeze(res[0]).astype(np.float64)
    seg = np.squeeze(res[1]).astype(np.float32) if len(res) > 1 else None
    attr = np.squeeze(res[2]).astype(np.float32) if len(res) > 2 else None
    return cls, seg, attr


@st.cache_data(show_spinner="Analysing scan…", max_entries=24)
def infer(img_bytes, crop, robust, tta):
    """Cached: moving sliders / switching tabs never re-runs the network."""
    disp, t = preprocess(Image.open(io.BytesIO(img_bytes)), crop, robust)
    t0 = time.perf_counter()
    cls, seg, attr = run_once(t)
    if tta:  # horizontal-flip test-time augmentation
        c2, s2, a2 = run_once(np.ascontiguousarray(t[..., ::-1]))
        cls = (cls + c2) / 2
        if seg is not None:
            seg = (seg + s2[..., ::-1]) / 2
        if attr is not None:
            attr = (attr + a2[..., ::-1]) / 2
    return disp, cls, seg, attr, (time.perf_counter() - t0) * 1000


def softmax(z, temp=1.0):
    z = np.asarray(z, np.float64).ravel() / temp
    e = np.exp(np.clip(z - z.max(), -60, 0))
    return e / e.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def blend_mask(img, mask, alpha):
    out = img.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array([248, 113, 113]) * alpha).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 1)
    return out


def blend_heat(img, attr, alpha):
    heat = cv2.cvtColor(cv2.applyColorMap((attr * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    return (img * (1 - alpha) + heat * alpha).astype(np.uint8)


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ───────────────────────── SIDEBAR ─────────────────────────
with st.sidebar:
    st.markdown("#### Settings")
    samples = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in SAMPLES_DIR.glob(ext)) if SAMPLES_DIR.exists() else []
    sample_choice = st.selectbox("Try a sample", ["None"] + [p.name for p in samples])
    st.divider()
    crop = st.toggle("Auto-crop brain", True)
    robust = st.toggle("Intensity alignment", True)
    tta = st.toggle("Flip averaging (slower, steadier)", False)
    seg_thr = st.slider("Segmentation threshold", 0.10, 0.90, 0.50, 0.05)
    alpha = st.slider("Overlay strength", 0.1, 0.9, 0.5, 0.05)
    st.divider()
    st.caption("Khwaja Yunus Ali University · CSE thesis prototype")

# ───────────────────────── HEADER + UPLOAD ─────────────────────────
st.markdown('<h1 class="t">🧠 TinyBrainNet-MT</h1>'
            '<p class="s">Upload a brain MRI slice to classify the tumour type and outline the lesion.</p>',
            unsafe_allow_html=True)

upload = st.file_uploader("Upload MRI", type=["jpg", "jpeg", "png"], label_visibility="collapsed",
                          help="JPG or PNG, single axial / coronal / sagittal slice")

img_bytes = None
if upload is not None:
    img_bytes = upload.getvalue()
elif sample_choice != "None":
    img_bytes = (SAMPLES_DIR / sample_choice).read_bytes()

if img_bytes is None:
    st.caption("Drag and drop a scan above, or pick a sample from the sidebar.")
    st.stop()

# ───────────────────────── INFERENCE ─────────────────────────
try:
    disp, cls_logits, seg_raw, attr_raw, ms = infer(img_bytes, crop, robust, tta)
except Exception as e:
    st.error(f"Could not analyse this image: {e}")
    st.stop()

probs = softmax(cls_logits, TEMP)
pred = int(probs.argmax())
conf = float(probs[pred])
entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(len(probs)))  # 0 = sure, 1 = unsure

if seg_raw is not None:
    sp = seg_raw if SEG_IS_PROB else sigmoid(seg_raw)
    sp = cv2.resize(sp, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
    mask = sp >= seg_thr
else:
    mask = np.zeros((IMG, IMG), bool)
area = float(mask.mean()) * 100

# Disagreement between the two heads: classifier says "no tumour" but segmentation found a mass.
conflict = NO_TUMOR is not None and pred == NO_TUMOR and area >= 0.40
final_idx, final_conf = pred, conf
if conflict:
    idxs = [i for i in range(len(CLASSES)) if i != NO_TUMOR]
    sub = probs[idxs] / probs[idxs].sum()
    j = int(sub.argmax())
    final_idx, final_conf = idxs[j], float(sub[j])
final_name = CLASSES[final_idx]

# Saliency for the class that is actually reported
attr = None
if attr_raw is not None and attr_raw.ndim == 3:
    k = final_idx if final_idx < attr_raw.shape[0] else 0
    a = cv2.resize(np.maximum(attr_raw[k], 0), (IMG, IMG), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(a, [1, 99.5])
    if hi > lo + 1e-7:
        attr = np.clip((a - lo) / (hi - lo), 0, 1)

# ───────────────────────── RESULT ─────────────────────────
if conflict:
    color, title = "var(--warn)", "Heads disagree, please review"
    body = (f"The classifier says <b style='color:var(--text)'>No_tumor ({conf*100:.1f}%)</b>, but the segmentation head "
            f"outlined a mass covering <b style='color:var(--text)'>{area:.2f}%</b> of the slice. "
            f"The most likely tumour type is <b style='color:var(--text)'>{final_name}</b> "
            f"({final_conf*100:.1f}% among tumour classes).")
elif conf >= 0.70:
    color, title = ("var(--ok)", f"{final_name} detected") if final_name.lower() != "no_tumor" else ("var(--ok)", "No tumour detected")
    body = f"Calibrated confidence {conf*100:.1f}%."
else:
    color, title = "var(--warn)", "Low confidence"
    body = f"Best guess is {final_name} at {conf*100:.1f}%. Treat this result with caution."

st.markdown(f'<div class="banner" style="--c:{color}"><b>{title}</b><div>{body}</div></div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
for col, lab, val in [(k1, "Diagnosis", final_name), (k2, "Confidence", f"{final_conf*100:.1f}%"),
                      (k3, "Lesion area", f"{area:.2f}%"), (k4, "Uncertainty", f"{entropy*100:.0f}%")]:
    col.markdown(f'<div class="kpi"><div class="l">{lab}</div><div class="v">{val}</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([3, 2], gap="large")

overlay = blend_mask(disp, mask, alpha) if mask.any() else disp
heat = blend_heat(disp, attr, alpha) if attr is not None else None
mask_img = (np.stack([mask] * 3, -1) * 255).astype(np.uint8)

with left:
    names = ["Segmentation", "Saliency", "Input"] if heat is not None else ["Segmentation", "Input"]
    for tab, n in zip(st.tabs(names), names):
        with tab:
            st.image({"Segmentation": overlay, "Saliency": heat, "Input": disp}[n], use_container_width=True)
    st.caption(f"{ms:.0f} ms on CPU{' · flip averaging on' if tta else ''}")

with right:
    st.markdown("**Class probabilities**")
    html = ""
    for i in np.argsort(-probs):
        top = " top" if i == pred else ""
        html += (f'<div class="bar{top}"><div class="row"><span>{CLASSES[i]}</span><span>{probs[i]*100:.1f}%</span></div>'
                 f'<div class="track"><div class="fill" style="width:{probs[i]*100:.1f}%"></div></div></div>')
    st.markdown(html, unsafe_allow_html=True)

# ───────────────────────── EXPORT ─────────────────────────
panels = [label(disp, "Input"), label(overlay, f"Lesion {area:.2f}%")]
if heat is not None:
    panels.append(label(heat, "Saliency"))
buf = io.BytesIO()
Image.fromarray(np.hstack(panels)).save(buf, format="PNG")

report = {
    "diagnosis": final_name,
    "confidence": round(final_conf, 4),
    "raw_prediction": CLASSES[pred],
    "heads_disagree": bool(conflict),
    "lesion_area_percent": round(area, 3),
    "uncertainty": round(entropy, 4),
    "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
    "settings": {"threshold": seg_thr, "auto_crop": crop, "intensity_alignment": robust, "flip_averaging": tta},
    "image_sha1": hashlib.sha1(img_bytes).hexdigest()[:12],
}

st.write("")
d1, d2, _ = st.columns([1, 1, 2])
d1.download_button("Download report (JSON)", json.dumps(report, indent=2), "report.json", "application/json", use_container_width=True)
d2.download_button("Download panels (PNG)", buf.getvalue(), "panels.png", "image/png", use_container_width=True)

st.markdown('<div class="note">Research prototype for a thesis project. Not a medical device and not for clinical decisions.</div>',
            unsafe_allow_html=True)
