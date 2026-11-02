"""
TinyBrainNet / UNI-NET — Research Deployment Dashboard
Run: streamlit run app.py
Expected:
  model/tinybrainnet.onnx
  model/meta.json (optional but recommended)
  samples/ (optional JPG/PNG)
Expected ONNX outputs by default:
  output 0: classification logits [1,C]
  output 1: segmentation logits/probabilities [1,1,H,W]
  output 2: class-conditioned attribution [1,C,H,W] (optional)
Research prototype only; not for clinical decisions.
"""
from __future__ import annotations
import io, json, hashlib, traceback
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

st.set_page_config(page_title="TinyBrainNet | Research MRI Dashboard", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:#0b1020;color:#e5e7eb}
section[data-testid="stSidebar"]{background:#10182b}
.hero{padding:1.35rem 1.55rem;border-radius:16px;margin-bottom:1rem;
background:linear-gradient(120deg,#14233c,#111827);border:1px solid #29415f}
.hero h1{margin:0;font-size:1.8rem;color:#e0f2fe}
.hero p{margin:.45rem 0 0;color:#a8b7ca}
.metric-card{background:#111c30;border:1px solid #27364d;border-radius:12px;padding:14px 16px;min-height:92px}
.metric-label{color:#9caec4;font-size:.78rem;text-transform:uppercase}
.metric-value{color:#f8fafc;font-size:1.45rem;font-weight:800;margin-top:5px}
.notice{border:1px solid #8b6b2c;background:#2a2111;color:#f7dfaa;padding:12px 14px;border-radius:10px;margin:12px 0}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
MODEL_CANDIDATES = [BASE/"model"/"tinybrainnet.onnx", BASE/"model"/"uninet.onnx", BASE/"tinybrainnet.onnx"]
META_CANDIDATES = [BASE/"model"/"meta.json", BASE/"meta.json"]
SAMPLES = BASE/"samples"

def first_existing(paths):
    return next((p for p in paths if p.exists()), paths[0])

MODEL_PATH, META_PATH = first_existing(MODEL_CANDIDATES), first_existing(META_CANDIDATES)

def nested(obj, keys, default):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur: return default
        cur = cur[key]
    return cur

def read_meta(path):
    if not path.exists(): return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        st.error(f"meta.json parse error: {e}")
        return {}

@st.cache_resource
def load_onnx(path_text, mtime):
    del mtime
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 2, 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path_text, sess_options=opts, providers=["CPUExecutionProvider"])

def softmax(logits, temperature):
    z = np.asarray(logits, dtype=np.float64).reshape(-1) / max(float(temperature), 1e-6)
    z -= z.max()
    e = np.exp(np.clip(z, -80, 80))
    if not np.isfinite(e.sum()) or e.sum() <= 0: raise ValueError("Invalid classification logits.")
    return (e/e.sum()).astype(np.float32)

def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float32), -40, 40)
    return 1/(1+np.exp(-x))

def entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1)
    return float(-(p*np.log2(p)).sum())

def prep_image(pil, size, mean, std, color_order):
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    rgb = np.asarray(pil, dtype=np.uint8)
    resized = cv2.resize(rgb, (size,size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32)/255.
    if color_order == "BGR": x = x[..., ::-1].copy()
    mean, std = np.asarray(mean,dtype=np.float32).reshape(1,1,3), np.asarray(std,dtype=np.float32).reshape(1,1,3)
    if np.any(std <= 0): raise ValueError("meta.json input.std must contain positive values.")
    return resized, ((x-mean)/std).transpose(2,0,1)[None].astype(np.float32)

def squeeze_logits(a, n):
    out = np.squeeze(np.asarray(a))
    if out.ndim != 1 or out.size != n:
        raise ValueError(f"Classification output shape {np.asarray(a).shape}; expected {n} logits.")
    return out.astype(np.float32)

def squeeze_mask(a):
    out = np.squeeze(np.asarray(a))
    if out.ndim != 2: raise ValueError(f"Segmentation output must reduce to HxW, got {np.asarray(a).shape}.")
    return out.astype(np.float32)

def attribution_map(a, class_idx, size, is_probability=False):
    if a is None: return None
    a = np.squeeze(np.asarray(a))
    if a.ndim == 3:
        if class_idx >= a.shape[0]: return None
        a = a[class_idx]
    if a.ndim != 2: return None
    if not is_probability: a = np.maximum(a, 0)
    a = cv2.resize(a.astype(np.float32), (size,size), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(np.percentile(a,1)), float(np.percentile(a,99.5))
    if hi <= lo+1e-8: return np.zeros_like(a)
    return np.clip((a-lo)/(hi-lo),0,1)

def fig_png(fig):
    b=io.BytesIO()
    fig.savefig(b,format="png",dpi=180,bbox_inches="tight",facecolor=fig.get_facecolor())
    b.seek(0)
    return b.getvalue()

st.markdown('<div class="hero"><h1>🧠 TinyBrainNet / UNI-NET</h1><p>ONNX-powered brain MRI research dashboard · classification · segmentation · attribution</p></div>', unsafe_allow_html=True)
st.markdown('<div class="notice"><b>Research prototype only.</b> These are experimental model outputs, not a medical diagnosis, clinical triage decision, or substitute for a qualified radiologist. Do not use for patient-care decisions.</div>', unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("ONNX model not found. Expected: " + " | ".join(map(str,MODEL_CANDIDATES)))
    st.stop()
meta=read_meta(META_PATH)
try: sess=load_onnx(str(MODEL_PATH),MODEL_PATH.stat().st_mtime)
except Exception as e: st.error(f"ONNX load failed: {e}"); st.stop()

classes=meta.get("classes",["Glioma","Meningioma","Pituitary","No_tumor"])
if not isinstance(classes,list) or len(classes)<2: st.error("meta.json 'classes' must be a list."); st.stop()
classes=[str(x) for x in classes]
inp=nested(meta,("input",),{})
size=int(inp.get("size",224))
mean=inp.get("mean",[0.1838]*3); std=inp.get("std",[0.1984]*3)
temperature=max(float(nested(meta,("calibration","temperature"),1.0)),1e-6)
color_order=str(inp.get("color_order","RGB")).upper()
ins=sess.get_inputs(); outs=sess.get_outputs()
input_name=meta.get("onnx_input_name",ins[0].name)
outmeta=nested(meta,("outputs",),{})
outnames=[o.name for o in outs]
cls_name=outmeta.get("classification",outnames[0])
seg_name=outmeta.get("segmentation",outnames[1] if len(outnames)>1 else None)
attr_name=outmeta.get("attribution",outnames[2] if len(outnames)>2 else None)
if input_name not in [x.name for x in ins]: st.error(f"Unknown ONNX input {input_name}. Available: {[x.name for x in ins]}"); st.stop()
if cls_name not in outnames or seg_name not in outnames: st.error(f"Output mapping invalid. Available outputs: {outnames}"); st.stop()
if attr_name not in outnames: attr_name=None
seg_is_prob=bool(outmeta.get("segmentation_is_probability",False))
attr_is_prob=bool(outmeta.get("attribution_is_probability",False))

with st.sidebar:
    st.subheader("Input MRI")
    sample_files=sorted(list(SAMPLES.glob("*.jpg"))+list(SAMPLES.glob("*.jpeg"))+list(SAMPLES.glob("*.png"))) if SAMPLES.exists() else []
    choice=st.selectbox("Choose input",["Upload an MRI image"]+[p.name for p in sample_files])
    upload=st.file_uploader("Upload JPG / JPEG / PNG",type=["jpg","jpeg","png"]) if choice=="Upload an MRI image" else None
    st.divider()
    threshold=st.slider("Segmentation threshold",0.05,0.95,float(nested(meta,("segmentation","threshold"),0.5)),0.01,
                        help="Changes displayed mask threshold only; does not retrain the model.")
    show_attr=st.checkbox("Show model attribution output",True)
    st.divider()
    st.caption(f"Model: {MODEL_PATH.name}")
    st.caption(f"Runtime: ONNX Runtime CPU")
    st.caption(f"Input: {size} × {size} · {color_order}")

pil=None
try:
    if choice=="Upload an MRI image" and upload is not None:
        pil=Image.open(upload); pil.load()
    elif choice!="Upload an MRI image":
        pil=Image.open(SAMPLES/choice); pil.load()
except Exception as e: st.error(f"Could not read image: {e}")

if pil is None:
    st.info("Choose a bundled sample or upload a JPG/PNG image.")
    st.stop()

try:
    pil=ImageOps.exif_transpose(pil).convert("RGB")
    ow,oh=pil.size
    if min(ow,oh)<32 or max(ow,oh)>12000: raise ValueError("Unsupported image dimensions.")
    display,x=prep_image(pil,size,mean,std,color_order)
    requested=[cls_name,seg_name]
    if show_attr and attr_name: requested.append(attr_name)
    raw=sess.run(requested,{input_name:x})
    outputs=dict(zip(requested,raw))
    logits=squeeze_logits(outputs[cls_name],len(classes))
    probs=softmax(logits,temperature)
    idx=int(np.argmax(probs)); pred=classes[idx]; conf=float(probs[idx])
    top2=float(np.partition(probs,-2)[-2]); margin=conf-top2
    norm_entropy=entropy(probs)/np.log2(len(classes))
    segraw=squeeze_mask(outputs[seg_name])
    segprob=np.clip(segraw,0,1) if seg_is_prob else sigmoid(segraw)
    segprob=cv2.resize(segprob,(size,size),interpolation=cv2.INTER_LINEAR)
    mask=segprob>=threshold
    attr=attribution_map(outputs.get(attr_name) if attr_name else None,idx,size,attr_is_prob) if show_attr else None
    image_hash=hashlib.sha256(np.asarray(pil).tobytes()).hexdigest()[:16]
except Exception as e:
    st.error(f"Inference failed: {e}")
    with st.expander("Technical traceback"): st.code(traceback.format_exc())
    st.stop()

st.subheader("Prediction summary")
a,b,c,d=st.columns(4)
for col,label,value in [
    (a,"Predicted class",pred),(b,"Top-class probability",f"{conf*100:.2f}%"),
    (c,"Top-2 margin",f"{margin*100:.2f} pp"),(d,"Normalized entropy",f"{norm_entropy:.3f}")
]:
    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',unsafe_allow_html=True)
st.caption("Probability uses temperature from meta.json. It is not a guarantee of correctness and may not be calibrated for this image source.")
if norm_entropy>=0.75 or margin<0.15:
    st.warning("Relatively uncertain model output (high entropy or small top-2 margin). Interpret cautiously.")
else:
    st.info("The output distribution is comparatively concentrated; this does not establish diagnostic certainty.")

l,r=st.columns([1,1])
with l:
    st.markdown("#### Class probabilities")
    for name,p in sorted(zip(classes,probs),key=lambda z:z[1],reverse=True):
        st.progress(float(p),text=f"{name}: {p*100:.2f}%")
    st.dataframe({"Class":classes,"Probability":[float(v) for v in probs]},hide_index=True,use_container_width=True)
with r:
    st.markdown("#### Segmentation summary")
    area=float(mask.mean())
    st.metric("Predicted foreground area",f"{area*100:.2f}%")
    st.metric("Selected threshold",f"{threshold:.2f}")
    st.caption("Foreground area is the fraction of pixels above the selected threshold; it is not a validated tumor volume.")
    st.metric("Original resolution",f"{ow} × {oh}")
    st.metric("Model input resolution",f"{size} × {size}")

st.subheader("Visual inspection")
n=3 if attr is not None else 2
fig,axes=plt.subplots(1,n,figsize=(5*n,4.8)); fig.patch.set_facecolor("#0b1020"); axes=np.atleast_1d(axes)
axes[0].imshow(display); axes[0].set_title("Resized model input",color="white"); axes[0].axis("off")
axes[1].imshow(display)
if mask.any():
    axes[1].imshow(np.ma.masked_where(~mask,mask),cmap="autumn",alpha=.48)
    axes[1].contour(mask.astype(np.uint8),levels=[.5],colors="red",linewidths=1.2)
axes[1].set_title(f"Predicted mask · threshold {threshold:.2f}",color="white"); axes[1].axis("off")
if attr is not None:
    axes[2].imshow(display); axes[2].imshow(attr,cmap="jet",alpha=.42,vmin=0,vmax=1)
    axes[2].set_title("Model attribution output",color="white"); axes[2].axis("off")
fig.tight_layout(); st.pyplot(fig,use_container_width=True); png=fig_png(fig); plt.close(fig)

report={
 "application":"TinyBrainNet / UNI-NET research dashboard",
 "prediction_type":"research_model_output_not_clinical_diagnosis",
 "model_file":MODEL_PATH.name,
 "model_hash_prefix":hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:16],
 "input_hash_prefix":image_hash,"original_size":[int(ow),int(oh)],"model_input_size":size,
 "classes":classes,"logits":[float(v) for v in logits],"temperature":temperature,
 "probabilities":[float(v) for v in probs],"predicted_class":pred,
 "top_class_probability":conf,"top2_margin":float(margin),
 "normalized_entropy":float(norm_entropy),"segmentation_threshold":float(threshold),
 "foreground_area_fraction":area,"foreground_area_percent":area*100,
 "warning":"Research/educational output only; not for clinical decisions."
}
st.subheader("Export")
j1,j2=st.columns(2)
j1.download_button("Download prediction report (JSON)",json.dumps(report,indent=2,ensure_ascii=False),
                   file_name="tinybrainnet_prediction_report.json",mime="application/json",use_container_width=True)
j2.download_button("Download visualization (PNG)",png,file_name="tinybrainnet_visualization.png",
                   mime="image/png",use_container_width=True)
with st.expander("ONNX technical details"):
    st.write("Input:",[(v.name,v.shape,v.type) for v in ins])
    st.write("Outputs:",[(v.name,v.shape,v.type) for v in outs])
    st.write("Configured output mapping:",{"classification":cls_name,"segmentation":seg_name,"attribution":attr_name})
    st.json(meta)
st.caption("Review hosting, logging, backup, and access controls before using any sensitive data.")
