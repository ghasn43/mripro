# app.py (cv2-free)
# Streamlit demo: Low-field degradation -> enhancement (UNetTiny) -> uncertainty map (MC Dropout)
# Added: Difference maps (Δ enhanced - low-field) + uncertainty histogram/stats

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# Local modules
from src.toy_data import make_toy_clean_images
from src.toy_pretrain import make_checkpoint
from src.degrade import degrade_low_field
from src.model import UNetTiny
from src.infer import mc_dropout_predict


# =============================
# Helpers (NO cv2)
# =============================
def read_gray01_resize(path: Path, size: int = 128) -> np.ndarray:
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return (np.asarray(im).astype(np.float32) / 255.0).clip(0, 1)


def to_u8(gray01: np.ndarray) -> np.ndarray:
    return (np.clip(gray01, 0, 1) * 255).astype(np.uint8)


def upscale_for_display(img01: np.ndarray, size: int = 256) -> np.ndarray:
    im = Image.fromarray(to_u8(img01))
    im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im)


def upscale_rgb_for_display(rgb_u8: np.ndarray, size: int = 256) -> np.ndarray:
    im = Image.fromarray(rgb_u8.astype(np.uint8), mode="RGB")
    im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im)


def colorize_uncertainty(u01: np.ndarray) -> np.ndarray:
    rgba = cm.inferno(np.clip(u01, 0, 1))
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay_heatmap(gray01: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    g = np.repeat(to_u8(gray01)[..., None], 3, axis=-1)
    out = (1 - alpha) * g + alpha * heat_rgb
    return np.clip(out, 0, 255).astype(np.uint8)


def list_clean_images(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    return [
        p.name
        for p in sorted(folder.iterdir())
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ]


def colorize_signed_delta(delta: np.ndarray, vmax: float) -> np.ndarray:
    """
    Colorize signed delta in [-vmax, vmax] using a diverging colormap.
    Output: RGB uint8
    """
    if vmax <= 1e-8:
        vmax = 1e-8
    d = np.clip(delta / vmax, -1, 1)  # [-1, 1]
    # Map [-1,1] -> [0,1]
    d01 = (d + 1) / 2.0
    rgba = cm.coolwarm(d01)
    return (rgba[..., :3] * 255).astype(np.uint8)


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# =============================
# App
# =============================
st.set_page_config(page_title="MRI Safe Enhance Demo", layout="wide")
st.title("🧲 Low-Field MRI Enhancement + Uncertainty")
st.caption("Research demo (non-clinical). Enhances low-field MRI-like images and visualizes model uncertainty.")


# =============================
# Session State defaults + NAV SYNC (MUST BE BEFORE WIDGETS)
# =============================
if "radiologist_mode" not in st.session_state:
    st.session_state["radiologist_mode"] = False
if "workflow_step" not in st.session_state:
    st.session_state["workflow_step"] = 1  # owned by the radio widget
# nav_step is set by top buttons, then copied into workflow_step here (before radio exists)
if "nav_step" in st.session_state and st.session_state["nav_step"] in (1, 2, 3, 4):
    st.session_state["workflow_step"] = int(st.session_state["nav_step"])
    del st.session_state["nav_step"]


# =============================
# IP / Ownership Banner (FIRST PAGE)
# =============================
# --- EDIT ME: put your real contact details here ---
OWNER_NAME = "Ghassan Muammar"
OWNER_COMPANY = "Experts Group FZE"
OWNER_EMAIL = "info@expertsgroup.me"         # <-- change
OWNER_PHONE = "+971-50-6690381"       # <-- change
OWNER_WEBSITE = "www.expertsgroup.me"                     # optional, e.g. "https://expertsgroup.ae"

st.markdown(
    f"""
<div style="
    border: 1px solid rgba(255,255,255,0.15);
    padding: 12px 14px;
    border-radius: 12px;
    background: rgba(255,255,255,0.04);
    margin-top: 10px;
    margin-bottom: 10px;
">
  <div style="font-size: 0.95rem; font-weight: 700;">
    © Intellectual Property Notice
  </div>
  <div style="font-size: 0.9rem; line-height: 1.5; margin-top: 6px;">
    This application, its code, workflows, UI, and associated materials are the intellectual property of
    <b>{OWNER_COMPANY}</b> — <b>{OWNER_NAME}</b>.
    <br/>
    Contact: <b>{OWNER_EMAIL}</b> | <b>{OWNER_PHONE}</b>
    {"<br/>Website: <b>" + OWNER_WEBSITE + "</b>" if OWNER_WEBSITE else ""}
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# =============================
# Sidebar controls (radio owns workflow_step)
# =============================
with st.sidebar:
    st.header("Controls")

    radiologist_mode = st.toggle(
        "🩺 Radiologist Mode (technical language)",
        value=st.session_state["radiologist_mode"],
        key="radiologist_mode",
        help="Toggles more technical explanations. Does not change processing.",
    )

    st.subheader("Workflow")
    step_labels = {1: "1) Input", 2: "2) Degrade", 3: "3) Enhance", 4: "4) Review"}
    st.radio(
        "Go to step",
        options=[1, 2, 3, 4],
        format_func=lambda x: step_labels[x],
        key="workflow_step",
    )

    st.divider()
    st.subheader("Parameters")
    severity = st.slider("Low-field degradation", 0.0, 1.0, 0.65, 0.01)
    mc_samples = st.slider("MC Dropout samples", 4, 64, 24, 2)
    heat_alpha = st.slider("Uncertainty overlay", 0.0, 1.0, 0.45, 0.01)
    u_gain = st.slider("Uncertainty gain", 0.25, 5.0, 2.0, 0.05)
    display_size = st.select_slider("Display size", [128, 256, 384, 512], 256)

    st.divider()
    st.subheader("Difference Maps (Δ)")
    delta_gain = st.slider("Δ visualization gain", 0.25, 8.0, 2.0, 0.05)
    delta_clip = st.slider("Δ clip (fraction of full scale)", 0.01, 0.50, 0.10, 0.01)
    st.caption("Δ maps are for transparency: show where enhancement changes the low-field image.")

    st.divider()
    st.subheader("Uncertainty Stats")
    u_thresh = st.slider("High-uncertainty threshold", 0.05, 0.95, 0.60, 0.01)
    st.caption("Used to compute % of pixels above threshold.")


    st.divider()
    source_mode = st.radio(
        "Input source",
        ["Toy clean images", "From data/clean", "Upload image"],
        index=1,
    )


# =============================
# Active workflow top bar (clickable) - sets nav_step NOT workflow_step
# =============================
def step_badge(n: int, title: str, active: bool) -> str:
    bg = "rgba(255,255,255,0.12)" if active else "rgba(255,255,255,0.05)"
    bd = "rgba(255,255,255,0.18)" if active else "rgba(255,255,255,0.10)"
    fw = "800" if active else "600"
    return f"""
<div style="padding:8px 12px;border-radius:999px;background:{bg};border:1px solid {bd};font-weight:{fw};">
  {n}) {title}
</div>
"""


b1, b2, b3, b4 = st.columns(4)
with b1:
    if st.button("1) Input", use_container_width=True):
        st.session_state["nav_step"] = 1
        st.rerun()
with b2:
    if st.button("2) Degrade", use_container_width=True):
        st.session_state["nav_step"] = 2
        st.rerun()
with b3:
    if st.button("3) Enhance", use_container_width=True):
        st.session_state["nav_step"] = 3
        st.rerun()
with b4:
    if st.button("4) Review", use_container_width=True):
        st.session_state["nav_step"] = 4
        st.rerun()

active_step = int(st.session_state["workflow_step"])
st.markdown(
    f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 14px 0;">
  {step_badge(1, "Input", active_step == 1)}
  {step_badge(2, "Degrade", active_step == 2)}
  {step_badge(3, "Enhance", active_step == 3)}
  {step_badge(4, "Review", active_step == 4)}
</div>
""",
    unsafe_allow_html=True,
)


# =============================
# Disclaimers / safety copy
# =============================
with st.expander("⚠️ Medical Context & Disclaimer", expanded=False):
    st.markdown(
        f"""
This application is a **research and decision-support demonstration**.

It does **not** provide medical diagnoses and does **not** replace a radiologist or physician.

**Purpose of the images shown:**
- Improve visual interpretability of **low-field MRI** data
- Highlight **areas of uncertainty** where caution is advised
- Support **remote expert review and triage** decisions

Final clinical decisions must always be made by **qualified medical professionals**.

**Ownership / IP:**  
This application, its code, workflows, UI, and associated materials are the intellectual property of
**{OWNER_COMPANY} — {OWNER_NAME}**.  
Contact: **{OWNER_EMAIL}** | **{OWNER_PHONE}**
"""
    )

with st.expander("🧭 Confidence hints (how to use uncertainty safely)", expanded=not radiologist_mode):
    if radiologist_mode:
        st.markdown(
            """
**Conservative usage guidance (non-diagnostic):**
- Treat uncertainty as an **attention signal**, not a probability of disease.
- High uncertainty can reflect **low SNR**, motion, partial-volume effects, coil/sequence differences, or out-of-distribution patterns.
- Use uncertainty to **prioritize cautious review**, compare with raw/low-field image, and correlate clinically.
- Avoid over-trusting enhancement in regions showing **persistently high uncertainty**.
"""
        )
    else:
        st.markdown(
            """
**Simple guidance (non-diagnostic):**
- The uncertainty colors show where the AI is **less sure**.
- Bright/hot colors → **be cautious** and double-check the low-field image.
- Dark/cool colors → the AI is **more consistent**, but it still does not guarantee correctness.
- If an area looks important **and** uncertainty is high, consider **expert review** or additional imaging.
"""
        )


# =============================
# Model (lazy load)
# =============================
ckpt_path = make_checkpoint(force=False)

@st.cache_resource
def load_model(ckpt: str):
    import torch

    model = UNetTiny(base=8, p_drop=0.15)
    payload = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


# =============================
# Input (always computed, but only shown per step)
# =============================
clean01 = None

if source_mode == "Toy clean images":
    toys = make_toy_clean_images(n=12, size=128, seed=7)
    idx = st.sidebar.slider("Toy index", 0, len(toys) - 1, 0)
    clean01 = toys[idx]

elif source_mode == "From data/clean":
    files = list_clean_images(Path("data/clean"))
    if files:
        pick = st.sidebar.selectbox("Choose image", files)
        clean01 = read_gray01_resize(Path("data/clean") / pick)
    else:
        st.warning("No images found in data/clean")

else:
    up = st.sidebar.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if up:
        im = Image.open(up).convert("L").resize((128, 128))
        clean01 = (np.asarray(im).astype(np.float32) / 255.0).clip(0, 1)

if clean01 is None:
    st.warning("Please select or upload an input image to continue.")
    st.stop()

low01 = degrade_low_field(clean01, severity)


# =============================
# Enhancement caching (run on demand)
# =============================
run_key = (float(severity), int(mc_samples))

if "last_run_key" not in st.session_state:
    st.session_state["last_run_key"] = None
if "enhanced01" not in st.session_state:
    st.session_state["enhanced01"] = None
if "unc01" not in st.session_state:
    st.session_state["unc01"] = None
if "dt" not in st.session_state:
    st.session_state["dt"] = None


def ensure_enhanced():
    """Compute enhanced + uncertainty if missing or params changed."""
    if (st.session_state["last_run_key"] != run_key) or (st.session_state["enhanced01"] is None):
        model = load_model(ckpt_path)
        with st.spinner("Running model..."):
            t0 = time.time()
            enhanced01, unc01 = mc_dropout_predict(model, low01, mc_samples)
            dt = time.time() - t0
        st.session_state["enhanced01"] = enhanced01
        st.session_state["unc01"] = unc01
        st.session_state["dt"] = dt
        st.session_state["last_run_key"] = run_key


# =============================
# Render active step
# =============================
step = int(st.session_state["workflow_step"])

if step == 1:
    st.subheader("1) Input")
    st.markdown("Select an input source in the sidebar. The current input is shown below.")
    st.image(upscale_for_display(clean01, display_size), channels="L", caption="Selected input (clean/reference)")

elif step == 2:
    st.subheader("2) Degrade (simulate low-field)")
    st.markdown("This step simulates typical low-field effects (noise/blur/contrast loss) at the chosen severity.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Clean (reference)**")
        st.image(upscale_for_display(clean01, display_size), channels="L")
    with c2:
        st.markdown("**Low-field (degraded)**")
        st.image(upscale_for_display(low01, display_size), channels="L")
    st.caption("Adjust severity in the sidebar to see degradation change.")

elif step == 3:
    st.subheader("3) Enhance (UNetTiny)")
    st.markdown("Runs conservative enhancement and estimates uncertainty using MC Dropout (on demand).")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Low-field input**")
        st.image(upscale_for_display(low01, display_size), channels="L")
    with c2:
        if st.button("▶ Run enhancement", type="primary", use_container_width=True):
            ensure_enhanced()

        if st.session_state["enhanced01"] is None:
            st.info("Click **Run enhancement** to compute the enhanced image and uncertainty.")
        else:
            st.markdown("**Enhanced output**")
            st.image(upscale_for_display(st.session_state["enhanced01"], display_size), channels="L")
            if st.session_state["dt"] is not None:
                st.caption(f"Inference time: {st.session_state['dt']:.2f}s")

    # Optional preview of Δ after enhancement
    if st.session_state["enhanced01"] is not None:
        enhanced01 = st.session_state["enhanced01"]
        delta = (enhanced01 - low01).astype(np.float32)
        abs_delta = np.abs(delta)

        # Determine visualization scale
        vmax = max(1e-6, float(delta_clip))
        # scale delta_gain by clip: effective vmax = clip/delta_gain => higher gain => more saturation
        eff_vmax = max(1e-6, vmax / max(1e-6, float(delta_gain)))

        delta_rgb = colorize_signed_delta(delta, vmax=eff_vmax)

        st.markdown("---")
        st.subheader("Δ Preview (Enhanced − Low-field)")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Signed Δ (adds vs removes)**")
            st.image(upscale_rgb_for_display(delta_rgb, display_size))
            st.caption("Red/blue indicates direction of change (visualization only).")
        with d2:
            st.markdown("**|Δ| (magnitude of change)**")
            st.image(upscale_for_display(np.clip(abs_delta * delta_gain, 0, 1), display_size), channels="L")
            st.caption("Brighter = larger change; used to verify conservative behavior.")

elif step == 4:
    st.subheader("4) Review uncertainty + transparency")
    if st.session_state["enhanced01"] is None or st.session_state["unc01"] is None:
        st.warning("Enhancement not computed yet. Go to **Step 3** and click **Run enhancement**.")
        st.stop()

    enhanced01 = st.session_state["enhanced01"]
    unc01 = st.session_state["unc01"]
    dt = safe_float(st.session_state["dt"], 0.0)

    # Uncertainty visuals
    unc_vis = np.clip(unc01 * u_gain, 0, 1)
    heat = colorize_uncertainty(unc_vis)
    overlay = overlay_heatmap(enhanced01, heat, heat_alpha)

    # Difference maps
    delta = (enhanced01 - low01).astype(np.float32)
    abs_delta = np.abs(delta)

    # Effective signed delta scale:
    # delta_clip is a fraction of full [0..1] range. delta_gain increases saturation.
    eff_vmax = max(1e-6, float(delta_clip) / max(1e-6, float(delta_gain)))
    delta_rgb = colorize_signed_delta(delta, vmax=eff_vmax)

    abs_delta_vis = np.clip(abs_delta * delta_gain, 0, 1)

    # Quick comparison
    st.subheader("🔍 Quick comparison (core panels)")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("**Low-field (degraded)**")
        st.image(upscale_for_display(low01, display_size), channels="L")
    with r2:
        st.markdown("**Enhanced**")
        st.image(upscale_for_display(enhanced01, display_size), channels="L")
    with r3:
        st.markdown("**Uncertainty (MC Dropout)**")
        st.image(upscale_rgb_for_display(heat, display_size))

    st.markdown("---")

    # Δ maps row
    st.subheader("🧾 Transparency: What changed? (Δ maps)")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**Signed Δ (Enhanced − Low-field)**")
        st.image(upscale_rgb_for_display(delta_rgb, display_size))
        st.caption("Direction of change (visualization only).")
    with d2:
        st.markdown("**|Δ| magnitude**")
        st.image(upscale_for_display(abs_delta_vis, display_size), channels="L")
        st.caption("Magnitude of change. Brighter = larger change.")
    with d3:
        # Simple scalar summaries of Δ
        mean_abs = float(np.mean(abs_delta))
        p95_abs = float(np.quantile(abs_delta, 0.95))
        max_abs = float(np.max(abs_delta))
        st.markdown("**Δ summary (numbers)**")
        st.write(f"Mean |Δ|: **{mean_abs:.4f}**")
        st.write(f"95th pct |Δ|: **{p95_abs:.4f}**")
        st.write(f"Max |Δ|: **{max_abs:.4f}**")
        st.caption("These values should remain small in conservative enhancement.")

    st.markdown("---")

    # Overlay
    st.subheader("🧪 Enhanced + Uncertainty (Overlay)")
    st.image(upscale_rgb_for_display(overlay, display_size))
    st.caption(f"Overlay alpha: {heat_alpha:.2f} | Inference time: {dt:.2f}s")

    # Uncertainty histogram + stats
    st.markdown("---")
    st.subheader("📊 Uncertainty profile (histogram + stats)")

    uvals = np.clip(unc_vis, 0, 1).ravel()
    mean_u = float(np.mean(uvals))
    med_u = float(np.median(uvals))
    p95_u = float(np.quantile(uvals, 0.95))
    frac_hi = float(np.mean(uvals >= float(u_thresh))) * 100.0

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Mean", f"{mean_u:.3f}")
    with s2:
        st.metric("Median", f"{med_u:.3f}")
    with s3:
        st.metric("95th pct", f"{p95_u:.3f}")
    with s4:
        st.metric(f"% ≥ {u_thresh:.2f}", f"{frac_hi:.1f}%")

    # Histogram plot (matplotlib)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(uvals, bins=30)
    ax.set_xlabel("Uncertainty (0..1)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Uncertainty distribution (after gain + clip)")
    st.pyplot(fig, clear_figure=True)

    with st.expander("🧠 How to interpret Δ + uncertainty (non-diagnostic)", expanded=radiologist_mode):
        if radiologist_mode:
            st.markdown(
                """
**Δ maps (Enhanced − Low-field):**
- Signed Δ highlights **directionality** of intensity adjustments.
- |Δ| highlights **magnitude** of change; large clusters can indicate aggressive enhancement.
- For conservative residual enhancement, you expect **small, localized** |Δ|.

**Uncertainty histogram:**
- A heavy tail near 1.0 suggests substantial regions of instability/out-of-distribution behavior.
- Use the **% ≥ threshold** as a quick scan-level “caution indicator.”
"""
            )
        else:
            st.markdown(
                """
**Δ maps (what changed):**
- Signed Δ shows whether the AI made areas brighter or darker (just a visualization).
- |Δ| shows how strong the change was. Big bright areas mean “AI changed a lot.”

**Uncertainty histogram:**
- If many pixels are high uncertainty, be more cautious and double-check the low-field image.
"""
            )

    st.info(
        "This system enhances low-field MRI images for better interpretability "
        "while explicitly visualizing uncertainty and showing transparency (Δ maps). "
        "It is designed as a companion to radiologists — not a replacement."
    )
