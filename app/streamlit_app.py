"""Streamlit web interface for the Automated Blood Cell Classification System."""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.pipeline import analyze_image, save_report
from src.wbc_classifier import DEFAULT_CLASSIFIER_PATH
from tools.generate_report import generate as generate_pdf, generate_csv

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Blood Cell Analyzer",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.metric-box {
    background: #f0f4ff;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
    border-left: 4px solid #1a3a6b;
}
.metric-label { font-size: 12px; color: #555; margin-bottom: 4px; }
.metric-value { font-size: 28px; font-weight: bold; color: #1a3a6b; }
.disorder-alert {
    background: #fff3f3;
    border-left: 4px solid #c62828;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 4px 0;
    color: #c62828;
    font-weight: 500;
}
.normal-badge {
    background: #e8f5e9;
    border-left: 4px solid #2e7d32;
    border-radius: 6px;
    padding: 10px 14px;
    color: #2e7d32;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    model_path = st.text_input(
        "Detection model path",
        value="runs/detect/test_yolo_dataset_stage12/weights/best.pt",
        help="Path to trained YOLOv8 detection .pt weights file"
    )
    wbc_cls_path = st.text_input(
        "WBC subtype classifier path",
        value=DEFAULT_CLASSIFIER_PATH,
        help="Path to YOLOv8 classification model for WBC subtypes (Eosinophil/Lymphocyte/Monocyte/Neutrophil)"
    )
    conf = st.slider("Confidence threshold", 0.01, 0.95, 0.25, 0.01,
                     help="Detection confidence cutoff")
    include_all = st.checkbox("All contours per detection",
                              help="Include all contour features, not just the largest one")

    st.divider()
    st.caption("**About**")
    st.caption(
        "Automated Blood Cell Classification and Hematological Disorder Detection System. "
        "Built with YOLOv8 + OpenCV + scikit-image."
    )
    st.caption("University of Ruhuna — EE7204/EC7205")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🩸 Automated Blood Cell Classifier")
st.markdown(
    "Upload a peripheral blood smear image to detect and classify blood cells "
    "(WBC, RBC, Platelets) and screen for hematological disorders."
)

uploaded = st.file_uploader(
    "Upload a blood smear image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
)

# ---------------------------------------------------------------------------
# Helper: auto-detect threshold from confidence distribution
# ---------------------------------------------------------------------------

def auto_threshold(image_path, model_path, include_all_contours):
    result = analyze_image(image_path, model_path=model_path, conf=0.001,
                           include_all_contours=include_all_contours)
    boxes = result["summary"].get("boxes", [])
    if not boxes:
        return 0.25, None

    confs = sorted(b.get("conf", 0) for b in boxes)
    mean_c = float(np.mean(confs))
    std_c = float(np.std(confs))
    m1 = mean_c - 0.5 * std_c
    m2 = m1
    if len(confs) > 1:
        gaps = [confs[i + 1] - confs[i] for i in range(len(confs) - 1)]
        idx = int(np.argmax(gaps))
        m2 = confs[idx] + 0.3 * gaps[idx]
    suggested = round(max(0.15, min(0.50, max(m1, m2))), 3)

    stats = {
        "count": len(boxes),
        "mean": round(mean_c, 4),
        "std": round(std_c, 4),
        "suggested": suggested,
    }
    return suggested, stats


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if uploaded is not None:
    # Save to temp file
    suffix = Path(uploaded.name).suffix
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded.getbuffer())
    tfile.flush()

    col_img, col_info = st.columns([1.2, 1])
    with col_img:
        st.image(tfile.name, caption="Uploaded image", use_container_width=True)

    with col_info:
        st.markdown("### Image Details")
        st.write(f"**File:** {uploaded.name}")
        st.write(f"**Size:** {uploaded.size / 1024:.1f} KB")
        st.write(f"**Format:** {suffix.upper().strip('.')}")

    if st.button("▶ Run Analysis", type="primary", use_container_width=True):
        cls_path = wbc_cls_path if Path(wbc_cls_path).exists() else None
        stage2_msg = "with WBC subtype classification" if cls_path else "without WBC subtype classifier (model not found)"
        with st.spinner(f"Running full pipeline: preprocessing → detection → segmentation → features → disorders… ({stage2_msg})"):
            result = analyze_image(
                tfile.name,
                model_path=model_path if Path(model_path).exists() else None,
                conf=conf,
                include_all_contours=include_all,
                wbc_classifier_path=cls_path,
            )

        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        stem = Path(uploaded.name).stem
        save_report(result, str(out_dir), stem)

        summary = result["summary"]
        counts = summary.get("counts", {})
        boxes = summary.get("boxes", [])
        disorders = summary.get("disorders", [])

        # Build class-name map from boxes
        class_names = {}
        for b in boxes:
            cid = b.get("class")
            cname = b.get("class_name", "")
            if cid is not None and cname:
                class_names[int(cid)] = cname

        def label(k):
            fallback = {"0": "WBC", "1": "RBC", "2": "Platelets"}
            try:
                k_int = int(k)
                if k_int in class_names:
                    return class_names[k_int]
            except Exception:
                pass
            return fallback.get(str(k), f"Class {k}")

        total = sum(int(v) for v in counts.values())

        st.divider()

        # --- Annotated image ---
        st.subheader("📷 Detection Result")
        st.image(result["annotated"], caption=f"Annotated ({len(boxes)} detections, conf≥{conf})",
                 use_container_width=True)

        # --- Count metrics ---
        st.subheader("🔢 Cell Counts")
        all_classes = [("0", "WBC"), ("1", "RBC"), ("2", "Platelets")]
        cols = st.columns(len(all_classes) + 1)
        for i, (cid, cname) in enumerate(all_classes):
            count = counts.get(cid, 0)
            pct = f"{count/total*100:.1f}%" if total > 0 else "—"
            cols[i].metric(cname, str(count), delta=f"{pct} of total")
        cols[-1].metric("Total Cells", str(total))

        # Pie chart of counts
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            if counts:
                labels = [label(k) for k in counts]
                vals = [int(v) for v in counts.values()]
                colors_list = ["#1a3a6b", "#e53935", "#f9a825"]
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.pie(vals, labels=labels, autopct="%1.0f%%",
                       colors=colors_list[:len(vals)], startangle=90)
                ax.set_title("Cell Type Distribution", fontsize=10)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        except Exception:
            pass

        # --- WBC subtype breakdown ---
        subtype_counts = summary.get('subtype_counts', {})
        wbc_subtypes_available = summary.get('wbc_subtype_available', False)

        if wbc_subtypes_available and subtype_counts:
            st.subheader("🧬 WBC Differential (Subtype Breakdown)")
            wbc_total = counts.get('0', 0)  # WBC class id
            sub_cols = st.columns(4)
            subtype_labels = {'eosinophil': 'Eosinophil', 'lymphocyte': 'Lymphocyte',
                              'monocyte': 'Monocyte', 'neutrophil': 'Neutrophil'}
            for i, (key, lbl) in enumerate(subtype_labels.items()):
                n = subtype_counts.get(key, 0)
                pct = f"{n/wbc_total*100:.0f}%" if wbc_total > 0 else "—"
                sub_cols[i].metric(lbl, str(n), delta=pct)
        elif wbc_subtypes_available:
            st.info("WBC subtype classifier loaded but no WBCs were detected.")
        else:
            st.caption(
                "WBC subtype classifier not available — set the classifier path in the sidebar "
                "and re-run to see Eosinophil/Lymphocyte/Monocyte/Neutrophil breakdown."
            )

        # --- Morphological summary ---
        st.subheader("🔬 Morphological Features")
        feat_rows = []
        for b in boxes:
            for f in b.get("features", []):
                if f.get("area", 0) > 0:
                    cell_type = b.get("effective_name") or label(b.get("class", "?"))
                    subtype_conf = b.get("subtype_conf", "")
                    feat_rows.append({
                        "Cell Type": cell_type,
                        "Subtype Conf": round(float(subtype_conf), 3) if subtype_conf else "—",
                        "Det Conf": round(float(b.get("conf", 0)), 3),
                        "Area (px)": round(f.get("area", 0), 1),
                        "Circularity": round(f.get("circularity", 0), 3),
                        "Eccentricity": round(f.get("eccentricity", 0), 3),
                        "N/C Ratio": round(f.get("nc_ratio", 0), 3),
                        "Hue Mean": round(f.get("hue_mean", 0), 1),
                        "GLCM Contrast": round(f.get("glcm_contrast", 0), 3),
                    })

        if feat_rows:
            import pandas as pd
            df = pd.DataFrame(feat_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No morphological features extracted. Try lowering the confidence threshold.")

        # --- Disorder detection ---
        st.subheader("🏥 Hematological Disorder Screening")
        if disorders:
            for d in disorders:
                st.markdown(f'<div class="disorder-alert">⚠️ {d}</div>', unsafe_allow_html=True)
            st.warning("**Clinical note:** Findings require confirmation by a qualified hematologist.")
        else:
            st.markdown('<div class="normal-badge">✅ No hematological disorders detected.</div>',
                        unsafe_allow_html=True)

        # --- Segmentation mask overlay (proposal §4.8) ---
        with st.expander("🔲 Segmentation Masks (Watershed)"):
            try:
                from src.preprocessing import preprocess_image
                from src.segmentation import segment_cells

                img_pre = preprocess_image(tfile.name)
                seg_strips = []
                for b in boxes[:6]:  # show first 6 cells
                    x1, y1, x2, y2 = [int(v) for v in b['xyxy']]
                    h, w = img_pre.shape[:2]
                    crop = img_pre[max(0,y1-4):min(h,y2+4), max(0,x1-4):min(w,x2+4)]
                    if crop.size == 0:
                        continue
                    mask = segment_cells(crop)
                    # Create RGB overlay: green mask over crop
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    overlay = crop_rgb.copy()
                    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
                    seg_strips.append(overlay)

                if seg_strips:
                    seg_cols = st.columns(len(seg_strips))
                    for i, (col, img) in enumerate(zip(seg_cols, seg_strips)):
                        lbl = boxes[i].get('effective_name') or boxes[i].get('class_name', '')
                        col.image(img, caption=lbl, use_container_width=True)
                else:
                    st.info("No cell crops to segment.")
            except Exception as e:
                st.warning(f"Segmentation preview failed: {e}")

        # --- JSON ---
        with st.expander("📄 Raw JSON Summary"):
            # lbp_hist is large — truncate for display
            display_summary = json.loads(json.dumps(summary))
            for b in display_summary.get("boxes", []):
                for f in b.get("features", []):
                    if "lbp_hist" in f:
                        f["lbp_hist"] = f"[{len(f['lbp_hist'])} values — omitted for display]"
            st.json(display_summary)

        # --- PDF + CSV download ---
        st.divider()
        pdf_out = out_dir / f"{stem}_report.pdf"
        csv_out = out_dir / f"{stem}_report.csv"
        dl_col1, dl_col2 = st.columns(2)
        try:
            generate_pdf(
                out_dir / f"{stem}_annotated.jpg",
                out_dir / f"{stem}_summary.json",
                pdf_out,
            )
            with open(pdf_out, "rb") as f:
                dl_col1.download_button(
                    "📥 Download PDF Report",
                    data=f.read(),
                    file_name=pdf_out.name,
                    mime="application/pdf",
                    use_container_width=True,
                )
        except Exception as e:
            dl_col1.error(f"PDF generation failed: {e}")
        try:
            generate_csv(out_dir / f"{stem}_summary.json", csv_out)
            with open(csv_out, "rb") as f:
                dl_col2.download_button(
                    "📥 Download CSV (LIS)",
                    data=f.read(),
                    file_name=csv_out.name,
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            dl_col2.error(f"CSV generation failed: {e}")

else:
    # Landing state
    st.info("👆 Upload a blood smear image (JPEG / PNG / TIFF) to begin analysis.")
    st.markdown("""
**Pipeline overview:**
1. **Preprocessing** — Gaussian filter → LAB color space → CLAHE enhancement
2. **Detection** — YOLOv8 detects WBC, RBC, and Platelets with bounding boxes
3. **WBC Subtype** — YOLOv8n-cls classifies each WBC as Eosinophil / Lymphocyte / Monocyte / Neutrophil
4. **Segmentation** — Otsu + Marker-Controlled Watershed on each cell crop (filters <100px noise, >5000px clumps)
5. **Feature Extraction** — 22 morphological, nuclear, color (HSV), and textural features per cell
6. **Disorder Screening** — ALL, leukocytosis, eosinophilia, microcytic anaemia, sickle cell, thrombocytopenia
7. **Report** — Annotated image + PDF clinical report with CBC + WBC differential summary
    """)
