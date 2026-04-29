import streamlit as st
from pathlib import Path
import tempfile
import sys
import numpy as np

# ensure repo root is on sys.path so `src` package is importable when Streamlit runs
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.pipeline import analyze_image, save_report
from tools.generate_report import generate

st.set_page_config(page_title="Blood Cell Analyzer", layout='wide')

st.title('Automated Blood Cell Classification and Disorder Detection')

uploaded = st.file_uploader('Upload a blood smear image', type=['jpg', 'jpeg', 'png', 'tif', 'tiff'])
model_path = st.text_input('Model weights path', value='runs/detect/test_yolo_dataset_stage12/weights/best.pt')

# Auto-threshold feature with manual override option
col1, col2 = st.columns([1, 2])
with col1:
    auto_threshold = st.checkbox('Auto-detect best threshold', value=False,
                                 help='Analyze confidence distribution and automatically select optimal threshold')
with col2:
    if auto_threshold:
        st.info('Auto-threshold ON: Will scan at 0.001 conf and compute optimal threshold.')
        manual_conf = st.slider('Confidence threshold (manual)', 0.0, 1.0, 0.25, disabled=True)
    else:
        manual_conf = st.slider('Confidence threshold', 0.0, 1.0, 0.25)

# Option to include all contour features per detection
include_all = st.checkbox('Include all contours per detection (else only largest)', value=False)

def compute_optimal_threshold(image_path, model_path, include_all_contours):
    """Run at very low confidence and compute optimal threshold from distribution."""
    result = analyze_image(image_path, model_path=model_path, conf=0.001,
                          include_all_contours=include_all_contours)
    boxes = result['summary'].get('boxes', [])

    if not boxes:
        return 0.25, None, "No detections found even at low threshold. Using default 0.25."

    confidences = sorted([b.get('conf', 0) for b in boxes])

    mean_conf = float(np.mean(confidences))
    std_conf = float(np.std(confidences))
    median_conf = float(np.median(confidences))
    min_conf = float(np.min(confidences))
    max_conf = float(np.max(confidences))

    # Method 1: Mean minus half standard deviation
    method1 = mean_conf - 0.5 * std_conf

    # Method 2: Largest gap in confidence scores
    method2 = method1
    if len(confidences) > 1:
        gaps = [confidences[i+1] - confidences[i] for i in range(len(confidences)-1)]
        max_gap_idx = int(np.argmax(gaps))
        gap_threshold = confidences[max_gap_idx] + 0.3 * gaps[max_gap_idx]
        method2 = gap_threshold

    # Final: use the higher (more conservative) of the two, bounded
    suggested = max(0.15, min(0.50, max(method1, method2)))
    suggested = round(suggested, 3)

    stats = {
        'count': len(boxes),
        'mean': round(mean_conf, 4),
        'std': round(std_conf, 4),
        'median': round(median_conf, 4),
        'min': round(min_conf, 4),
        'max': round(max_conf, 4),
        'method1': round(method1, 4),
        'method2': round(method2, 4)
    }

    explanation = (
        f"**{stats['count']} candidates** detected at low confidence.\n\n"
        f"| Statistic | Value |\n"
        f"|-----------|-------|\n"
        f"| Mean | {stats['mean']} |\n"
        f"| Std Dev | {stats['std']} |\n"
        f"| Median | {stats['median']} |\n"
        f"| Min | {stats['min']} |\n"
        f"| Max | {stats['max']} |\n\n"
        f"Method 1 (\u03bc - 0.5\u03c3): **{stats['method1']}**  \n"
        f"Method 2 (largest gap): **{stats['method2']}**  \n\n"
        f"### \u2705 Optimal threshold: **{suggested}**"
    )

    return suggested, stats, explanation

if uploaded is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded.name.split('.')[-1])
    tfile.write(uploaded.getbuffer())
    tfile.flush()
    st.image(tfile.name, caption='Input image', use_container_width=True)

    if st.button('Run analysis'):
        final_conf = manual_conf

        # Auto-threshold mode
        if auto_threshold:
            with st.spinner('Analyzing confidence distribution...'):
                suggested_conf, stats, explanation = compute_optimal_threshold(
                    tfile.name, model_path, include_all
                )
                final_conf = suggested_conf

                with st.expander('Threshold Analysis Details', expanded=True):
                    st.markdown(explanation)

                    if stats:
                        # Show metrics
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Detections", stats['count'])
                        m2.metric("Mean Conf", f"{stats['mean']:.3f}")
                        m3.metric("Median Conf", f"{stats['median']:.3f}")
                        m4.metric("Suggested", f"{suggested_conf}")

            st.info(f'Running final analysis at auto-detected threshold: **{final_conf}**')

        # Run final analysis
        with st.spinner('Analyzing...'):
            result = analyze_image(tfile.name, model_path=model_path, conf=final_conf,
                                  include_all_contours=include_all)
            out_dir = Path('results')
            out_dir.mkdir(exist_ok=True)
            save_report(result, str(out_dir), Path(uploaded.name).stem)
            st.image(result['annotated'], caption=f'Annotated result (threshold={final_conf})', use_container_width=True)

            # Show detection table
            try:
                boxes = result['summary'].get('boxes', [])
                if len(boxes) > 0:
                    rows = []
                    for i, b in enumerate(boxes):
                        rows.append({
                            'box_index': i,
                            'class_id': b.get('class'),
                            'class_name': b.get('class_name', ''),
                            'conf': round(float(b.get('conf', 0)), 4)
                        })
                    st.table(rows)
                else:
                    st.warning(f'No detections at threshold {final_conf}')
            except Exception as e:
                st.write('Could not render detection table: ' + str(e))

            st.json(result['summary'])

            # PDF generation
            pdf_out = out_dir / f"{Path(uploaded.name).stem}_report.pdf"
            generate(out_dir / f"{Path(uploaded.name).stem}_annotated.jpg",
                    out_dir / f"{Path(uploaded.name).stem}_summary.json", pdf_out)
            with open(pdf_out, 'rb') as f:
                st.download_button('Download PDF report', data=f, file_name=pdf_out.name,
                                  mime='application/pdf')
