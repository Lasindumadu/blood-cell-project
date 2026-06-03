# """High-level pipeline: preprocess -> detect -> segment -> extract features -> disorder detection

# This uses existing helpers in `src/preprocessing.py`, `src/segmentation.py`, and `src/features.py`.
# """
# from pathlib import Path
# import json
# import numpy as np
# import cv2
# from ultralytics import YOLO
# from PIL import Image

# from src.preprocessing import preprocess_image
# from src.segmentation import segment_cells
# from src.features import extract_features
# from src.disorder import detect_all_disorders


# def crop_bbox(img_np, xyxy, pad=4):
#     x1, y1, x2, y2 = [int(v) for v in xyxy]
#     h, w = img_np.shape[:2]
#     x1 = max(0, x1 - pad)
#     y1 = max(0, y1 - pad)
#     x2 = min(w - 1, x2 + pad)
#     y2 = min(h - 1, y2 + pad)
#     return img_np[y1:y2, x1:x2]


# def analyze_image(image_path: str, model_path: str = None, conf: float = 0.25, include_all_contours: bool = False):
#     """Run the full analysis on one image and return a dictionary report.

#     Report keys: annotated (PIL Image), summary (dict) with counts, per-box features, disorders list.
#     """
#     # Read and preprocess
#     img_pre = preprocess_image(image_path)
#     img_np = np.array(Image.fromarray(img_pre)) if not isinstance(img_pre, np.ndarray) else img_pre

#     # Load model
#     model = None
#     if model_path is None or not Path(model_path).exists():
#         # fallback to default ultralytics weight (will download if needed)
#         model = YOLO('yolov8n.pt')
#     else:
#         model = YOLO(str(model_path))

#     # Ultralytics expects images in RGB order. Our preprocessing uses OpenCV (BGR).
#     # Convert to RGB to avoid color-channel issues that can cause missing detections.
#     try:
#         img_for_model = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
#     except Exception:
#         img_for_model = img_np

#     results = model(img_for_model, imgsz=640, conf=conf)

#     # Try to obtain class name mapping for human-readable reports
#     names = None
#     try:
#         names = results[0].names
#     except Exception:
#         try:
#             names = getattr(model, 'names', None)
#         except Exception:
#             names = None

#     # Prepare outputs
#     r = results[0]
#     boxes_out = []

#     # annotated image
#     annotated_np = r.plot()  # RGB np
#     annotated_img = Image.fromarray(annotated_np)

#     # For each detected box, crop, segment, extract features
#     if hasattr(r, 'boxes') and r.boxes is not None:
#         for b in r.boxes:
#             # xyxy, conf, cls
#             try:
#                 xyxy = b.xyxy[0].tolist()
#             except Exception:
#                 xyxy = b.xyxy.tolist()
#             try:
#                 conf_v = float(b.conf[0])
#             except Exception:
#                 conf_v = float(b.conf)
#             try:
#                 cls_v = int(b.cls[0])
#             except Exception:
#                 cls_v = int(b.cls)

#             crop = crop_bbox(img_np, xyxy)

#             # segmentation on crop
#             try:
#                 mask = segment_cells(crop)
#             except Exception:
#                 mask = None

#             # pass rgb crop to extract nucleus/cytoplasm metrics when available
#             try:
#                 features = extract_features(mask, rgb_crop=crop) if mask is not None else []
#             except Exception:
#                 features = extract_features(mask) if mask is not None else []

#             # Many segmentation algorithms return several contours per crop; pick the largest
#             # contour's features to match the expected one-feature-per-box output unless
#             # the caller requested all contours.
#             if isinstance(features, list) and len(features) > 1 and not include_all_contours:
#                 try:
#                     largest = max(features, key=lambda p: p.get('area', 0))
#                     features = [largest]
#                 except Exception:
#                     # fallback: leave as-is
#                     pass

#             entry = {
#                 'xyxy': xyxy,
#                 'conf': conf_v,
#                 'class': cls_v,
#                 'features': features
#             }
#             # add human-readable class name when available
#             try:
#                 if names is not None:
#                     entry['class_name'] = str(names.get(cls_v, str(cls_v)))
#             except Exception:
#                 entry['class_name'] = str(cls_v)

#             boxes_out.append(entry)

#     # counts
#     counts = {}
#     for b in boxes_out:
#         c = str(b['class'])
#         counts[c] = counts.get(c, 0) + 1

#     report = {
#         'counts': counts,
#         'boxes': boxes_out,
#     }

#     # Run rule-based disorder detection module
#     disorders = detect_all_disorders(report)
#     report['disorders'] = disorders

#     return {
#         'annotated': annotated_img,
#         'summary': report
#     }


# def save_report(report: dict, out_dir: str, image_stem: str):
#     out_dir_p = Path(out_dir)
#     out_dir_p.mkdir(parents=True, exist_ok=True)

#     annotated: Image.Image = report['annotated']
#     annotated.save(out_dir_p / f"{image_stem}_annotated.jpg")

#     def _fix(obj):
#         if isinstance(obj, np.integer): return int(obj)
#         if isinstance(obj, np.floating): return float(obj)
#         if isinstance(obj, np.ndarray): return obj.tolist()
#         raise TypeError(type(obj))

#     with open(out_dir_p / f"{image_stem}_summary.json", 'w') as f:
#         json.dump(report['summary'], f, indent=2, default=_fix)


# if __name__ == '__main__':
#     # small local test: not executed during imports
#     pass

"""
Master pipeline: connects all 5 steps.
"""
import cv2
import json
import os
import numpy as np
from pathlib import Path

from src.preprocessing import preprocess_array
from src.segmentation import segment_cells
from src.features import extract_features
from src.classifier import run_yolo, annotate_image, CLASS_NAMES
from src.disorder_detection import detect_disorders


def analyze_image(image_path: str, model_path: str,
                  conf: float = 0.25, include_all_contours: bool = False) -> dict:
    """
    Full 5-step pipeline.
    Returns dict with: 'annotated' (np.ndarray), 'summary' (dict)
    """
    # Step 1: Preprocessing
    img_orig = cv2.imread(image_path)
    img_preprocessed = preprocess_array(img_orig)

    # Step 2: Segmentation
    _, contours, _ = segment_cells(img_preprocessed)

    # Step 3: Feature extraction per contour
    contour_features = []
    for c in contours:
        feats = extract_features(img_preprocessed, c)
        contour_features.append(feats)

    # Step 4: YOLOv8 classification
    detections = run_yolo(image_path, model_path=model_path, conf=conf)

    # Merge detections with nearest contour features
    cell_results = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
        best_feat = {}
        best_dist = float('inf')
        for c, f in zip(contours, contour_features):
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            mcx = M['m10'] / M['m00']; mcy = M['m01'] / M['m00']
            d = ((cx - mcx) ** 2 + (cy - mcy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_feat = f
        cell_results.append({
            'class_name': det['class_name'],
            'conf': det['conf'],
            'bbox': det['bbox'],
            'features': best_feat
        })

    # Step 5: Disorder detection
    disorders = detect_disorders(cell_results)

    # Annotate image
    annotated = annotate_image(img_preprocessed, detections)

    # Build summary
    from collections import Counter
    counts = Counter(d['class_name'] for d in detections)
    summary = {
        'image': image_path,
        'total_cells': len(detections),
        'cell_counts': dict(counts),
        'disorders': disorders,
        'boxes': [{'class': d['class'], 'class_name': d['class_name'],
                   'conf': d['conf'], 'bbox': d['bbox']} for d in detections]
    }

    return {'annotated': annotated, 'summary': summary}


def save_report(result: dict, out_dir: str, stem: str):
    """Save annotated image and JSON summary."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img_path = os.path.join(out_dir, f"{stem}_annotated.jpg")
    json_path = os.path.join(out_dir, f"{stem}_summary.json")
    cv2.imwrite(img_path, result['annotated'])
    with open(json_path, 'w') as f:
        json.dump(result['summary'], f, indent=2)
