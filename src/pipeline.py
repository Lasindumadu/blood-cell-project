"""High-level pipeline: preprocess -> detect -> segment -> extract features -> disorder detection.

Two-stage classification:
  Stage 1: YOLOv8 detects WBC (group), RBC, Platelet
  Stage 2: WBC subtype classifier identifies Eosinophil / Lymphocyte / Monocyte / Neutrophil
"""
from pathlib import Path
import json
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

from src.preprocessing import preprocess_image
from src.segmentation import segment_cells
from src.features import extract_features
from src.disorder import detect_all_disorders
from src.wbc_classifier import get_classifier


def crop_bbox(img_np, xyxy, pad=4):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    h, w = img_np.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    return img_np[y1:y2, x1:x2]


def analyze_image(
    image_path: str,
    model_path: str = None,
    conf: float = 0.25,
    include_all_contours: bool = False,
    wbc_classifier_path: str = None,
):
    """Run the full two-stage analysis on one image and return a report dict.

    Report keys:
      annotated    PIL Image with bounding boxes
      summary      dict: counts, boxes (with features), disorder flags
    """
    # --- Stage 1: Preprocess + YOLO detection ---
    img_pre = preprocess_image(image_path)
    img_np = img_pre if isinstance(img_pre, np.ndarray) else np.array(img_pre)

    if model_path is None or not Path(model_path).exists():
        model = YOLO('yolov8n.pt')
    else:
        model = YOLO(str(model_path))

    try:
        img_for_model = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    except Exception:
        img_for_model = img_np

    results = model(img_for_model, imgsz=640, conf=conf)

    names = None
    try:
        names = results[0].names
    except Exception:
        names = getattr(model, 'names', None)

    r = results[0]
    annotated_img = Image.fromarray(r.plot())

    # --- Stage 2: WBC subtype classification ---
    wbc_cls = get_classifier(wbc_classifier_path)
    # Determine WBC class id from model names
    wbc_class_id = None
    if names:
        for cid, cname in names.items():
            if str(cname).upper() in ('WBC', 'WHITE BLOOD CELL'):
                wbc_class_id = cid
                break

    boxes_out = []
    if hasattr(r, 'boxes') and r.boxes is not None:
        for b in r.boxes:
            try:
                xyxy = b.xyxy[0].tolist()
            except Exception:
                xyxy = b.xyxy.tolist()
            try:
                conf_v = float(b.conf[0])
            except Exception:
                conf_v = float(b.conf)
            try:
                cls_v = int(b.cls[0])
            except Exception:
                cls_v = int(b.cls)

            crop = crop_bbox(img_np, xyxy)

            # Segmentation + feature extraction
            try:
                mask = segment_cells(crop)
            except Exception:
                mask = None

            try:
                features = extract_features(mask, rgb_crop=crop) if mask is not None else []
            except Exception:
                features = extract_features(mask) if mask is not None else []

            if isinstance(features, list) and len(features) > 1 and not include_all_contours:
                try:
                    features = [max(features, key=lambda p: p.get('area', 0))]
                except Exception:
                    pass

            # Determine class name; if WBC, run subtype classifier
            class_name = str(names.get(cls_v, str(cls_v))) if names else str(cls_v)
            subtype_info = {}

            if cls_v == wbc_class_id and wbc_cls.available:
                subtype_info = wbc_cls.classify(crop)
                # Override displayed class name with the specific subtype
                effective_name = subtype_info.get('subtype', class_name)
            else:
                effective_name = class_name

            entry = {
                'xyxy': xyxy,
                'conf': conf_v,
                'class': cls_v,
                'class_name': class_name,          # base YOLO class (WBC/RBC/Platelet)
                'effective_name': effective_name,  # refined name (subtype if WBC)
                'features': features,
            }
            if subtype_info:
                entry.update(subtype_info)

            boxes_out.append(entry)

    # --- Counts ---
    # Primary counts by YOLO class
    counts = {}
    for b in boxes_out:
        c = str(b['class'])
        counts[c] = counts.get(c, 0) + 1

    # Subtype counts (for WBCs where classifier ran)
    subtype_counts = {}
    for b in boxes_out:
        key = b.get('subtype_key')
        if key and key != 'wbc':
            subtype_counts[key] = subtype_counts.get(key, 0) + 1

    report = {
        'counts': counts,
        'subtype_counts': subtype_counts,
        'boxes': boxes_out,
        'wbc_subtype_available': wbc_cls.available,
    }

    report['disorders'] = detect_all_disorders(report)

    return {
        'annotated': annotated_img,
        'summary': report,
    }


def save_report(report: dict, out_dir: str, image_stem: str):
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    annotated: Image.Image = report['annotated']
    annotated.save(out_dir_p / f"{image_stem}_annotated.jpg")

    # Remove non-serialisable items before saving JSON
    summary = report['summary']
    with open(out_dir_p / f"{image_stem}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
