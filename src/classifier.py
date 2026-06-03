"""
Step 4.4 — YOLOv8 Classification Wrapper
6 classes: RBC, neutrophil, lymphocyte, monocyte, eosinophil, platelet
"""
from ultralytics import YOLO
import numpy as np
import cv2

CLASS_NAMES = {
    0: 'RBC',
    1: 'neutrophil',
    2: 'lymphocyte',
    3: 'monocyte',
    4: 'eosinophil',
    5: 'platelet'
}

CLASS_COLORS = {
    'RBC':        (0, 0, 255),
    'neutrophil': (0, 255, 0),
    'lymphocyte': (255, 0, 0),
    'monocyte':   (0, 255, 255),
    'eosinophil': (255, 0, 255),
    'platelet':   (255, 165, 0),
}

def run_yolo(image_path: str, model_path: str, conf: float = 0.25):
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            detections.append({
                'class':      cls_id,
                'class_name': CLASS_NAMES.get(cls_id, 'unknown'),
                'conf':       float(box.conf.item()),
                'bbox':       box.xyxy[0].tolist()
            })
    return detections

def annotate_image(img: np.ndarray, detections: list) -> np.ndarray:
    annotated = img.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        cls_name = det['class_name']
        conf     = det['conf']
        color    = CLASS_COLORS.get(cls_name, (200, 200, 200))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return annotated