"""WBC subtype classifier: classifies a WBC crop into one of four subtypes.

Uses a YOLOv8 classification model trained on dataset2-master (Blood Cell Images).
Falls back to 'WBC (unclassified)' if no model is available.

Classes: eosinophil, lymphocyte, monocyte, neutrophil
"""
from pathlib import Path
import numpy as np
import cv2

# Default path for the trained WBC subtype classifier
DEFAULT_CLASSIFIER_PATH = 'runs/classify/runs/classify/wbc_subtype_cls/weights/best.pt'

# Human-readable subtype labels
SUBTYPE_LABELS = {
    'eosinophil': 'Eosinophil',
    'lymphocyte': 'Lymphocyte',
    'monocyte': 'Monocyte',
    'neutrophil': 'Neutrophil',
}


class WBCSubtypeClassifier:
    """Classify a detected WBC crop into one of four subtypes."""

    def __init__(self, model_path: str = None):
        self._model = None
        self._names = None
        path = model_path or DEFAULT_CLASSIFIER_PATH
        if Path(path).exists():
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(path))
                self._names = self._model.names   # {0: 'eosinophil', ...}
            except Exception as e:
                print(f'[WBCSubtypeClassifier] Failed to load model: {e}')

    @property
    def available(self) -> bool:
        return self._model is not None

    def classify(self, bgr_crop: np.ndarray) -> dict:
        """Return subtype prediction dict for one BGR crop.

        Returns:
            {
              'subtype': str,          # e.g. 'Lymphocyte'
              'subtype_key': str,      # e.g. 'lymphocyte'
              'subtype_conf': float,   # confidence 0-1
            }
        """
        if not self.available or bgr_crop is None or bgr_crop.size == 0:
            return {'subtype': 'WBC (unclassified)', 'subtype_key': 'wbc', 'subtype_conf': 0.0}

        try:
            # Resize to 224 (model input size)
            resized = cv2.resize(bgr_crop, (224, 224))
            results = self._model(resized, verbose=False)
            top1_idx = int(results[0].probs.top1)
            top1_conf = float(results[0].probs.top1conf)
            key = self._names[top1_idx].lower()
            label = SUBTYPE_LABELS.get(key, key.capitalize())
            return {
                'subtype': label,
                'subtype_key': key,
                'subtype_conf': round(top1_conf, 4),
            }
        except Exception as e:
            return {'subtype': 'WBC (unclassified)', 'subtype_key': 'wbc', 'subtype_conf': 0.0}


# Module-level singleton — instantiated lazily
_classifier_instance: WBCSubtypeClassifier = None


def get_classifier(model_path: str = None) -> WBCSubtypeClassifier:
    global _classifier_instance
    if _classifier_instance is None or (model_path and not _classifier_instance.available):
        _classifier_instance = WBCSubtypeClassifier(model_path)
    return _classifier_instance
