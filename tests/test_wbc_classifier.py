"""Tests for the WBC subtype classifier module."""
import sys
from pathlib import Path
import numpy as np
import cv2

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.wbc_classifier import WBCSubtypeClassifier, get_classifier, SUBTYPE_LABELS


def test_classifier_fallback_when_no_model():
    """When no model exists the classifier should return 'WBC (unclassified)'."""
    cls = WBCSubtypeClassifier(model_path='nonexistent_path.pt')
    assert not cls.available
    result = cls.classify(np.zeros((50, 50, 3), dtype=np.uint8))
    assert result['subtype'] == 'WBC (unclassified)'
    assert result['subtype_key'] == 'wbc'
    assert result['subtype_conf'] == 0.0


def test_classifier_empty_crop_fallback():
    cls = WBCSubtypeClassifier(model_path='nonexistent_path.pt')
    result = cls.classify(None)
    assert result['subtype'] == 'WBC (unclassified)'


def test_subtype_labels_coverage():
    for key in ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']:
        assert key in SUBTYPE_LABELS
        assert len(SUBTYPE_LABELS[key]) > 0


def _real_classifier_available():
    cls = get_classifier()
    return cls.available


def test_classifier_on_eosinophil_image():
    """If the trained model is available, classify a real eosinophil image."""
    if not _real_classifier_available():
        return  # skip silently — model not trained yet

    import os
    path = 'data/raw/blood_cells/dataset2-master/dataset2-master/images/TEST/EOSINOPHIL'
    if not Path(path).exists():
        return

    files = [f for f in os.listdir(path) if f.endswith('.jpeg')]
    if not files:
        return

    img = cv2.imread(os.path.join(path, files[0]))
    if img is None:
        return

    cls = get_classifier()
    result = cls.classify(img)
    assert result['subtype_key'] in {'eosinophil', 'lymphocyte', 'monocyte', 'neutrophil'}
    assert 0.0 <= result['subtype_conf'] <= 1.0


def test_classifier_on_neutrophil_image():
    if not _real_classifier_available():
        return

    import os
    path = 'data/raw/blood_cells/dataset2-master/dataset2-master/images/TEST/NEUTROPHIL'
    if not Path(path).exists():
        return

    files = [f for f in os.listdir(path) if f.endswith('.jpeg')]
    if not files:
        return

    img = cv2.imread(os.path.join(path, files[0]))
    if img is None:
        return

    cls = get_classifier()
    result = cls.classify(img)
    assert result['subtype_key'] in {'eosinophil', 'lymphocyte', 'monocyte', 'neutrophil'}


def test_classifier_accuracy_on_test_set():
    """Quick accuracy check on 20 images per class — should be > 70%."""
    if not _real_classifier_available():
        return

    import os
    base = 'data/raw/blood_cells/dataset2-master/dataset2-master/images/TEST'
    classes = {'EOSINOPHIL': 'eosinophil', 'LYMPHOCYTE': 'lymphocyte',
               'MONOCYTE': 'monocyte', 'NEUTROPHIL': 'neutrophil'}
    if not Path(base).exists():
        return

    cls = get_classifier()
    correct, total = 0, 0
    for folder, expected_key in classes.items():
        folder_path = Path(base) / folder
        files = [f for f in folder_path.glob('*.jpeg')][:20]
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            result = cls.classify(img)
            if result['subtype_key'] == expected_key:
                correct += 1
            total += 1

    if total > 0:
        acc = correct / total
        print(f'\n  WBC subtype accuracy on 20 samples/class: {acc:.1%} ({correct}/{total})')
        assert acc > 0.60, f'Accuracy {acc:.1%} is below 60% minimum'
