import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np
import cv2
from src.preprocessing import preprocess_image
from src.segmentation import segment_cells
from src.features import extract_features, extract_hsv_features


def test_preprocess_preserves_shape(tmp_path):
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img_path = tmp_path / "tmp.jpg"
    cv2.imwrite(str(img_path), img)
    out = preprocess_image(str(img_path))
    assert isinstance(out, np.ndarray)
    assert out.shape == img.shape


def test_preprocess_output_dtype(tmp_path):
    img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    img_path = tmp_path / "rand.jpg"
    cv2.imwrite(str(img_path), img)
    out = preprocess_image(str(img_path))
    assert out.dtype == np.uint8


def test_segment_cells_returns_mask():
    # Create a synthetic image with a white circle (simulated cell) on dark background
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (200, 150, 180), -1)
    mask = segment_cells(img)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_extract_hsv_features():
    rgb = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
    feats = extract_hsv_features(rgb)
    for key in ['hue_mean', 'hue_std', 'saturation_mean', 'saturation_std', 'value_mean', 'value_std']:
        assert key in feats
        assert isinstance(feats[key], float)


def test_extract_features_from_mask():
    mask = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(mask, (40, 40), 25, 255, -1)
    feats = extract_features(mask)
    assert isinstance(feats, list)
    assert len(feats) >= 1
    for f in feats:
        assert 'area' in f
        assert 'circularity' in f
        assert f['area'] > 0


def test_extract_features_with_rgb_crop():
    mask = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(mask, (40, 40), 25, 255, -1)
    rgb = (np.random.rand(80, 80, 3) * 255).astype(np.uint8)
    feats = extract_features(mask, rgb_crop=rgb)
    assert len(feats) >= 1
    # Verify all 22 scalar features are present
    expected = [
        'area', 'perimeter', 'circularity', 'convexity', 'eccentricity',
        'nc_ratio', 'nucleus_area', 'nuclear_irregularity',
        'nuclear_compactness', 'nuclear_solidity',
        'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
        'glcm_energy', 'glcm_correlation', 'glcm_entropy',
        'hue_mean', 'hue_std', 'saturation_mean',
        'saturation_std', 'value_mean', 'value_std',
    ]
    for key in expected:
        assert key in feats[0], f'Missing feature: {key}'
    scalar_keys = [k for k in feats[0] if k != 'lbp_hist']
    assert len(scalar_keys) == 22, f'Expected 22 features, got {len(scalar_keys)}'


def test_segmentation_removes_large_clumps():
    from src.segmentation import segment_cells
    # Large white blob > 5000px should be filtered out
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (190, 190), (200, 150, 180), -1)  # ~32400px blob
    mask = segment_cells(img)
    # After filtering, no object > 5000px should remain
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 5000]
    assert len(large) == 0, f'Large clumps not filtered: {large}'
