"""Feature extraction: geometric, nuclear, color (HSV), and textural features.

Implements the 22-feature vector described in Section 4.3 of the proposal.
"""
import cv2
import numpy as np
from skimage import morphology
from skimage.feature import local_binary_pattern

try:
    from skimage.feature import greycomatrix, greycoprops
except ImportError:
    from skimage.feature import graycomatrix as greycomatrix, graycoprops as greycoprops


# ---------------------------------------------------------------------------
# Geometric features from binary mask contours
# ---------------------------------------------------------------------------

def contour_props_from_mask(mask: np.ndarray):
    """Return list of contour property dicts from a binary mask."""
    contours, _ = cv2.findContours(
        mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    props = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) if hull is not None else 0
        convexity = area / (hull_area + 1e-8)

        eccentricity = 0.0
        if len(cnt) >= 5:
            (cx, cy), (MA, ma), _ = cv2.fitEllipse(cnt)
            a = max(MA, ma) / 2.0
            b = min(MA, ma) / 2.0
            if a > 0:
                eccentricity = np.sqrt(1.0 - (b / a) ** 2)

        props.append({
            'area': float(area),
            'perimeter': float(perimeter),
            'circularity': float(np.clip(circularity, 0.0, 1.0)),
            'convexity': float(np.clip(convexity, 0.0, 1.0)),
            'eccentricity': float(eccentricity),
        })
    return props


# ---------------------------------------------------------------------------
# HSV color features (Section 4.3 — Color Features in HSV Space)
# ---------------------------------------------------------------------------

def extract_hsv_features(rgb_crop: np.ndarray) -> dict:
    """Compute hue/saturation/value statistics from an RGB crop."""
    hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return {
        'hue_mean': float(np.mean(h)),
        'hue_std': float(np.std(h)),
        'saturation_mean': float(np.mean(s)),
        'saturation_std': float(np.std(s)),
        'value_mean': float(np.mean(v)),
        'value_std': float(np.std(v)),
    }


# ---------------------------------------------------------------------------
# Nuclear / cytoplasm features (WBC-specific, Section 4.3)
# ---------------------------------------------------------------------------

def extract_nucleus_cytoplasm_metrics(rgb_crop: np.ndarray) -> dict:
    """Segment nucleus via blue-channel thresholding and return N/C metrics + texture."""
    b_channel = rgb_crop[:, :, 2]
    blur = cv2.GaussianBlur(b_channel, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    th_bool = morphology.remove_small_objects(th.astype(bool), max_size=49)
    nuc_mask = (th_bool.astype('uint8') * 255)

    nucleus_area = int(np.sum(nuc_mask > 0))
    total_area = rgb_crop.shape[0] * rgb_crop.shape[1]
    nc_ratio = nucleus_area / (total_area + 1e-8)

    # Nuclear irregularity: circularity of the largest nucleus contour
    nuclear_irregularity = 0.0
    nuc_cnts, _ = cv2.findContours(nuc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if nuc_cnts:
        largest = max(nuc_cnts, key=cv2.contourArea)
        n_area = cv2.contourArea(largest)
        n_peri = cv2.arcLength(largest, True)
        if n_peri > 0 and n_area > 0:
            circ = 4 * np.pi * n_area / (n_peri ** 2 + 1e-8)
            # irregularity = inverse of circularity: 1/circ (perfect circle = 1.0)
            nuclear_irregularity = float(1.0 / (circ + 1e-8))

    # Nuclear compactness and solidity (proposal §4.3)
    nuclear_compactness = 0.0
    nuclear_solidity = 0.0
    nuc_cnts_props, _ = cv2.findContours(nuc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if nuc_cnts_props:
        largest_nuc = max(nuc_cnts_props, key=cv2.contourArea)
        n_area = cv2.contourArea(largest_nuc)
        n_peri = cv2.arcLength(largest_nuc, True)
        if n_area > 0 and n_peri > 0:
            nuclear_compactness = float(n_peri ** 2 / (4 * np.pi * n_area + 1e-8))
        hull = cv2.convexHull(largest_nuc)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            nuclear_solidity = float(n_area / hull_area)

    # GLCM texture features (including entropy — proposal §4.3)
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2GRAY)
    levels = 8
    bins = np.linspace(0, 256, levels + 1)
    gray_q = np.clip(np.digitize(gray, bins) - 1, 0, levels - 1).astype('uint8')
    try:
        glcm = greycomatrix(gray_q, distances=[1], angles=[0], levels=levels,
                            symmetric=True, normed=True)
        contrast = float(greycoprops(glcm, 'contrast')[0, 0])
        dissimilarity = float(greycoprops(glcm, 'dissimilarity')[0, 0])
        homogeneity = float(greycoprops(glcm, 'homogeneity')[0, 0])
        energy = float(greycoprops(glcm, 'energy')[0, 0])
        correlation = float(greycoprops(glcm, 'correlation')[0, 0])
        # Entropy from GLCM: -sum(p * log2(p))
        glcm_flat = glcm[:, :, 0, 0]
        glcm_flat = glcm_flat / (glcm_flat.sum() + 1e-8)
        entropy = float(-np.sum(glcm_flat * np.log2(glcm_flat + 1e-10)))
    except Exception:
        contrast = dissimilarity = homogeneity = energy = correlation = entropy = 0.0

    # LBP texture features
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return {
        'nc_ratio': float(nc_ratio),
        'nucleus_area': nucleus_area,
        'nuclear_irregularity': nuclear_irregularity,
        'nuclear_compactness': nuclear_compactness,
        'nuclear_solidity': nuclear_solidity,
        'glcm_contrast': contrast,
        'glcm_dissimilarity': dissimilarity,
        'glcm_homogeneity': homogeneity,
        'glcm_energy': energy,
        'glcm_correlation': correlation,
        'glcm_entropy': entropy,
        'lbp_hist': lbp_hist.tolist(),
        'nucleus_mask': nuc_mask,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(mask: np.ndarray, rgb_crop: np.ndarray = None):
    """Return a list of feature dicts for each contour in *mask*.

    If *rgb_crop* is provided, nuclear/cytoplasm and HSV features are added.
    Each dict contains geometric + (optionally) nuclear + HSV + textural features.
    """
    props = contour_props_from_mask(mask)

    if rgb_crop is not None and len(props) > 0:
        nuc = extract_nucleus_cytoplasm_metrics(rgb_crop)
        hsv = extract_hsv_features(rgb_crop)

        # Remove non-serialisable numpy array before attaching to each prop
        nuc_serialisable = {k: v for k, v in nuc.items() if k != 'nucleus_mask'}

        for p in props:
            p.update(nuc_serialisable)
            p.update(hsv)

    return props
