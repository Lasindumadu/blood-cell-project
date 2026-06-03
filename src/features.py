# import cv2
# import numpy as np
# from skimage import measure, color, feature, morphology
# from skimage.feature import local_binary_pattern
# # skimage historically exposes `greycomatrix`/`greycoprops` (British spelling);
# # some versions also provide `graycomatrix`/`graycoprops`. Import robustly.
# try:
#     from skimage.feature import greycomatrix, greycoprops
# except Exception:
#     from skimage.feature import graycomatrix as greycomatrix, graycoprops as greycoprops
# from scipy import ndimage as ndi


# def contour_props_from_mask(mask):
#     """Return list of contour properties from a binary mask."""
#     contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     props = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area < 10:
#             continue
#         perimeter = cv2.arcLength(cnt, True)
#         circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
#         hull = cv2.convexHull(cnt)
#         hull_area = cv2.contourArea(hull) if hull is not None else 0
#         solidity = (area / hull_area) if hull_area > 0 else 0
#         # fit ellipse for eccentricity if possible
#         eccentricity = 0.0
#         if len(cnt) >= 5:
#             ellipse = cv2.fitEllipse(cnt)
#             (cx, cy), (MA, ma), angle = ellipse
#             a = max(MA, ma) / 2.0
#             b = min(MA, ma) / 2.0
#             if a > 0:
#                 eccentricity = np.sqrt(1 - (b ** 2 / a ** 2))

#         props.append({
#             'area': float(area),
#             'perimeter': float(perimeter),
#             'circularity': float(circularity),
#             'solidity': float(solidity),
#             'eccentricity': float(eccentricity)
#         })

#     return props


# def extract_nucleus_cytoplasm_metrics(rgb_crop):
#     """Estimate nucleus segmentation and compute N/C ratio plus texture features.

#     Returns dict with 'nc_ratio', 'glcm' features, 'lbp' histogram, and 'nucleus_mask'.
#     """
#     # Convert to grayscale (use blue channel emphasis)
#     b_channel = rgb_crop[:, :, 2]
#     # adaptive threshold on blue channel
#     blur = cv2.GaussianBlur(b_channel, (5, 5), 0)
#     th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                cv2.THRESH_BINARY_INV, 11, 2)

#     # Morphological cleanup
#     kernel = np.ones((3, 3), np.uint8)
#     th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
#     th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

#     # remove small objects (use `max_size` to be compatible with newer scikit-image)
#     # th = morphology.remove_small_objects(th.astype(bool), max_size=50)
#     th = morphology.remove_small_objects(th.astype(bool), min_size=50)
#     nuc_mask = (th.astype('uint8') * 255).astype('uint8')

#     # compute areas
#     nucleus_area = np.sum(nuc_mask > 0)
#     total_area = rgb_crop.shape[0] * rgb_crop.shape[1]
#     nc_ratio = (nucleus_area / (total_area + 1e-8))

#     # GLCM features on gray image
#     gray = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2GRAY)
#     # quantize to 8 levels
#     levels = 8
#     bins = np.linspace(0, 256, levels + 1)
#     gray_q = np.digitize(gray, bins) - 1
#     # compute GLCM with distance 1 and angles 0
#     try:
#         glcm = greycomatrix(gray_q.astype('uint8'), distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
#         contrast = greycoprops(glcm, 'contrast')[0, 0]
#         dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
#         homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
#         energy = greycoprops(glcm, 'energy')[0, 0]
#         correlation = greycoprops(glcm, 'correlation')[0, 0]
#     except Exception:
#         contrast = dissimilarity = homogeneity = energy = correlation = 0.0

#     # LBP
#     lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
#     n_bins = int(lbp.max() + 1)
#     lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

#     return {
#         'nc_ratio': float(nc_ratio),
#         'nucleus_area': int(nucleus_area),
#         'glcm_contrast': float(contrast),
#         'glcm_dissimilarity': float(dissimilarity),
#         'glcm_homogeneity': float(homogeneity),
#         'glcm_energy': float(energy),
#         'glcm_correlation': float(correlation),
#         'lbp_hist': lbp_hist.tolist(),
#         'nucleus_mask': nuc_mask
#     }


# def extract_features(mask, rgb_crop=None):
#     """Public API: given a binary mask (or None) and optional rgb crop, return features list.

#     If rgb_crop is provided, nucleus/cytoplasm metrics will be computed per object.
#     """
#     props = contour_props_from_mask(mask)
#     if rgb_crop is not None and len(props) > 0:
#         # attach nucleus/cytoplasm metrics to the largest contour's props for now
#         nuc = extract_nucleus_cytoplasm_metrics(rgb_crop)
#         # add top-level nucleus metrics to each prop to keep shape consistent
#         for p in props:
#             p.update({
#                 'nc_ratio': nuc['nc_ratio'],
#                 'nucleus_area': nuc['nucleus_area'],
#                 'glcm_contrast': nuc['glcm_contrast'],
#                 'lbp_hist': nuc['lbp_hist']
#             })

#     return props

"""
Step 4.3 — Morphological and Textural Feature Extraction
Extracts 22 features per cell:
  Geometric (5): area, perimeter, circularity, eccentricity, convexity
  Nuclear (4): nucleus_ratio, nuclear_irregularity, compactness, solidity
  Color/HSV (4): hue_mean, hue_std, saturation_mean, value_mean
  Textural (4): GLCM energy, entropy, contrast, homogeneity
  + LBP variance, color histogram entropy (×4 bins)
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def extract_features(img: np.ndarray, contour) -> dict:
    """Extract all 22 features for a single cell contour."""
    x, y, w, h = cv2.boundingRect(contour)
    # Pad a little
    pad = 5
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)
    cell_roi = img[y1:y2, x1:x2]

    features = {}

    # ── GEOMETRIC FEATURES ──────────────────────────────────────────────
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    features['area'] = float(area)
    features['perimeter'] = float(perimeter)

    # Circularity: 4π×Area / Perimeter²
    features['circularity'] = (4 * np.pi * area / (perimeter ** 2 + 1e-6))

    # Eccentricity via ellipse fitting
    if len(contour) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(contour)
        features['eccentricity'] = float(np.sqrt(1 - (min(MA, ma) / (max(MA, ma) + 1e-6)) ** 2))
    else:
        features['eccentricity'] = 0.0

    # Convexity: area / convex hull area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    features['convexity'] = float(area / (hull_area + 1e-6))

    # ── NUCLEAR CHARACTERISTICS (WBC) ────────────────────────────────────
    if cell_roi.size > 0:
        # Use blue channel (hematoxylin staining targets nucleus)
        blue_ch = cell_roi[:, :, 0]
        _, nuc_mask = cv2.threshold(blue_ch, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        nuc_area = float(np.sum(nuc_mask > 0))
        features['nucleus_ratio'] = nuc_area / (area + 1e-6)

        # Nuclear irregularity: perimeter²/(4π×area)
        nuc_cnts, _ = cv2.findContours(nuc_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if nuc_cnts:
            nc = max(nuc_cnts, key=cv2.contourArea)
            np_ = cv2.arcLength(nc, True)
            na = cv2.contourArea(nc)
            features['nuclear_irregularity'] = float(np_ ** 2 / (4 * np.pi * na + 1e-6))
        else:
            features['nuclear_irregularity'] = 0.0

        # Compactness & Solidity
        hull_nuc = cv2.convexHull(contour)
        hull_nuc_area = cv2.contourArea(hull_nuc)
        features['nuclear_compactness'] = float(
            4 * np.pi * area / (perimeter ** 2 + 1e-6))
        features['nuclear_solidity'] = float(area / (hull_nuc_area + 1e-6))
    else:
        features.update({'nucleus_ratio': 0, 'nuclear_irregularity': 0,
                          'nuclear_compactness': 0, 'nuclear_solidity': 0})

    # ── COLOR FEATURES (HSV) ─────────────────────────────────────────────
    if cell_roi.size > 0:
        hsv = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        features['hue_mean'] = float(np.mean(hsv[:, :, 0]))
        features['hue_std'] = float(np.std(hsv[:, :, 0]))
        features['saturation_mean'] = float(np.mean(hsv[:, :, 1]))
        features['value_mean'] = float(np.mean(hsv[:, :, 2]))
    else:
        features.update({'hue_mean': 0, 'hue_std': 0,
                          'saturation_mean': 0, 'value_mean': 0})

    # ── TEXTURAL FEATURES (GLCM + LBP) ───────────────────────────────────
    if cell_roi.size > 0 and cell_roi.shape[0] > 4 and cell_roi.shape[1] > 4:
        gray_roi = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)
        # GLCM
        glcm = graycomatrix(gray_roi, distances=[1], angles=[0],
                             levels=256, symmetric=True, normed=True)
        features['glcm_energy'] = float(graycoprops(glcm, 'energy')[0, 0])
        features['glcm_contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
        features['glcm_homogeneity'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
        # Entropy from GLCM
        glcm_flat = glcm[:, :, 0, 0].ravel()
        glcm_flat = glcm_flat[glcm_flat > 0]
        features['glcm_entropy'] = float(-np.sum(glcm_flat * np.log2(glcm_flat + 1e-12)))

        # Local Binary Pattern — granularity detection
        lbp = local_binary_pattern(gray_roi, P=8, R=1, method='uniform')
        features['lbp_variance'] = float(np.var(lbp))
    else:
        features.update({'glcm_energy': 0, 'glcm_contrast': 0,
                          'glcm_homogeneity': 0, 'glcm_entropy': 0,
                          'lbp_variance': 0})

    return features  # returns 22 features total