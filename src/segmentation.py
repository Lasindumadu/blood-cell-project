# import cv2
# import numpy as np
# from skimage import morphology, measure
# from scipy import ndimage as ndi


# def segment_cells(image):
#     """Marker-controlled watershed segmentation returning binary mask of cells.

#     Returns a binary mask (uint8 0/255) representing segmented regions (cells).
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Contrast-enhanced + denoise
#     blur = cv2.GaussianBlur(gray, (3, 3), 0)

#     # Otsu threshold
#     _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Invert if background is dark
#     if np.mean(thresh) < 128:
#         thresh = cv2.bitwise_not(thresh)

#     # Morphological opening to remove small noise
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

#     # Sure background and sure foreground
#     sure_bg = cv2.dilate(opening, kernel, iterations=2)
#     dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
#     _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)

#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg, sure_fg)

#     # Marker labelling
#     ret, markers = cv2.connectedComponents(sure_fg)
#     # Add one to all labels so background is 1 instead of 0
#     markers = markers + 1
#     # Mark the unknown region with zero
#     markers[unknown == 255] = 0

#     # Apply watershed
#     markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), markers)

#     # Cells mask: markers >1
#     mask = np.zeros_like(gray, dtype=np.uint8)
#     mask[markers > 1] = 255

#     # Remove small objects
#     mask_bool = mask.astype(bool)
#     # Use `max_size` for forward-compatibility with newer scikit-image versions.
#     # Note: `max_size` removes objects smaller than or equal to the value.
#     # mask_clean = morphology.remove_small_objects(mask_bool, max_size=100)
#     mask_clean = morphology.remove_small_objects(mask_bool, min_size=100)
#     mask_final = (mask_clean.astype(np.uint8) * 255).astype(np.uint8)

#     return mask_final

"""
Step 4.2 — Cell Segmentation
- Otsu's thresholding
- Distance Transform
- Marker-Controlled Watershed
- Morphological Cleanup
- Contour filtering (100px < area < 5000px)
"""
import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage


def segment_cells(img: np.ndarray):
    """
    Returns: (labeled_image, contours_list, vis_image)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding — automatic foreground/background separation
    _, thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup — opening with 3×3 circular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Distance Transform — find how far each pixel is from background
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

    # Peak local max as markers for watershed
    coords = peak_local_max(dist_transform, min_distance=15,
                             labels=cleaned)
    mask = np.zeros(dist_transform.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)

    # Marker-Controlled Watershed
    labels = watershed(-dist_transform, markers, mask=cleaned)

    # Extract contours and filter by area
    contours_filtered = []
    vis = img.copy()
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        cell_mask = np.zeros(gray.shape, dtype=np.uint8)
        cell_mask[labels == label_id] = 255
        cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if 100 < area < 5000:           # filter noise (<100) and clumps (>5000)
                contours_filtered.append(c)
                cv2.drawContours(vis, [c], -1, (0, 255, 0), 1)

    return labels, contours_filtered, vis
