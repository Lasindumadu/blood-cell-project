import cv2
import numpy as np
from skimage import morphology, measure
from scipy import ndimage as ndi


def segment_cells(image):
    """Marker-controlled watershed segmentation returning binary mask of cells.

    Returns a binary mask (uint8 0/255) representing segmented regions (cells).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Contrast-enhanced + denoise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu threshold
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if background is dark
    if np.mean(thresh) < 128:
        thresh = cv2.bitwise_not(thresh)

    # Morphological opening to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Sure background and sure foreground
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so background is 1 instead of 0
    markers = markers + 1
    # Mark the unknown region with zero
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), markers)

    # Cells mask: markers >1
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255

    # Remove small objects
    mask_bool = mask.astype(bool)
    # Use `max_size` for forward-compatibility with newer scikit-image versions.
    # Note: `max_size` removes objects smaller than or equal to the value.
    mask_clean = morphology.remove_small_objects(mask_bool, max_size=100)
    mask_final = (mask_clean.astype(np.uint8) * 255).astype(np.uint8)

    return mask_final