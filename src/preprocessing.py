# import cv2

# def preprocess_image(image_path):

#     img = cv2.imread(image_path)

#     # Gaussian Blur
#     img = cv2.GaussianBlur(img, (5,5), 0)

#     # Convert to LAB
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

#     l,a,b = cv2.split(lab)

#     # CLAHE
#     clahe = cv2.createCLAHE(
#         clipLimit=2.0,
#         tileGridSize=(8,8)
#     )

#     cl = clahe.apply(l)

#     merged = cv2.merge((cl,a,b))

#     final = cv2.cvtColor(
#         merged,
#         cv2.COLOR_LAB2BGR
#     )

#     return final

"""
Step 4.1 — Image Preprocessing and Enhancement
- Gaussian filtering (noise reduction)
- RGB → LAB color space transformation
- CLAHE contrast enhancement
"""
import cv2
import numpy as np


def preprocess_image(image_path: str) -> np.ndarray:
    """Full preprocessing pipeline. Returns enhanced BGR image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return preprocess_array(img)


def preprocess_array(img: np.ndarray) -> np.ndarray:
    """Apply preprocessing to a numpy BGR image array."""
    # 1. Gaussian filtering — kernel 3×3, sigma=1.5 (reduce speckle noise)
    blurred = cv2.GaussianBlur(img, (3, 3), 1.5)

    # 2. RGB→LAB color space transformation (separate luminance from color)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    # 3. CLAHE on L-channel only (clip limit 3.5, tile grid 8×8)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # 4. Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

