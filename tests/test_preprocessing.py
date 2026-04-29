import sys
from pathlib import Path
# ensure repo root is on sys.path so `src` package is importable when running tests
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np
from src.preprocessing import preprocess_image
import cv2


def test_preprocess_preserves_shape(tmp_path):
    # create a dummy image
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img_path = tmp_path / "tmp.jpg"
    cv2.imwrite(str(img_path), img)

    out = preprocess_image(str(img_path))
    assert isinstance(out, np.ndarray)
    assert out.shape == img.shape
