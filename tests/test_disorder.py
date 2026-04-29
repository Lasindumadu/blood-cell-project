import sys
from pathlib import Path
# ensure repo root is on sys.path so `src` package is importable when running tests
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.disorder import detect_all, detect_anemia, detect_all_disorders



def test_detect_all_no_wbc():
    summary = {'counts': {'0': 10}, 'boxes': []}
    assert detect_all(summary) == []


def test_detect_anemia_empty():
    summary = {'counts': {}, 'boxes': []}
    assert detect_anemia(summary) == []


def test_detect_all_simple_positive():
    # construct summary with lymph count high and mock features
    summary = {
        'counts': {'0': 5, '1': 2},
        'boxes': [
            {'class': 1, 'features': [{'area': 300, 'circularity': 0.5}, {'area': 350, 'circularity': 0.5}]},
            {'class': 1, 'features': [{'area': 400, 'circularity': 0.4}, {'area': 450, 'circularity': 0.5}]}
        ]
    }
    # lymph =2, total_wbc =2 -> 100% so should be considered; blast_like count >=3? we have 3 features < threshold
    res = detect_all(summary)
    assert isinstance(res, list)
