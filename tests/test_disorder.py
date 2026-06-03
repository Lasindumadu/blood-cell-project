import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.disorder import detect_leukemia as detect_all_leukemia, detect_anemia, detect_thrombocytopenia, detect_all_disorders


# Default config used by all tests
_CFG = {
    'wbc_class': '0',
    'rbc_class': '1',
    'platelet_class': '2',
    'wbc_fraction_threshold': 0.30,
    'blast_like_min': 3,
    'blast_like_area': 200,
    'blast_like_circularity': 0.55,
    'nc_ratio_blast_threshold': 0.5,
    'rbc_area_mean_threshold': 150,
    'rbc_circularity_low': 0.20,
    'rbc_sickle_fraction': 0.30,
    'platelet_rbc_ratio_low': 0.02,
}


# ---------------------------------------------------------------------------
# Leukemia / ALL detection
# ---------------------------------------------------------------------------

def test_leukemia_no_wbc():
    summary = {'counts': {'1': 50, '2': 5}, 'boxes': []}
    assert detect_all_leukemia(summary, _CFG) == []


def test_leukemia_low_wbc_fraction():
    # 2 WBCs out of 50 total = 4% — below threshold
    summary = {'counts': {'0': 2, '1': 46, '2': 2}, 'boxes': []}
    assert detect_all_leukemia(summary, _CFG) == []


def test_leukemia_high_wbc_no_blast():
    # 20 WBCs out of 30 total = 67% — above threshold but no blast morphology
    summary = {
        'counts': {'0': 20, '1': 8, '2': 2},
        'boxes': [
            {'class': 0, 'class_name': 'WBC',
             'features': [{'area': 100, 'circularity': 0.8, 'nc_ratio': 0.2}]},
        ]
    }
    result = detect_all_leukemia(summary, _CFG)
    # Should flag leukocytosis, not ALL (no blast morphology)
    assert any('Leukocytosis' in r for r in result) or result == []


def test_leukemia_blast_detected():
    # 15 WBCs out of 20 total = 75%, 4 blast-like morphologies
    # nuclear_irregularity > 1.4 matches proposal §4.3 blast criterion
    blast_feat = {'area': 300, 'circularity': 0.40, 'nc_ratio': 0.6, 'nuclear_irregularity': 1.6}
    boxes = [
        {'class': 0, 'class_name': 'WBC', 'features': [blast_feat]},
        {'class': 0, 'class_name': 'WBC', 'features': [blast_feat]},
        {'class': 0, 'class_name': 'WBC', 'features': [blast_feat]},
        {'class': 0, 'class_name': 'WBC', 'features': [blast_feat]},
    ]
    summary = {'counts': {'0': 15, '1': 4, '2': 1}, 'boxes': boxes}
    result = detect_all_leukemia(summary, _CFG)
    assert any('ALL' in r for r in result), f"Expected ALL flag, got: {result}"


# ---------------------------------------------------------------------------
# Anemia detection
# ---------------------------------------------------------------------------

def test_anemia_no_rbc():
    summary = {'counts': {'0': 5}, 'boxes': []}
    assert detect_anemia(summary, _CFG) == []


def test_anemia_normal_rbc_area():
    boxes = [
        {'class': 1, 'class_name': 'RBC', 'features': [{'area': 300, 'circularity': 0.80}]},
        {'class': 1, 'class_name': 'RBC', 'features': [{'area': 350, 'circularity': 0.78}]},
    ]
    summary = {'counts': {'1': 2}, 'boxes': boxes}
    assert detect_anemia(summary, _CFG) == []


def test_anemia_microcytic():
    boxes = [
        {'class': 1, 'class_name': 'RBC', 'features': [{'area': 50, 'circularity': 0.75}]},
        {'class': 1, 'class_name': 'RBC', 'features': [{'area': 60, 'circularity': 0.70}]},
    ]
    summary = {'counts': {'1': 2}, 'boxes': boxes}
    result = detect_anemia(summary, _CFG)
    assert any('microcytic' in r.lower() for r in result), f"Expected microcytic flag, got: {result}"


def test_anemia_sickle_cell():
    # Most RBCs have very low circularity (< 0.20)
    boxes = [
        {'class': 1, 'class_name': 'RBC', 'features': [{'area': 200, 'circularity': 0.10}]},
        {'class': 1, 'class_name': 'RBC', 'features': [{'area': 200, 'circularity': 0.12}]},
        {'class': 1, 'class_name': 'RBC', 'features': [{'area': 200, 'circularity': 0.15}]},
        {'class': 1, 'class_name': 'RBC', 'features': [{'area': 200, 'circularity': 0.70}]},
    ]
    summary = {'counts': {'1': 4}, 'boxes': boxes}
    result = detect_anemia(summary, _CFG)
    assert any('sickle' in r.lower() for r in result), f"Expected sickle flag, got: {result}"


# ---------------------------------------------------------------------------
# Thrombocytopenia detection
# ---------------------------------------------------------------------------

def test_thrombocytopenia_normal():
    summary = {'counts': {'0': 5, '1': 80, '2': 10}, 'boxes': []}
    assert detect_thrombocytopenia(summary, _CFG) == []


def test_thrombocytopenia_low_platelet():
    summary = {'counts': {'0': 5, '1': 100, '2': 1}, 'boxes': []}
    result = detect_thrombocytopenia(summary, _CFG)
    assert any('thrombocytopenia' in r.lower() for r in result), f"Expected thrombocytopenia flag, got: {result}"


# ---------------------------------------------------------------------------
# Integration: detect_all_disorders
# ---------------------------------------------------------------------------

def test_detect_all_disorders_returns_list():
    summary = {'counts': {}, 'boxes': []}
    result = detect_all_disorders(summary)
    assert isinstance(result, list)
