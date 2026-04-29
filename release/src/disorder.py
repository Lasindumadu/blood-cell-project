"""Rule-based disorder detection utilities.

Functions operate on the summary dict produced by `src.pipeline.analyze_image`.
"""
from typing import List, Dict
import yaml
from pathlib import Path


def _load_config():
    cfg_path = Path('config') / 'disorders.yaml'
    default = {
        'lymph_class': '1',
        'lymph_fraction_threshold': 0.2,
        'blast_like_min': 3,
        'blast_like_area': 200,
        'blast_like_circularity': 0.6,
        'rbc_class': '0',
        'rbc_area_threshold': 500
    }
    if cfg_path.exists():
        try:
            with open(cfg_path, 'r') as f:
                cfg = yaml.safe_load(f)
            default.update(cfg or {})
        except Exception:
            pass
    return default


def detect_all(summary: Dict) -> List[str]:
    cfg = _load_config()
    lymph_class = str(cfg['lymph_class'])
    lymph_fraction_threshold = float(cfg['lymph_fraction_threshold'])
    blast_like_min = int(cfg['blast_like_min'])
    blast_like_area = float(cfg['blast_like_area'])
    blast_like_circularity = float(cfg['blast_like_circularity'])

    counts = summary.get('counts', {})
    boxes = summary.get('boxes', [])
    total_wbc = sum(v for k, v in counts.items() if k != cfg['rbc_class'])
    if total_wbc == 0:
        return []

    lymph = counts.get(lymph_class, 0)
    disorders = []
    if lymph / total_wbc > lymph_fraction_threshold:
        blast_like = 0
        for b in boxes:
            if str(b.get('class')) == lymph_class:
                for f in b.get('features', []):
                    if f.get('circularity', 1.0) < blast_like_circularity and f.get('area', 0) > blast_like_area:
                        blast_like += 1
        if blast_like >= blast_like_min:
            disorders.append('Suspected ALL')
    return disorders


def detect_anemia(summary: Dict) -> List[str]:
    cfg = _load_config()
    rbc_class = str(cfg['rbc_class'])
    rbc_area_threshold = float(cfg['rbc_area_threshold'])

    counts = summary.get('counts', {})
    boxes = summary.get('boxes', [])
    rbc_boxes = [b for b in boxes if str(b.get('class')) == rbc_class]
    if not rbc_boxes:
        return []

    areas = []
    for b in rbc_boxes:
        for f in b.get('features', []):
            areas.append(f.get('area', 0))
    if not areas:
        return []

    avg_area = sum(areas) / len(areas)
    disorders = []
    if avg_area < rbc_area_threshold:
        disorders.append('Suspected microcytic anemia (low RBC area)')

    return disorders


def detect_all_disorders(summary: Dict) -> List[str]:
    out = []
    out += detect_all(summary)
    out += detect_anemia(summary)
    return out
