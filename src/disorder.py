"""Rule-based disorder detection.

Works with both 3-class output (WBC/RBC/Platelet) and 6-class output (subtype-resolved WBCs).
When subtype_counts is present in the summary, richer rules are applied.
"""
from typing import List, Dict
import yaml
from pathlib import Path


def _load_config() -> Dict:
    cfg_path = Path('config') / 'disorders.yaml'
    default = {
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
        # Subtype-based rules
        'nuclear_irregularity_threshold': 1.4,  # proposal §4.3: nuclear irregularity > 1.4 => blast cell
        'lymphocyte_all_fraction': 0.20,   # proposal §4.6: lymphoblasts > 20% of WBCs => ALL suspicion
        'eosinophil_high_fraction': 0.10,  # eosinophils > 10% of WBCs => eosinophilia
    }
    if cfg_path.exists():
        try:
            with open(cfg_path, 'r') as f:
                cfg = yaml.safe_load(f)
            if cfg:
                default.update(cfg)
        except Exception:
            pass
    return default


def _get_class_id(cfg, key, boxes, name) -> str:
    for b in boxes:
        if str(b.get('class_name', '')).lower() == name.lower():
            return str(b.get('class'))
    return str(cfg.get(key, '0'))


# ---------------------------------------------------------------------------

def detect_leukemia(summary: Dict, cfg: Dict) -> List[str]:
    """ALL / leukocytosis detection — uses subtype_counts when available."""
    boxes = summary.get('boxes', [])
    counts = summary.get('counts', {})
    subtype_counts = summary.get('subtype_counts', {})

    wbc_id = _get_class_id(cfg, 'wbc_class', boxes, 'WBC')
    rbc_id = _get_class_id(cfg, 'rbc_class', boxes, 'RBC')
    plt_id = _get_class_id(cfg, 'platelet_class', boxes, 'Platelets')

    wbc_count = counts.get(wbc_id, 0)
    rbc_count = counts.get(rbc_id, 0)
    plt_count = counts.get(plt_id, 0)
    total_cells = wbc_count + rbc_count + plt_count

    if total_cells == 0 or wbc_count == 0:
        return []

    wbc_fraction = wbc_count / total_cells
    disorders = []

    # --- Subtype-based ALL rule (richer, when classifier available) ---
    if subtype_counts:
        lymph = subtype_counts.get('lymphocyte', 0)
        lymph_frac_of_wbc = lymph / wbc_count if wbc_count > 0 else 0
        lymph_frac_thresh = float(cfg.get('lymphocyte_all_fraction', 0.40))

        if wbc_count >= 5 and lymph_frac_of_wbc >= lymph_frac_thresh and wbc_fraction >= 0.15:
            # Check morphology for blast-like cells
            nuc_irr = float(cfg.get('nuclear_irregularity_threshold', 1.4))
            blast_count = sum(
                1 for b in boxes
                if b.get('subtype_key') == 'lymphocyte'
                for f in b.get('features', [])
                if f.get('area', 0) > float(cfg.get('blast_like_area', 200))
                and (
                    f.get('circularity', 1.0) < float(cfg.get('blast_like_circularity', 0.55))
                    or f.get('nuclear_irregularity', 0.0) > nuc_irr
                )
            )
            disorders.append(
                f'Suspected ALL: Lymphocytes={lymph} ({lymph_frac_of_wbc:.0%} of WBCs)'
                + (f', {blast_count} blast-like morphologies' if blast_count > 0 else '')
            )

        # Eosinophilia — require at least 5 WBCs to avoid small-sample false positives
        eosino = subtype_counts.get('eosinophil', 0)
        eosino_frac = eosino / wbc_count if wbc_count > 0 else 0
        if wbc_count >= 5 and eosino_frac >= float(cfg.get('eosinophil_high_fraction', 0.10)):
            disorders.append(
                f'Eosinophilia: {eosino} eosinophils ({eosino_frac:.0%} of WBCs)'
            )

        return disorders

    # --- Fallback: morphology-only rule (no subtype info) ---
    threshold = float(cfg.get('wbc_fraction_threshold', 0.30))
    if wbc_fraction <= threshold:
        return []

    nuc_irr_thresh = float(cfg.get('nuclear_irregularity_threshold', 1.4))
    blast_count = sum(
        1 for b in boxes
        if str(b.get('class')) == wbc_id
        for f in b.get('features', [])
        if f.get('area', 0) > float(cfg.get('blast_like_area', 200))
        and (
            f.get('circularity', 1.0) < float(cfg.get('blast_like_circularity', 0.55))
            or f.get('nuclear_irregularity', 0.0) > nuc_irr_thresh
        )
    )

    if blast_count >= int(cfg.get('blast_like_min', 3)):
        disorders.append(
            f'Suspected ALL: WBCs={wbc_count} ({wbc_fraction:.0%} of cells), '
            f'{blast_count} blast-like morphologies'
        )
    elif wbc_fraction > 0.50:
        disorders.append(
            f'Leukocytosis: elevated WBC count ({wbc_count}, {wbc_fraction:.0%} of cells)'
        )
    return disorders


def detect_anemia(summary: Dict, cfg: Dict) -> List[str]:
    boxes = summary.get('boxes', [])
    counts = summary.get('counts', {})
    rbc_id = _get_class_id(cfg, 'rbc_class', boxes, 'RBC')

    if counts.get(rbc_id, 0) == 0:
        return []

    areas, circs = [], []
    for b in boxes:
        if str(b.get('class')) != rbc_id:
            continue
        for f in b.get('features', []):
            if f.get('area', 0) > 0:
                areas.append(f['area'])
                circs.append(f.get('circularity', 1.0))

    disorders = []
    if areas:
        mean_area = sum(areas) / len(areas)
        if mean_area < float(cfg.get('rbc_area_mean_threshold', 150)):
            disorders.append(
                f'Suspected microcytic anemia: mean RBC area={mean_area:.1f}px'
            )
    if circs:
        circ_thresh = float(cfg.get('rbc_circularity_low', 0.20))
        frac_thresh = float(cfg.get('rbc_sickle_fraction', 0.30))
        n_sickle = sum(1 for c in circs if c < circ_thresh)
        frac = n_sickle / len(circs)
        if frac >= frac_thresh:
            disorders.append(
                f'Suspected sickle cell disease: {n_sickle}/{len(circs)} '
                f'RBCs have very low circularity ({frac:.0%})'
            )
    return disorders


def detect_thrombocytopenia(summary: Dict, cfg: Dict) -> List[str]:
    boxes = summary.get('boxes', [])
    counts = summary.get('counts', {})
    plt_id = _get_class_id(cfg, 'platelet_class', boxes, 'Platelets')
    rbc_id = _get_class_id(cfg, 'rbc_class', boxes, 'RBC')
    wbc_id = _get_class_id(cfg, 'wbc_class', boxes, 'WBC')

    plt_count = counts.get(plt_id, 0)
    total = plt_count + counts.get(rbc_id, 0) + counts.get(wbc_id, 0)
    if total == 0:
        return []

    ratio = plt_count / total
    if ratio < float(cfg.get('platelet_rbc_ratio_low', 0.02)) and plt_count < 3:
        return [
            f'Suspected thrombocytopenia: only {plt_count} platelets '
            f'({ratio:.1%} of total cells)'
        ]
    return []


def detect_all_disorders(summary: Dict) -> List[str]:
    cfg = _load_config()
    out = []
    out += detect_leukemia(summary, cfg)
    out += detect_anemia(summary, cfg)
    out += detect_thrombocytopenia(summary, cfg)
    return out
