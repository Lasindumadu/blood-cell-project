"""
Step 4.6 — Hematological Disorder Detection
Rule-based analysis exactly as described in the proposal.
"""

def detect_disorders(cell_results: list) -> dict:
    disorders = {}
    total = len(cell_results)
    if total == 0:
        return {'error': 'No cells detected'}

    wbc_cells = [c for c in cell_results
                 if c['class_name'] in
                 ['WBC','neutrophil','lymphocyte','monocyte','eosinophil']]
    rbc_cells = [c for c in cell_results if c['class_name'] == 'RBC']
    total_wbc = len(wbc_cells)

    # ── ALL Detection ────────────────────────────────────────────────
    blast_like = [c for c in cell_results
                  if c.get('features', {}).get('nuclear_irregularity', 0) > 1.4]
    blast_count       = len(blast_like)
    lymphoblast_frac  = blast_count / (total_wbc + 1e-6)

    all_flag       = False
    all_confidence = 0.0
    all_reasons    = []

    if lymphoblast_frac > 0.20:
        all_reasons.append(
            f"Blast-like cells = {lymphoblast_frac:.1%} of WBC (threshold: >20%)")
        all_confidence += 0.5
    if blast_count > 10:
        all_reasons.append(
            f"{blast_count} cells with nuclear irregularity > 1.4 (threshold: >10)")
        all_confidence += 0.3
    high_nc = [c for c in blast_like
               if c.get('features', {}).get('nucleus_ratio', 0) > 0.7
               and c.get('features', {}).get('area', 0) > 200]
    if high_nc:
        all_reasons.append(
            f"{len(high_nc)} large cells with N:C ratio > 0.7 (supporting indicator)")
        all_confidence += 0.2

    if all_confidence >= 0.5:
        all_flag = True

    disorders['acute_lymphoblastic_leukemia'] = {
        'detected':       all_flag,
        'confidence':     round(all_confidence, 2),
        'blast_count':    blast_count,
        'blast_fraction': round(lymphoblast_frac, 3),
        'reasons':        all_reasons
    }

    # ── Sickle Cell Disease ──────────────────────────────────────────
    low_circ = [c for c in rbc_cells
                if c.get('features', {}).get('circularity', 1) < 0.65]
    sickle_frac = len(low_circ) / (len(rbc_cells) + 1e-6)
    disorders['sickle_cell_disease'] = {
        'detected':                    sickle_frac > 0.05,
        'low_circularity_rbc_fraction': round(sickle_frac, 3),
        'threshold':                   0.05
    }

    # ── Anemia Indicators ───────────────────────────────────────────
    if rbc_cells:
        areas     = [c.get('features', {}).get('area', 0) for c in rbc_cells]
        mean_area = float(sum(areas) / len(areas))
        std_area  = float((sum((a - mean_area)**2 for a in areas) / len(areas))**0.5)
        disorders['anemia_indicators'] = {
            'rbc_mean_area': round(mean_area, 1),
            'rbc_std_area':  round(std_area, 1),
            'high_variation': std_area > 2 * mean_area * 0.15
        }

    disorders['cell_counts'] = {
        'total':    total,
        'wbc':      total_wbc,
        'rbc':      len(rbc_cells),
        'platelets': len([c for c in cell_results if c['class_name'] == 'platelet'])
    }
    return disorders