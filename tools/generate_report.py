"""Generate clinical reports (PDF and CSV) from an annotated image and the summary JSON.

Usage:
  python tools/generate_report.py --image results/BloodImage_00007_annotated.jpg \
      --summary results/BloodImage_00007_summary.json --out reports/report.pdf
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.utils import ImageReader


# Human-readable class name mapping (fallback when not embedded in JSON)
_CLASS_LABELS = {
    '0': 'WBC (White Blood Cell)',
    '1': 'RBC (Red Blood Cell)',
    '2': 'Platelet',
    0: 'WBC (White Blood Cell)',
    1: 'RBC (Red Blood Cell)',
    2: 'Platelet',
}

_NORMAL_RANGES = {
    'WBC': '4,500–11,000 /µL',
    'RBC': '4.5–5.9 M/µL (M), 4.0–5.2 M/µL (F)',
    'Platelet': '150,000–400,000 /µL',
}


def _label(class_key, class_names: dict = None) -> str:
    if class_names:
        k = int(class_key) if str(class_key).isdigit() else class_key
        if k in class_names:
            return class_names[k]
    return _CLASS_LABELS.get(class_key, f'Class {class_key}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--summary', required=True)
    p.add_argument('--out', required=True)
    return p.parse_args()


def generate(image_path, summary_path, out_pdf):
    image_path = Path(image_path)
    summary_path = Path(summary_path)
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    counts = summary.get('counts', {})
    disorders = summary.get('disorders', [])
    boxes = summary.get('boxes', [])

    # Build class-name map from boxes if available
    class_names = {}
    for b in boxes:
        cid = b.get('class')
        cname = b.get('class_name', '')
        if cid is not None and cname:
            class_names[int(cid)] = cname

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    # --- Title ---
    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'], fontSize=16, spaceAfter=6,
        textColor=colors.HexColor('#1a3a6b')
    )
    story.append(Paragraph('Automated Blood Cell Analysis Report', title_style))
    story.append(Paragraph(
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;&nbsp;|&nbsp;&nbsp; '
        f'Source: {image_path.stem}',
        styles['Normal']
    ))
    story.append(Spacer(1, 0.3 * cm))

    # --- Annotated image ---
    if image_path.exists():
        img_reader = ImageReader(str(image_path))
        iw, ih = img_reader.getSize()
        max_w = 16 * cm
        scale = min(max_w / iw, 10 * cm / ih)
        story.append(RLImage(str(image_path), width=iw * scale, height=ih * scale))
    story.append(Spacer(1, 0.4 * cm))

    # --- Cell Count Summary table ---
    story.append(Paragraph('Complete Blood Count (CBC) Summary', styles['Heading2']))

    total = sum(int(v) for v in counts.values())
    count_data = [['Cell Type', 'Count', '% of Total', 'Normal Range']]
    for k, v in counts.items():
        name = _label(k, class_names)
        pct = f'{int(v)/total*100:.1f}%' if total > 0 else 'N/A'
        short = name.split(' ')[0]
        normal = _NORMAL_RANGES.get(short, '—')
        count_data.append([name, str(v), pct, normal])
    count_data.append(['Total cells detected', str(total), '100%', ''])

    # WBC subtype breakdown (if available)
    subtype_counts = summary.get('subtype_counts', {})
    wbc_available = summary.get('wbc_subtype_available', False)
    if wbc_available and subtype_counts:
        wbc_total = sum(subtype_counts.values()) or 1
        for key in ['neutrophil', 'lymphocyte', 'monocyte', 'eosinophil']:
            n = subtype_counts.get(key, 0)
            pct = f'{n/wbc_total*100:.1f}%'
            count_data.insert(-1, [f'  ↳ {key.capitalize()}', str(n), pct + ' of WBC', ''])

    tbl = Table(count_data, colWidths=[6 * cm, 2.5 * cm, 2.5 * cm, 5 * cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3a6b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#eef2f7')]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#d0d8e8')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#aaaaaa')),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4 * cm))

    # --- Morphological summary ---
    story.append(Paragraph('Morphological Feature Summary', styles['Heading2']))

    # Gather features per class
    class_feats: dict = {}
    for b in boxes:
        cid = str(b.get('class', '?'))
        cname = _label(cid, class_names)
        for f in b.get('features', []):
            class_feats.setdefault(cname, {'area': [], 'circularity': [], 'nc_ratio': []})
            if f.get('area', 0) > 0:
                class_feats[cname]['area'].append(f['area'])
            if 'circularity' in f:
                class_feats[cname]['circularity'].append(f['circularity'])
            if 'nc_ratio' in f:
                class_feats[cname]['nc_ratio'].append(f['nc_ratio'])

    if class_feats:
        morph_data = [['Cell Type', 'Mean Area (px)', 'Mean Circularity', 'Mean N/C Ratio']]
        for cname, feats in class_feats.items():
            mean_area = f"{sum(feats['area'])/len(feats['area']):.1f}" if feats['area'] else '—'
            mean_circ = f"{sum(feats['circularity'])/len(feats['circularity']):.3f}" if feats['circularity'] else '—'
            mean_nc = f"{sum(feats['nc_ratio'])/len(feats['nc_ratio']):.3f}" if feats['nc_ratio'] else '—'
            morph_data.append([cname, mean_area, mean_circ, mean_nc])

        mtbl = Table(morph_data, colWidths=[6 * cm, 3 * cm, 3.5 * cm, 3.5 * cm])
        mtbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e7d32')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f5e9')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#aaaaaa')),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(mtbl)
    else:
        story.append(Paragraph(
            'No morphological features were extracted (no segmented regions found).',
            styles['Normal']
        ))
    story.append(Spacer(1, 0.4 * cm))

    # --- Disorder Detection ---
    story.append(Paragraph('Hematological Disorder Detection', styles['Heading2']))

    if disorders:
        flag_style = ParagraphStyle(
            'Flag', parent=styles['Normal'],
            textColor=colors.HexColor('#c62828'),
            fontName='Helvetica-Bold',
            fontSize=10,
            spaceAfter=4,
        )
        story.append(Paragraph('⚠ Abnormality Flags Detected:', flag_style))
        for d in disorders:
            story.append(Paragraph(f'• {d}', styles['Normal']))
    else:
        ok_style = ParagraphStyle(
            'OK', parent=styles['Normal'],
            textColor=colors.HexColor('#2e7d32'),
            fontName='Helvetica-Bold',
            fontSize=10,
        )
        story.append(Paragraph('✓ No hematological disorders detected.', ok_style))

    story.append(Spacer(1, 0.6 * cm))

    # --- Disclaimer ---
    disc_style = ParagraphStyle(
        'Disc', parent=styles['Normal'], fontSize=7,
        textColor=colors.HexColor('#666666'), fontName='Helvetica-Oblique'
    )
    story.append(Paragraph(
        'DISCLAIMER: This report is generated by an automated AI system for research and '
        'educational purposes only. It is not a substitute for professional medical diagnosis. '
        'All findings must be reviewed and confirmed by a qualified hematologist.',
        disc_style
    ))

    doc.build(story)


_CSV_FEATURE_KEYS = [
    'area', 'perimeter', 'circularity', 'eccentricity', 'convexity',
    'nc_ratio', 'nucleus_area', 'nuclear_irregularity',
    'nuclear_compactness', 'nuclear_solidity',
    'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
    'glcm_energy', 'glcm_correlation', 'glcm_entropy',
    'hue_mean', 'hue_std', 'saturation_mean', 'saturation_std',
    'value_mean', 'value_std',
]


def generate_csv(summary_path, out_csv):
    """Export per-cell detection results as CSV for LIS integration (proposal §4.8)."""
    summary_path = Path(summary_path)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    boxes = summary.get('boxes', [])
    disorders = summary.get('disorders', [])

    fieldnames = [
        'cell_index', 'class_id', 'class_name', 'effective_name',
        'detection_conf', 'subtype', 'subtype_conf',
        'x1', 'y1', 'x2', 'y2',
    ] + _CSV_FEATURE_KEYS + ['disorders']

    disorder_str = '; '.join(disorders) if disorders else 'None'

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, b in enumerate(boxes):
            xyxy = b.get('xyxy', [None, None, None, None])
            feats = b.get('features', [{}])
            feat = feats[0] if feats else {}
            row = {
                'cell_index': idx,
                'class_id': b.get('class', ''),
                'class_name': b.get('class_name', ''),
                'effective_name': b.get('effective_name', ''),
                'detection_conf': round(float(b.get('conf', 0)), 4),
                'subtype': b.get('subtype', ''),
                'subtype_conf': b.get('subtype_conf', ''),
                'x1': round(xyxy[0], 1) if xyxy[0] is not None else '',
                'y1': round(xyxy[1], 1) if xyxy[1] is not None else '',
                'x2': round(xyxy[2], 1) if xyxy[2] is not None else '',
                'y2': round(xyxy[3], 1) if xyxy[3] is not None else '',
                'disorders': disorder_str,
            }
            for key in _CSV_FEATURE_KEYS:
                v = feat.get(key, '')
                row[key] = round(float(v), 4) if isinstance(v, (int, float)) else v
            writer.writerow(row)


def main():
    args = parse_args()
    generate(Path(args.image), Path(args.summary), Path(args.out))
    print(f'Generated report -> {args.out}')
    csv_out = Path(args.out).with_suffix('.csv')
    generate_csv(Path(args.summary), csv_out)
    print(f'Generated CSV    -> {csv_out}')


if __name__ == '__main__':
    main()
