# """Generate a simple PDF clinical-style report from an annotated image and the summary JSON.

# Usage:
#   python tools/generate_report.py --image results/BloodImage_00007_annotated.jpg --summary results/BloodImage_00007_summary.json --out reports/report.pdf
# """
# import argparse
# from pathlib import Path
# from reportlab.lib.pagesizes import A4
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import ImageReader
# import json


# def parse_args():
#     p = argparse.ArgumentParser(description='Generate PDF report from annotated image and summary JSON')
#     p.add_argument('--image', required=True, help='Annotated image path')
#     p.add_argument('--summary', required=True, help='Summary JSON path')
#     p.add_argument('--out', required=True, help='Output PDF path')
#     return p.parse_args()


# def generate(image_path, summary_path, out_pdf):
#     img = ImageReader(str(image_path))
#     with open(summary_path, 'r') as f:
#         summary = json.load(f)

#     c = canvas.Canvas(str(out_pdf), pagesize=A4)
#     w, h = A4

#     # Title
#     c.setFont('Helvetica-Bold', 16)
#     c.drawString(40, h - 50, 'Automated Blood Cell Analysis Report')

#     # Insert image (fit to width)
#     img_w = w - 80
#     img_h = img_w * 0.6
#     c.drawImage(img, 40, h - 60 - img_h, width=img_w, height=img_h)

#     # Summary text
#     text_y = h - 80 - img_h
#     c.setFont('Helvetica', 10)
#     c.drawString(40, text_y, 'Counts:')
#     text_y -= 14
#     for k, v in summary.get('counts', {}).items():
#         c.drawString(60, text_y, f'Class {k}: {v}')
#         text_y -= 12

#     text_y -= 6
#     c.drawString(40, text_y, 'Detected Disorders:')
#     text_y -= 14
#     disorders = summary.get('disorders', [])
#     if disorders:
#         for d in disorders:
#             c.drawString(60, text_y, f'- {d}')
#             text_y -= 12
#     else:
#         c.drawString(60, text_y, 'None')
#         text_y -= 12

#     c.showPage()
#     c.save()


# def main():
#     args = parse_args()
#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     generate(Path(args.image), Path(args.summary), out_path)
#     print(f'Generated report -> {out_path}')


# if __name__ == '__main__':
#     main()

"""
Step 4.8 — Automated Clinical Report Generation
Generates PDF using ReportLab.
"""
from pathlib import Path
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm


def generate(annotated_image_path, summary_json_path, output_pdf_path):
    annotated_image_path = Path(annotated_image_path)
    summary_json_path = Path(summary_json_path)
    output_pdf_path = Path(output_pdf_path)

    with open(summary_json_path) as f:
        summary = json.load(f)

    doc = SimpleDocTemplate(str(output_pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Automated Blood Cell Analysis Report", styles['Title']))
    story.append(Paragraph("University of Ruhuna — EE7204/EC7205", styles['Normal']))
    story.append(Spacer(1, 0.5 * cm))

    # Annotated Image
    if annotated_image_path.exists():
        story.append(RLImage(str(annotated_image_path), width=14 * cm, height=10 * cm))
        story.append(Spacer(1, 0.3 * cm))

    # Cell Count Table (CBC format)
    story.append(Paragraph("Complete Blood Count (CBC)", styles['Heading2']))
    counts = summary.get('cell_counts', {})
    table_data = [['Cell Type', 'Count', 'Status']]
    normal_ranges = {
        'RBC': (4, 10, 'million/µL'),
        'neutrophil': (40, 70, '% WBC'),
        'lymphocyte': (20, 40, '% WBC'),
        'monocyte': (2, 8, '% WBC'),
        'eosinophil': (1, 4, '% WBC'),
        'platelet': (5, 50, 'per field')
    }
    for cell, cnt in counts.items():
        lo, hi, unit = normal_ranges.get(cell, (0, 999, ''))
        status = 'Normal' if lo <= cnt <= hi else ('↑ High' if cnt > hi else '↓ Low')
        table_data.append([cell.capitalize(), str(cnt), status])

    t = Table(table_data, colWidths=[6 * cm, 4 * cm, 4 * cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    # Disorder Detection Section
    story.append(Paragraph("Hematological Disorder Detection", styles['Heading2']))
    disorders = summary.get('disorders', {})

    all_info = disorders.get('acute_lymphoblastic_leukemia', {})
    all_detected = all_info.get('detected', False)
    status_style = ParagraphStyle('status', parent=styles['Normal'],
                                   textColor=colors.red if all_detected else colors.green)
    story.append(Paragraph(
        f"Acute Lymphoblastic Leukemia (ALL): {'⚠ FLAGGED' if all_detected else '✓ Not detected'}",
        status_style))
    if all_info.get('reasons'):
        for r in all_info['reasons']:
            story.append(Paragraph(f"  • {r}", styles['Normal']))
    story.append(Spacer(1, 0.2 * cm))

    sickle = disorders.get('sickle_cell_disease', {})
    story.append(Paragraph(
        f"Sickle Cell Disease: {'⚠ FLAGGED' if sickle.get('detected') else '✓ Not detected'}",
        status_style))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "⚠ This report is for research/educational purposes. Clinical validation required.",
        styles['Italic']))

    doc.build(story)
    print(f"PDF report saved: {output_pdf_path}")