"""Generate a simple PDF clinical-style report from an annotated image and the summary JSON.

Usage:
  python tools/generate_report.py --image results/BloodImage_00007_annotated.jpg --summary results/BloodImage_00007_summary.json --out reports/report.pdf
"""
import argparse
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import json


def parse_args():
    p = argparse.ArgumentParser(description='Generate PDF report from annotated image and summary JSON')
    p.add_argument('--image', required=True, help='Annotated image path')
    p.add_argument('--summary', required=True, help='Summary JSON path')
    p.add_argument('--out', required=True, help='Output PDF path')
    return p.parse_args()


def generate(image_path, summary_path, out_pdf):
    img = ImageReader(str(image_path))
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    w, h = A4

    # Title
    c.setFont('Helvetica-Bold', 16)
    c.drawString(40, h - 50, 'Automated Blood Cell Analysis Report')

    # Insert image (fit to width)
    img_w = w - 80
    img_h = img_w * 0.6
    c.drawImage(img, 40, h - 60 - img_h, width=img_w, height=img_h)

    # Summary text
    text_y = h - 80 - img_h
    c.setFont('Helvetica', 10)
    c.drawString(40, text_y, 'Counts:')
    text_y -= 14
    for k, v in summary.get('counts', {}).items():
        c.drawString(60, text_y, f'Class {k}: {v}')
        text_y -= 12

    text_y -= 6
    c.drawString(40, text_y, 'Detected Disorders:')
    text_y -= 14
    disorders = summary.get('disorders', [])
    if disorders:
        for d in disorders:
            c.drawString(60, text_y, f'- {d}')
            text_y -= 12
    else:
        c.drawString(60, text_y, 'None')
        text_y -= 12

    c.showPage()
    c.save()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generate(Path(args.image), Path(args.summary), out_path)
    print(f'Generated report -> {out_path}')


if __name__ == '__main__':
    main()
