#!/usr/bin/env python3
"""Demo CLI: run the full two-stage analysis pipeline on a single image."""
import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.pipeline import analyze_image, save_report
from src.wbc_classifier import DEFAULT_CLASSIFIER_PATH


def parse_args():
    p = argparse.ArgumentParser(description="Run full analysis on one image and save results")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--model", default="runs/detect/test_yolo_dataset_stage12/weights/best.pt",
                   help="Path to YOLOv8 detection model weights")
    p.add_argument("--wbc-classifier", default=DEFAULT_CLASSIFIER_PATH,
                   help="Path to WBC subtype classifier weights (YOLOv8n-cls)")
    p.add_argument("--out", default="results", help="Output folder")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--include-all-contours", action="store_true",
                   help="Include all contour features per detection (default: only largest)")
    return p.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wbc_cls = args.wbc_classifier if Path(args.wbc_classifier).exists() else None

    result = analyze_image(
        str(img_path),
        model_path=args.model,
        conf=args.conf,
        include_all_contours=args.include_all_contours,
        wbc_classifier_path=wbc_cls,
    )

    save_report(result, str(out_dir), img_path.stem)

    summary = result['summary']
    counts = summary.get('counts', {})
    subtype_counts = summary.get('subtype_counts', {})
    disorders = summary.get('disorders', [])

    print(f"Saved results for {img_path.name} -> {out_dir}/")
    print(f"  Detections: {dict(counts)}")
    if subtype_counts:
        print(f"  WBC subtypes: {dict(subtype_counts)}")
    if disorders:
        for d in disorders:
            print(f"  [DISORDER] {d}")
    else:
        print("  No disorders detected.")


if __name__ == '__main__':
    main()
