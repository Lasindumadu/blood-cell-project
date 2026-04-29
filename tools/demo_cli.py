#!/usr/bin/env python3
"""Simple demo CLI: run YOLOv8 inference, save annotated image and JSON summary."""
import argparse
import json
from pathlib import Path
import os

import numpy as np

# ensure repo root is on sys.path so `src` package is importable when running from tools/
import sys
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.pipeline import analyze_image, save_report


def parse_args():
    p = argparse.ArgumentParser(description="Run full analysis on one image and save results")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--model", default="runs/detect/test_yolo_dataset_stage12/weights/best.pt", help="Path to model weights")

    p.add_argument("--out", default="results", help="Output folder")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--include-all-contours", action="store_true", help="Include all contour features per detection (default: only largest)")
    return p.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = analyze_image(str(img_path), model_path=args.model, conf=args.conf, include_all_contours=args.include_all_contours)

    # save annotated and summary
    save_report(result, str(out_dir), img_path.stem)

    print(f"Saved annotated image and summary for {img_path.name} in {out_dir}")


if __name__ == '__main__':
    main()
