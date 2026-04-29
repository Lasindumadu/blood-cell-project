"""Evaluation wrapper: run Ultralytics model.val on a dataset and save results to CSV."""
import argparse
import csv
import sys
from pathlib import Path

# ensure repo root is on sys.path so local packages are importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLO model on dataset")
    p.add_argument("--model", required=True, help="Path to model weights")
    p.add_argument("--data", required=True, help="Ultralytics data yaml (e.g., data/bccd.yaml)")
    p.add_argument("--out", default="runs/val_results.csv", help="CSV output path")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    print(f"Running validation on {args.data} ...")
    metrics = model.val(data=args.data)
    # `metrics` is often a dict-like or printed object; try to save summary
    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Try to include class_name when model provides a names mapping
        names = getattr(model, 'names', None)

        # If metrics is a dict-like, attempt to flatten
        try:
            items = dict(metrics).items()
        except Exception:
            # Fallback: write a single result cell
            writer.writerow(['result', str(metrics)])
        else:
            # For each metric, if its value is itself a mapping of class->value,
            # write rows with metric, class_id, class_name (when available), value.
            writer.writerow(['metric', 'class_id', 'class_name', 'value'])
            for k, v in items:
                # If v is mapping-like (per-class)
                if isinstance(v, dict):
                    for class_k, val_k in v.items():
                        try:
                            class_id = int(class_k)
                        except Exception:
                            class_id = class_k
                        class_name = names.get(class_id, '') if names is not None and isinstance(class_id, int) else ''
                        writer.writerow([k, class_id, class_name, val_k])
                else:
                    # scalar metric
                    writer.writerow([k, '', '', v])

    print(f"Saved validation summary to {args.out}")


if __name__ == '__main__':
    main()
