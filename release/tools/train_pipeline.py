"""Train pipeline wrapper to run multi-stage training as described in the proposal.

Stage 1: detection on BCCD
Stage 2: fine-tune classification on single-cell dataset (if provided)
Stage 3: combined fine-tune

This uses Ultralytics YOLO API.
"""
import argparse
import sys
from pathlib import Path

# ensure repo root is on sys.path so local packages are importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description='Run multi-stage training for YOLOv8')
    p.add_argument('--data', required=True, help='YOLO data yaml for stage1 (e.g., data/bccd.yaml)')
    p.add_argument('--weights', default='yolov8n.pt', help='initial weights (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt)')
    p.add_argument('--imgsz', type=int, default=640, help='image size for training and inference')
    p.add_argument('--batch', type=int, default=8, help='batch size')
    p.add_argument('--lr', type=float, default=0.01, help='learning rate')
    p.add_argument('--hyp', default=None, help='path to hyp.yaml for optimizer/loss hyperparameters')
    p.add_argument('--augment', action='store_true', help='Enable Ultralytics built-in augmentation during training')
    p.add_argument('--epochs1', type=int, default=150, help='epochs stage1')
    p.add_argument('--epochs2', type=int, default=100, help='epochs stage2 (fine-tune)')
    p.add_argument('--epochs3', type=int, default=50, help='epochs stage3 (combined)')
    p.add_argument('--name', default='blood_cell_model', help='run name prefix')
    return p.parse_args()


def main():
    args = parse_args()

    # Stage 1: detection on BCCD
    print('Stage 1: training detection on', args.data)
    model = YOLO(args.weights)
    # Ultralytics expects lr0 (initial LR) and lrf (final LR factor) rather than `lr`.
    train_kwargs = dict(data=args.data, epochs=args.epochs1, imgsz=args.imgsz, batch=args.batch,
                        lr0=args.lr, augment=args.augment, name=args.name + '_stage1')
    # If a hyp YAML is provided, load it and merge allowed keys into training kwargs
    if args.hyp:
        import yaml
        hyp_p = Path(args.hyp)
        if hyp_p.exists():
            try:
                with open(hyp_p, 'r', encoding='utf-8') as hf:
                    hypd = yaml.safe_load(hf)
            except Exception:
                hypd = None
            if isinstance(hypd, dict):
                # merge hyp keys into kwargs (Ultralytics accepts many of these as train args)
                train_kwargs.update(hypd)
    model.train(**train_kwargs)

    # use best weights from stage1
    stage1_weights = f'runs/train/{args.name}_stage1/weights/best.pt'

    # Stage 2: classification fine-tune (if dataset provided by user, this assumes a data yaml)
    # For now we re-use stage1 weights and run a short fine-tune to simulate
    print('Stage 2: fine-tuning (optional)')
    model2 = YOLO(stage1_weights)
    train_kwargs2 = dict(data=args.data, epochs=args.epochs2, imgsz=args.imgsz, batch=args.batch,
                         lr0=args.lr / 10.0, augment=args.augment, name=args.name + '_stage2')
    if args.hyp and 'hypd' in locals() and isinstance(hypd, dict):
        train_kwargs2.update(hypd)
    model2.train(**train_kwargs2)

    # Stage 3: combined fine-tune
    stage2_weights = f'runs/train/{args.name}_stage2/weights/best.pt'
    print('Stage 3: combined fine-tuning')
    model3 = YOLO(stage2_weights)
    train_kwargs3 = dict(data=args.data, epochs=args.epochs3, imgsz=args.imgsz, batch=args.batch,
                         lr0=args.lr / 20.0, augment=args.augment, name=args.name + '_stage3')
    if args.hyp and 'hypd' in locals() and isinstance(hypd, dict):
        train_kwargs3.update(hypd)
    model3.train(**train_kwargs3)


if __name__ == '__main__':
    main()
