"""Generate evaluation charts: confusion matrix, PR curves, and training accuracy plots.

Usage:
  python tools/plot_evaluation.py --out reports/eval_charts/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


# ---------------------------------------------------------------------------
# 1. YOLO detection PR / accuracy chart from results.csv
# ---------------------------------------------------------------------------

def plot_yolo_training(results_csv: Path, out_dir: Path):
    """Plot YOLO training metrics from Ultralytics results.csv."""
    import csv
    if not results_csv.exists():
        print(f'  [SKIP] {results_csv} not found')
        return

    raw_rows = list(csv.DictReader(open(results_csv)))
    # Strip whitespace from all keys
    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in raw_rows]
    epochs = [int(r['epoch']) for r in rows]

    def get(rows, key):
        candidates = [k for k in rows[0].keys() if key in k]
        if not candidates:
            return []
        k = candidates[0]
        return [float(r[k]) for r in rows]

    box_map50 = get(rows, 'mAP50(B)')
    precision  = get(rows, 'metrics/precision(B)')
    recall     = get(rows, 'metrics/recall(B)')
    train_loss = get(rows, 'train/box_loss')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('YOLOv8 Detection Training Metrics', fontsize=13, fontweight='bold')

    if box_map50:
        axes[0].plot(epochs, box_map50, color='#1a3a6b', linewidth=2)
        axes[0].axhline(max(box_map50), color='orange', linestyle='--', alpha=0.7,
                        label=f'Best: {max(box_map50):.3f}')
        axes[0].set_title('mAP@0.5')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

    if precision and recall:
        axes[1].plot(epochs, precision, color='green', linewidth=2, label='Precision')
        axes[1].plot(epochs, recall, color='red', linewidth=2, label='Recall')
        axes[1].set_title('Precision & Recall')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

    if train_loss:
        axes[2].plot(epochs, train_loss, color='purple', linewidth=2)
        axes[2].set_title('Training Box Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / 'yolo_training_metrics.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# 2. WBC subtype classifier accuracy chart
# ---------------------------------------------------------------------------

def plot_classifier_training(results_csv: Path, out_dir: Path):
    import csv
    if not results_csv.exists():
        print(f'  [SKIP] {results_csv} not found')
        return

    rows = list(csv.DictReader(open(results_csv)))
    key_map = {k.strip(): k for k in rows[0].keys()}

    epochs = [int(r[key_map.get('epoch', 'epoch')]) for r in rows]
    top1   = [float(r[key_map.get('metrics/accuracy_top1', 'metrics/accuracy_top1')]) for r in rows]
    loss   = [float(r[key_map.get('val/loss', 'val/loss')]) for r in rows]
    tloss  = [float(r[key_map.get('train/loss', 'train/loss')]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('WBC Subtype Classifier Training (YOLOv8n-cls)', fontsize=13, fontweight='bold')

    axes[0].plot(epochs, top1, color='#1a3a6b', linewidth=2, marker='o', markersize=4)
    axes[0].axhline(max(top1), color='orange', linestyle='--', alpha=0.7,
                    label=f'Best: {max(top1)*100:.1f}%')
    axes[0].set_title('Top-1 Accuracy (val)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, tloss, color='green', linewidth=2, label='Train loss')
    axes[1].plot(epochs, loss,  color='red',   linewidth=2, label='Val loss')
    axes[1].set_title('Train vs Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / 'wbc_classifier_training.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# 3. WBC subtype confusion matrix on test set
# ---------------------------------------------------------------------------

def plot_wbc_confusion_matrix(out_dir: Path, n_per_class: int = 100):
    import cv2, os
    from src.wbc_classifier import get_classifier

    cls = get_classifier()
    if not cls.available:
        print('  [SKIP] WBC classifier not available')
        return

    base = Path('data/raw/blood_cells/dataset2-master/dataset2-master/images/TEST')
    if not base.exists():
        print(f'  [SKIP] Test data not found: {base}')
        return

    CLASS_MAP = {'EOSINOPHIL': 'eosinophil', 'LYMPHOCYTE': 'lymphocyte',
                 'MONOCYTE': 'monocyte',     'NEUTROPHIL': 'neutrophil'}
    ordered   = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']
    labels    = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
    n_cls     = len(ordered)
    matrix    = np.zeros((n_cls, n_cls), dtype=int)

    for folder, true_key in CLASS_MAP.items():
        true_idx = ordered.index(true_key)
        folder_path = base / folder
        files = list(folder_path.glob('*.jpeg'))[:n_per_class]
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            pred = cls.classify(img)
            pred_key = pred['subtype_key']
            if pred_key in ordered:
                matrix[true_idx][ordered.index(pred_key)] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'WBC Subtype Confusion Matrix\n(n={n_per_class}/class)', fontsize=12, fontweight='bold')

    total = matrix.sum(axis=1, keepdims=True)
    for i in range(n_cls):
        for j in range(n_cls):
            pct = matrix[i, j] / (total[i, 0] + 1e-8) * 100
            color = 'white' if matrix[i, j] > matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{matrix[i,j]}\n({pct:.0f}%)', ha='center', va='center',
                    fontsize=9, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    overall_acc = np.trace(matrix) / (matrix.sum() + 1e-8)
    ax.set_title(f'WBC Subtype Confusion Matrix  |  Accuracy: {overall_acc*100:.1f}%',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = out_dir / 'wbc_confusion_matrix.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}  (overall acc: {overall_acc*100:.1f}%)')


# ---------------------------------------------------------------------------
# 4. YOLO per-class bar chart from val results
# ---------------------------------------------------------------------------

def plot_detection_summary(out_dir: Path):
    """Bar chart of per-class mAP and precision from YOLO val."""
    classes   = ['WBC', 'RBC', 'Platelets']
    map50     = [0.805, 0.929, 0.692]
    precision = [0.722, 0.971, 0.628]
    recall    = [0.762, 0.762, 0.735]

    x = np.arange(len(classes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, map50,     w, label='mAP@0.5',  color='#1a3a6b', alpha=0.85)
    ax.bar(x,     precision, w, label='Precision', color='#2e7d32', alpha=0.85)
    ax.bar(x + w, recall,    w, label='Recall',    color='#c62828', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.92, color='orange', linestyle='--', linewidth=1.5, label='Target 92%')
    ax.set_title('YOLOv8 Detection Performance per Class (val set)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    for rect_group in ax.patches:
        h = rect_group.get_height()
        ax.annotate(f'{h:.2f}',
                    xy=(rect_group.get_x() + rect_group.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = out_dir / 'detection_summary.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='reports/eval_charts', help='Output directory')
    p.add_argument('--yolo-csv', default='runs/detect/test_yolo_dataset_stage12/results.csv')
    p.add_argument('--cls-csv',  default='runs/classify/runs/classify/wbc_subtype_cls/results.csv')
    p.add_argument('--n', type=int, default=100, help='Samples per class for confusion matrix')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Generating evaluation charts -> {out_dir}/')

    plot_yolo_training(Path(args.yolo_csv), out_dir)
    plot_classifier_training(Path(args.cls_csv), out_dir)
    plot_detection_summary(out_dir)
    plot_wbc_confusion_matrix(out_dir, n_per_class=args.n)

    print('\nAll charts saved.')


if __name__ == '__main__':
    main()
