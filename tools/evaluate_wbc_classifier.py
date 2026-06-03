"""Evaluate the WBC subtype classifier on the held-out TEST split.

Prints per-class precision, recall, F1 and overall accuracy.
Optionally saves a confusion matrix image.

Usage:
  python tools/evaluate_wbc_classifier.py
  python tools/evaluate_wbc_classifier.py --n 200 --out reports/eval_charts/
"""
import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.wbc_classifier import get_classifier

CLASSES   = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']
FOLDERS   = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
TEST_BASE = Path('data/raw/blood_cells/dataset2-master/dataset2-master/images/TEST')


def evaluate(n_per_class: int = 200):
    cls = get_classifier()
    if not cls.available:
        print('WBC classifier not available. Train first with tools/train_wbc_classifier.py')
        return

    n = len(CLASSES)
    matrix = np.zeros((n, n), dtype=int)

    for fi, (folder, true_key) in enumerate(zip(FOLDERS, CLASSES)):
        true_idx = CLASSES.index(true_key)
        folder_path = TEST_BASE / folder
        if not folder_path.exists():
            print(f'  [SKIP] {folder_path} not found')
            continue
        files = list(folder_path.glob('*.jpeg'))[:n_per_class]
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            pred = cls.classify(img)
            pk = pred['subtype_key']
            if pk in CLASSES:
                matrix[true_idx][CLASSES.index(pk)] += 1

    # Print confusion matrix
    labels = [c.capitalize() for c in CLASSES]
    col_w = 14
    print('\nConfusion Matrix (rows=true, cols=predicted):')
    print(' ' * col_w + ''.join(f'{l:>{col_w}}' for l in labels))
    for i, lbl in enumerate(labels):
        row = ''.join(f'{matrix[i,j]:>{col_w}}' for j in range(n))
        print(f'{lbl:>{col_w}}{row}')

    # Per-class metrics
    print('\nPer-Class Metrics:')
    print(f'{"Class":<14} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Support":>10}')
    print('-' * 55)
    for i, lbl in enumerate(labels):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        prec   = tp / (tp + fp + 1e-8)
        rec    = tp / (tp + fn + 1e-8)
        f1     = 2 * prec * rec / (prec + rec + 1e-8)
        support = matrix[i, :].sum()
        print(f'{lbl:<14} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {support:>10}')

    overall_acc = np.trace(matrix) / (matrix.sum() + 1e-8)
    print('-' * 55)
    print(f'{"Overall Accuracy":<14} {overall_acc:>10.3f}   (total={matrix.sum()})')
    return matrix, overall_acc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=200, help='Samples per class')
    p.add_argument('--out', default=None, help='Save confusion matrix chart to this directory')
    return p.parse_args()


def main():
    args = parse_args()
    result = evaluate(n_per_class=args.n)
    if result and args.out:
        matrix, acc = result
        from tools.plot_evaluation import plot_wbc_confusion_matrix
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_wbc_confusion_matrix(out_dir, n_per_class=args.n)


if __name__ == '__main__':
    main()
