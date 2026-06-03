"""Prepare WBC subtype classification dataset for YOLOv8 classification training.

Reads from:
  data/raw/blood_cells/dataset2-master/dataset2-master/images/TRAIN/{CLASS}/
  data/raw/blood_cells/dataset2-master/dataset2-master/images/TEST/{CLASS}/

Writes to:
  data/wbc_cls/train/{class}/
  data/wbc_cls/val/{class}/

Usage:
  python tools/prepare_wbc_cls_dataset.py
"""
import shutil
from pathlib import Path

SRC = Path('data/raw/blood_cells/dataset2-master/dataset2-master/images')
DST = Path('data/wbc_cls')

CLASS_MAP = {
    'EOSINOPHIL': 'eosinophil',
    'LYMPHOCYTE': 'lymphocyte',
    'MONOCYTE': 'monocyte',
    'NEUTROPHIL': 'neutrophil',
}

SPLITS = [('TRAIN', 'train'), ('TEST', 'val')]


def main():
    total = 0
    for src_split, dst_split in SPLITS:
        for src_cls, dst_cls in CLASS_MAP.items():
            src_dir = SRC / src_split / src_cls
            dst_dir = DST / dst_split / dst_cls
            if not src_dir.exists():
                print(f'  [SKIP] {src_dir} not found')
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            files = list(src_dir.glob('*'))
            for f in files:
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
                    dest = dst_dir / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)
            count = len(list(dst_dir.glob('*')))
            print(f'  {dst_split}/{dst_cls}: {count} images')
            total += count

    print(f'\nDone. Total images prepared: {total}')
    print(f'Dataset at: {DST.resolve()}')


if __name__ == '__main__':
    main()
