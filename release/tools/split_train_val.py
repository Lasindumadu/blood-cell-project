#!/usr/bin/env python3
"""Split a YOLO-format dataset's train folder into train+val (copy mode).

This script copies a fraction of images and their corresponding label files
from images/train + labels/train into images/val + labels/val. It is safe to
run multiple times (will create folders if missing).

Usage:
  python tools/split_train_val.py --images data/yolo_dataset_aug/images --labels data/yolo_dataset_aug/labels --val-frac 0.2
"""
import argparse
from pathlib import Path
import random
import shutil


def parse_args():
    p = argparse.ArgumentParser(description='Split YOLO train into train+val (copy)')
    p.add_argument('--images', required=True, help='Root images folder (contains train/ and optionally val/)')
    p.add_argument('--labels', required=True, help='Root labels folder (contains train/ and optionally val/)')
    p.add_argument('--val-frac', type=float, default=0.2, help='Fraction of train to copy to val')
    p.add_argument('--seed', type=int, default=0, help='Random seed')
    p.add_argument('--mode', choices=['copy', 'move'], default='copy', help='Whether to copy or move files')
    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    random.seed(args.seed)

    img_root = Path(args.images)
    lbl_root = Path(args.labels)

    train_img = img_root / 'train'
    val_img = img_root / 'val'
    train_lbl = lbl_root / 'train'
    val_lbl = lbl_root / 'val'

    if not train_img.exists():
        print('Train images folder not found:', train_img)
        return
    ensure_dir(val_img)
    ensure_dir(val_lbl)

    img_files = [p for p in train_img.glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')]
    n = len(img_files)
    if n == 0:
        print('No images found in', train_img)
        return

    k = max(1, int(n * args.val_frac))
    selected = set(random.sample([str(p.name) for p in img_files], k))

    copied = 0
    for img_name in selected:
        src_img = train_img / img_name
        dst_img = val_img / img_name
        src_lbl = train_lbl / (Path(img_name).stem + '.txt')
        dst_lbl = val_lbl / (Path(img_name).stem + '.txt')

        try:
            if args.mode == 'copy':
                shutil.copy2(src_img, dst_img)
            else:
                shutil.move(src_img, dst_img)
        except Exception as e:
            print('Failed to copy/move image', src_img, e)
            continue

        # copy label if exists, else create empty
        if src_lbl.exists():
            try:
                if args.mode == 'copy':
                    shutil.copy2(src_lbl, dst_lbl)
                else:
                    shutil.move(src_lbl, dst_lbl)
            except Exception as e:
                print('Failed to copy/move label', src_lbl, e)
        else:
            dst_lbl.write_text('')

        copied += 1

    print(f'Copied {copied} images to {val_img} and labels to {val_lbl}')


if __name__ == '__main__':
    main()
