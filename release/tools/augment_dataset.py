#!/usr/bin/env python3
"""Simple YOLO-format dataset augmenter using Albumentations.

Reads images and labels from an input YOLO dataset structure and writes an
augmented copy to an output folder, preserving YOLO label format.

Usage:
    python tools/augment_dataset.py --images data/yolo_dataset/images/train --labels data/yolo_dataset/labels/train --out data/yolo_dataset_aug --n 5
"""
import argparse
from pathlib import Path
import os
import cv2
import albumentations as A


def parse_args():
    p = argparse.ArgumentParser(description='Augment YOLO-format dataset')
    p.add_argument('--images', required=True, help='Input images folder')
    p.add_argument('--labels', required=True, help='Input labels folder (YOLO .txt)')
    p.add_argument('--out', required=True, help='Output dataset root (will create images/ and labels/)')
    p.add_argument('--n', type=int, default=3, help='Number of augmented copies per image')
    p.add_argument('--min_visibility', type=float, default=0.3, help='Minimum bbox visibility after transform')
    return p.parse_args()


def read_yolo_labels(label_path):
    """Return list of bboxes in YOLO format: (class_id, x_center, y_center, w, h)"""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            vals = list(map(float, parts[1:5]))
            boxes.append((cls, *vals))
    return boxes


def write_yolo_labels(label_path, boxes):
    with open(label_path, 'w') as f:
        for (cls, x, y, w, h) in boxes:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def main():
    args = parse_args()
    img_folder = Path(args.images)
    lbl_folder = Path(args.labels)
    out_root = Path(args.out)
    out_img = out_root / 'images' / 'train'
    out_lbl = out_root / 'labels' / 'train'
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # Compose augmentation pipeline
    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=args.min_visibility))

    img_paths = sorted([p for p in img_folder.glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')])
    if len(img_paths) == 0:
        print('No images found in', img_folder)
        return

    print(f'Found {len(img_paths)} images. Augmenting {args.n} copies each...')

    counter = 0
    for img_p in img_paths:
        stem = img_p.stem
        lbl_p = lbl_folder / (stem + '.txt')

        boxes = read_yolo_labels(lbl_p)
        # Prepare albumentations bboxes and class_labels arrays
        bboxes = []
        class_labels = []
        for (cls, x, y, w, h) in boxes:
            bboxes.append((x, y, w, h))
            class_labels.append(cls)

        # Copy original to output (as one baseline)
        img = cv2.imread(str(img_p))
        out_img_p = out_img / (stem + '.jpg')
        cv2.imwrite(str(out_img_p), img)
        out_lbl_p = out_lbl / (stem + '.txt')
        write_yolo_labels(out_lbl_p, boxes)
        counter += 1

        # Generate augmented copies
        for i in range(args.n):
            if len(bboxes) == 0:
                # no bboxes, just apply color transforms
                aug_img = A.Compose([A.RandomBrightnessContrast(p=0.5), A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)])
                res = aug_img(image=img)
                aug_im = res['image']
                aug_name = f"{stem}_aug{i}.jpg"
                cv2.imwrite(str(out_img / aug_name), aug_im)
                # write empty label
                (out_lbl / (stem + f"_aug{i}.txt")).write_text('')
                counter += 1
                continue

            try:
                transformed = aug(image=img, bboxes=bboxes, class_labels=class_labels)
            except Exception as e:
                print(f'Augmentation failed for {img_p} iteration {i}:', e)
                continue

            new_img = transformed['image']
            new_bboxes = transformed.get('bboxes', [])
            new_labels = transformed.get('class_labels', [])

            if len(new_bboxes) == 0:
                # discard augment that removed all boxes
                continue

            out_name = f"{stem}_aug{i}.jpg"
            cv2.imwrite(str(out_img / out_name), new_img)

            # combine new labels into YOLO text
            new_boxes = []
            for lab, bb in zip(new_labels, new_bboxes):
                x, y, w, h = bb
                new_boxes.append((lab, x, y, w, h))

            write_yolo_labels(out_lbl / (stem + f"_aug{i}.txt"), new_boxes)
            counter += 1

    print(f'Wrote {counter} images to {out_img}')


if __name__ == '__main__':
    main()
