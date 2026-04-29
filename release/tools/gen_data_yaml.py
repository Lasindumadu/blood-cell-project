#!/usr/bin/env python3
"""Generate a YOLO data YAML file from an images/labels directory structure.

Example usage:
  python tools/gen_data_yaml.py --images data/yolo_dataset_aug/images --labels data/yolo_dataset_aug/labels --out data/bccd_aug.yaml --names data/bccd_names.txt

If --names is not provided, tries to copy names from data/bccd.yaml if it exists.
"""
import argparse
from pathlib import Path
import yaml


def parse_args():
    p = argparse.ArgumentParser(description='Generate YOLO data yaml from folders')
    p.add_argument('--images', required=True, help='Root images folder (should contain train/val/test)')
    p.add_argument('--labels', required=True, help='Root labels folder (should contain train/val/test)')
    p.add_argument('--out', required=True, help='Output yaml path')
    p.add_argument('--names', default=None, help='Optional names file or YAML mapping (text file with one class name per line)')
    return p.parse_args()


def load_names_from_file(names_path):
    p = Path(names_path)
    if not p.exists():
        return None
    # if it's a txt file with one name per line
    if p.suffix.lower() in ['.txt']:
        with open(p, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        return {i: name for i, name in enumerate(lines)}
    # if yaml
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                # If this YAML contains a top-level 'names' mapping (e.g., data/bccd.yaml), return that
                if 'names' in data and isinstance(data['names'], dict):
                    return data['names']
                return data
    except Exception:
        return None
    return None


def main():
    args = parse_args()
    img_root = Path(args.images)
    lbl_root = Path(args.labels)
    out_path = Path(args.out)

    # expect train/val (and optionally test) subfolders
    train_img = img_root / 'train'
    val_img = img_root / 'val'
    test_img = img_root / 'test'

    if not train_img.exists():
        print('Train folder not found at', train_img)
    if not val_img.exists():
        print('Val folder not found at', val_img)

    # data yaml expects paths relative to `path` field
    # We'll set path to the parent of images folder if images contains train/val directly; else use provided images root
    # Determine common path
    common_path = img_root.parent if (img_root / 'train').exists() else img_root

    data = {
        'path': str(common_path).replace('\\', '/'),
        'train': str(train_img.relative_to(common_path)).replace('\\', '/'),
        'val': str(val_img.relative_to(common_path)).replace('\\', '/') if val_img.exists() else '',
    }
    if test_img.exists():
        data['test'] = str(test_img.relative_to(common_path)).replace('\\', '/')

    # names
    names = None
    if args.names:
        names = load_names_from_file(args.names)
    else:
        # try to copy from data/bccd.yaml if present
        default_yaml = Path('data/bccd.yaml')
        if default_yaml.exists():
            try:
                with open(default_yaml, 'r', encoding='utf-8') as f:
                    existing = yaml.safe_load(f)
                    if 'names' in existing:
                        names = existing['names']
            except Exception:
                names = None

    if names is None:
        print('No names provided; using placeholder numeric names')
        # create placeholders assuming classes discovered from labels
        # attempt to find max class id from label files
        max_id = -1
        for txt in (lbl_root / 'train').glob('*.txt'):
            try:
                with open(txt, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 0:
                            continue
                        cid = int(parts[0])
                        if cid > max_id:
                            max_id = cid
            except Exception:
                continue
        names = {i: f'class_{i}' for i in range(max_id + 1)}

    data['names'] = names

    # write out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print('Wrote data yaml to', out_path)


if __name__ == '__main__':
    main()
