"""
Converts BCCD dataset from Supervisely JSON format to YOLO txt format.
Run ONCE before Stage 1 training.
Your project already has this done — only run if you want to redo conversion.
"""
import json, os, shutil
from pathlib import Path

BCCD_CLASS_MAP = {'RBC': 0, 'WBC': 1, 'Platelets': 2}

def convert_split(split: str):
    base       = Path(f'data/raw/bccd/{split}')
    ann_dir    = base / 'ann'
    img_dir    = base / 'img'
    label_dir  = Path(f'data/yolo_dataset/labels/{split}')
    out_img    = Path(f'data/yolo_dataset/images/{split}')
    label_dir.mkdir(parents=True, exist_ok=True)
    out_img.mkdir(parents=True, exist_ok=True)

    for ann_file in ann_dir.glob('*.json'):
        with open(ann_file) as f:
            ann = json.load(f)
        H        = ann['size']['height']
        W        = ann['size']['width']
        img_name = ann_file.stem
        lines    = []
        for obj in ann['objects']:
            cls_name = obj['classTitle']
            if cls_name not in BCCD_CLASS_MAP:
                continue
            cls_id = BCCD_CLASS_MAP[cls_name]
            pts    = obj['points']['exterior']
            x1, y1 = pts[0]; x2, y2 = pts[1]
            xc = ((x1+x2)/2)/W; yc = ((y1+y2)/2)/H
            bw = abs(x2-x1)/W;  bh = abs(y2-y1)/H
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        lbl = label_dir / img_name.replace('.jpeg','.txt').replace('.jpg','.txt')
        with open(lbl, 'w') as f:
            f.write('\n'.join(lines))
        src = img_dir / img_name
        if src.exists():
            shutil.copy(src, out_img / img_name)
    print(f"  {split}: {len(list(ann_dir.glob('*.json')))} annotations converted")

if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        if Path(f'data/raw/bccd/{split}/ann').exists():
            convert_split(split)
    print("Done!")