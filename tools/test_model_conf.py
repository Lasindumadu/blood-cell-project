from ultralytics import YOLO
import sys

models = [
    'runs/detect/blood_cell_model2/weights/best.pt',
    'runs/detect/test_yolo_dataset_stage1/weights/best.pt',
    'runs/detect/test_yolo_dataset_stage12/weights/best.pt',
    'runs/detect/test_yolo_dataset_stage2/weights/best.pt',
]

img = 'data/yolo_dataset/images/test/BloodImage_00007.jpeg'

for m in models:
    try:
        model = YOLO(m)
        r = model(img, conf=0.25, verbose=False)[0]
        n = len(r.boxes) if r.boxes is not None else 0
        confs = [float(b.conf[0]) for b in r.boxes] if n > 0 else []
        avg = sum(confs)/len(confs) if confs else 0
        mx = max(confs) if confs else 0
        print(f'{m.split("/")[2]:30s}: {n:3d} boxes, avg_conf={avg:.3f}, max_conf={mx:.3f}')
    except Exception as e:
        print(f'{m.split("/")[2]:30s}: ERROR - {e}')
