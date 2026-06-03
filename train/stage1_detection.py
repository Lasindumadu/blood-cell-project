"""Stage 1: Train YOLOv8 on BCCD dataset — 150 epochs."""
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='data/bccd.yaml',
    epochs=150,
    batch=16,
    imgsz=640,
    optimizer='Adam',
    lr0=0.001,
    cos_lr=True,
    fl_gamma=2.0,
    patience=50,
    degrees=15,
    fliplr=0.5,
    flipud=0.5,
    hsv_v=0.2,
    project='runs/detect',
    name='stage1_bccd',
    exist_ok=True
)
print("Stage 1 done. Weights: runs/detect/stage1_bccd/weights/best.pt")