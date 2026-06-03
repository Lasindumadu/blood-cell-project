"""Stage 3: Fine-tune on combined dataset — 50 epochs, very low LR."""
from ultralytics import YOLO

model = YOLO('runs/detect/stage2_bloodcells/weights/best.pt')

results = model.train(
    data='data/combined.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    optimizer='Adam',
    lr0=0.0001,
    cos_lr=True,
    fl_gamma=2.0,
    patience=50,
    project='runs/detect',
    name='stage3_combined',
    exist_ok=True
)
print("Stage 3 FINAL done. Weights: runs/detect/stage3_combined/weights/best.pt")