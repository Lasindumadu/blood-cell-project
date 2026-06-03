"""Stage 2: Fine-tune on Blood Cell Images — 100 epochs, freeze backbone."""
from ultralytics import YOLO

model = YOLO('runs/detect/stage1_bccd/weights/best.pt')

results = model.train(
    data='data/blood_cells.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    optimizer='Adam',
    lr0=0.001,
    cos_lr=True,
    fl_gamma=2.0,
    patience=50,
    freeze=10,
    project='runs/detect',
    name='stage2_bloodcells',
    exist_ok=True
)
print("Stage 2 done. Weights: runs/detect/stage2_bloodcells/weights/best.pt")