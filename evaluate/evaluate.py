"""Evaluate trained model — mAP, precision, recall, confusion matrix."""
from ultralytics import YOLO

model = YOLO('runs/detect/stage3_combined/weights/best.pt')

metrics = model.val(data='data/bccd.yaml', split='val')

print(f"\n=== EVALUATION RESULTS ===")
print(f"mAP@0.5      : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
print(f"Precision    : {metrics.box.mp:.4f}")
print(f"Recall       : {metrics.box.mr:.4f}")
print(f"\nResults saved to runs/detect/val*/")