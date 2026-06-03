from ultralytics import YOLO

model = YOLO("runs/detect/test_yolo_dataset_stage12/weights/best.pt")

results = model("data/yolo_dataset/images/test/BloodImage_00007.jpeg", save=True)


print(results)
