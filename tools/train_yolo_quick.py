from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy

    model.train(
        data="data/bccd.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        name="blood_cell_model"
    )

if __name__ == "__main__":
    main()