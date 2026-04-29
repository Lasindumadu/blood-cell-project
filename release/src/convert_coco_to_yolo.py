import os
import json

classes = {
    "RBC": 0,
    "WBC": 1,
    "Platelets": 2
}

def convert_bbox(size, box):
    dw = 1.0 / size["width"]
    dh = 1.0 / size["height"]

    x_min, y_min = box[0]
    x_max, y_max = box[1]

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    return (x_center * dw, y_center * dh, w * dw, h * dh)


def process_split(split):
    ann_dir = f"data/raw/bccd/{split}/ann"
    img_dir = f"data/raw/bccd/{split}/img"

    out_img_dir = f"data/yolo_dataset/images/{split}"
    out_lbl_dir = f"data/yolo_dataset/labels/{split}"

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for file in os.listdir(ann_dir):
        if not file.endswith(".json"):
            continue

        path = os.path.join(ann_dir, file)

        with open(path, "r") as f:
            data = json.load(f)

        image_name = file.replace(".json", "")
        image_path = os.path.join(img_dir, image_name)

        if not os.path.exists(image_path):
            print("Missing image:", image_path)
            continue

        label_path = os.path.join(out_lbl_dir, image_name + ".txt")

        h = data["size"]["height"]
        w = data["size"]["width"]

        with open(label_path, "w") as out:
            for obj in data["objects"]:
                cls = obj["classTitle"]

                if cls not in classes:
                    continue

                x1 = obj["points"]["exterior"][0]
                x2 = obj["points"]["exterior"][1]

                x_center, y_center, bw, bh = convert_bbox(
                    {"width": w, "height": h},
                    [x1, x2]
                )

                out.write(f"{classes[cls]} {x_center} {y_center} {bw} {bh}\n")

        # copy image
        import shutil
        shutil.copy(image_path, os.path.join(out_img_dir, image_name))


def main():
    for split in ["train", "val", "test"]:
        print(f"Processing {split}...")
        process_split(split)

if __name__ == "__main__":
    main()