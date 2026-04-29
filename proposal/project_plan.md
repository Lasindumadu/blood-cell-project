Project plan and mapping — Automated Blood Cell Classification

Overview
--------
This file maps repository files to project components and defines immediate milestones.

Repository mapping
------------------
- data/ : contains datasets and prepared YOLO dataset folders (BCCD, yolo_dataset, processed). Use these for detection and fine-tuning.
- train_yolo.py : simple entry point to train a YOLOv8 model. Used for Stage 1 detection training (BCCD).
- predict.py : minimal inference example. Will be extended to a stable demo CLI in `tools/demo_cli.py`.
- yolov8n.pt : example starting weights in repo root.
- src/preprocessing.py : preprocessing functions (CLAHE, color transforms). Used by segmentation and demo pipeline.
- src/segmentation.py : segmentation helpers (threshold, distance transform). Used for mask generation and feature extraction.
- src/features.py : per-contour feature extraction (area, circularity). Extended for nuclear/cytoplasmic metrics.
- src/utils.py : small utilities (paths and plotting). Expand to include config parsing and paths used by scripts.

Short-term milestones (next 2--3 days)
-------------------------------------
1. Create a robust demo CLI that runs inference on a single image and outputs:
   - annotated image with bounding boxes
   - JSON summary with counts per class and per-box confidence
   File: `tools/demo_cli.py` (planned)

2. Add an evaluation wrapper using Ultralytics `model.val` to compute mAP and save results (CSV).
   File: `tools/evaluate.py` (planned)

3. Add basic unit test for preprocessing (pytest) and a README describing how to run the demo and tests.
   File: `tests/test_preprocessing.py`, `README_project.md` (planned)

4. Verify `requirements.txt` versions and pin key libraries if needed. (pending)

Longer-term milestones
----------------------
- Implement disorder detection rules module using `src/features.py` outputs and classification results.
- Add segmentation-quality evaluation (Dice) comparing masks from `src/segmentation.py` to ground-truth masks (if available).
- Build a Streamlit demo for interactive results and report generation.

How to run the demo (example)
-----------------------------
1. Install requirements: `pip install -r requirements.txt`.
2. Run demo on an image (PowerShell):
   python .\tools\demo_cli.py --image data/yolo_dataset/images/test/BloodImage_00007.jpg --model runs/detect/blood_cell_model/weights/best.pt --out results/

Notes
-----
If the trained weights are not present, the demo will fall back to `yolov8n.pt` shipped in the repo and produce sample outputs.

Data locations
--------------
- BCCD: `data/bccd/` and `data/bccd.yaml` (used by train_yolo.py)
- YOLO dataset: `data/yolo_dataset/` (images/labels organized for Ultralytics)

Authors: project team
