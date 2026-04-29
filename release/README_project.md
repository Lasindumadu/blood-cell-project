# Project demo and test instructions

This repo contains scripts to run inference, evaluate a YOLOv8 model, and basic unit tests.

Demo
----
Run the demo CLI on a single image (PowerShell):

```powershell
python .\tools\demo_cli.py --image data/yolo_dataset/images/test/BloodImage_00007.jpg --model runs/detect/blood_cell_model/weights/best.pt --out results/
```

If `runs/detect/blood_cell_model/weights/best.pt` is missing, the demo will fallback to `yolov8n.pt` in the repo root (if present).

Evaluation
----------
Run validation via Ultralytics wrapper:

```powershell
python .\tools\evaluate.py --model runs/detect/blood_cell_model/weights/best.pt --data data/bccd.yaml --out runs/val_results.csv
```

Tests
-----
Run pytest from the repo root:

```powershell
pytest -q
```

Notes
-----
- Ensure Python environment has the packages listed in `requirements.txt` installed. For GPU training/inference, install suitable `torch` and CUDA matching your system.
- If you want, I can pin and update `requirements.txt` to recommended versions for reproducibility.

Additional tools
----------------
- Evaluate model (metrics saved to CSV):

```powershell
python .\tools\evaluate.py --model runs/detect/blood_cell_model/weights/best.pt --data data/bccd.yaml --out runs/val_results.csv
```

- Multi-stage training wrapper (runs three stages and saves runs under `runs/train/`):

```powershell
python .\tools\train_pipeline.py --data data/bccd.yaml --weights yolov8n.pt --name blood_cell_model
```

- The demo now includes simple rule-based disorder detection in `src/pipeline.py` and outputs `*_summary.json` next to the annotated image.

- Generate a PDF report from the annotated image and JSON summary (requires `reportlab`):

```powershell
python .\tools\generate_report.py --image results\BloodImage_00007_annotated.jpg --summary results\BloodImage_00007_summary.json --out reports\BloodImage_00007_report.pdf
```

Install `reportlab` if needed:

```powershell
python -m pip install reportlab
```
