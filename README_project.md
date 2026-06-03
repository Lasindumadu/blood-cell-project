# Automated Blood Cell Classification and Hematological Disorder Detection System

**EE7204 / EC7205 — Image Processing and Computer Vision**  
Department of Electrical and Information Engineering, University of Ruhuna

---

## Overview

An end-to-end automated blood cell analysis pipeline that:
- Detects and counts **WBCs, RBCs, and Platelets** using a trained YOLOv8 model
- Segments each cell crop using **Otsu + Marker-Controlled Watershed**
- Extracts **22 morphological, nuclear, color (HSV), and textural features** per cell
- Screens for hematological disorders using **rule-based clinical logic**
- Generates **clinical reports** in PDF (archival), CSV (LIS integration), and JSON formats

---

## Performance Metrics

### Stage 1 — YOLOv8 Detection (BCCD val set)

| Class     | mAP@0.5 | Precision | Recall |
|-----------|---------|-----------|--------|
| WBC       | 0.805   | 0.722     | 0.762  |
| RBC       | **0.929** | 0.971   | 0.762  |
| Platelets | 0.692   | 0.628     | 0.735  |
| **All**   | **0.809** | 0.774   | 0.753  |

### Stage 2 — WBC Subtype Classifier (Blood Cell Images dataset)

| Metric | Value |
|--------|-------|
| Top-1 Accuracy (val) | **85.8%** |
| Quick test (20 samples/class) | **81.2%** |
| Classes | Eosinophil, Lymphocyte, Monocyte, Neutrophil |
| Training images | 9,957 (TRAIN split) |
| Model | YOLOv8n-cls |

Processing speed: ~150ms/image detection + ~5ms/WBC subtype (CPU)

---

## Quick Start

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Run CLI analysis on a single image

```powershell
python tools/demo_cli.py `
  --image data/yolo_dataset/images/test/BloodImage_00007.jpeg `
  --model runs/detect/test_yolo_dataset_stage12/weights/best.pt `
  --out results
```

Outputs:
- `results/<name>_annotated.jpg` — image with bounding boxes and class labels
- `results/<name>_summary.json` — JSON with counts, features, and disorder flags

### 3. Generate PDF + CSV clinical report

```powershell
python tools/generate_report.py `
  --image results/BloodImage_00007_annotated.jpg `
  --summary results/BloodImage_00007_summary.json `
  --out reports/report.pdf
```

Outputs `reports/report.pdf` (archival) and `reports/report.csv` (LIS integration).

### 4. Launch Streamlit web interface

```powershell
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Run evaluation

```powershell
python tools/evaluate.py `
  --model runs/detect/test_yolo_dataset_stage12/weights/best.pt `
  --data data/bccd.yaml `
  --out runs/val_results.csv
```

### 6. Run tests

```powershell
python -m pytest tests/ -v
```

---

## Project Structure

```
blood_cell_project/
├── src/
│   ├── preprocessing.py      # Gaussian + LAB + CLAHE pipeline
│   ├── segmentation.py       # Otsu + Watershed cell segmentation
│   ├── features.py           # 22-feature extraction (geometric/nuclear/HSV/GLCM/LBP)
│   ├── wbc_classifier.py     # WBC subtype classifier (Eosinophil/Lymphocyte/Monocyte/Neutrophil)
│   ├── disorder.py           # Rule-based disorder detection (ALL/anemia/thrombocytopenia/eosinophilia)
│   └── pipeline.py           # Two-stage orchestration (detect → subtype → segment → features → disorders)
├── app/
│   └── streamlit_app.py      # Web UI with CBC dashboard, PDF and CSV download
├── tools/
│   ├── demo_cli.py           # CLI: run inference on one image
│   ├── generate_report.py    # PDF + CSV clinical report generator
│   ├── train_pipeline.py     # Multi-stage YOLOv8 training script
│   └── evaluate.py           # Model evaluation / mAP calculation
├── config/
│   ├── disorders.yaml        # Disorder detection thresholds
│   └── datasets.yaml         # Dataset registry
├── data/
│   └── yolo_dataset/         # YOLO-format annotated BCCD images
├── runs/
│   └── detect/               # Trained YOLOv8 model weights
└── tests/
    ├── test_disorder.py        # 11 disorder detection unit tests
    ├── test_preprocessing.py   # 7 preprocessing/segmentation/feature tests
    └── test_wbc_classifier.py  # 6 WBC subtype classifier tests
```

---

## Disorder Detection Rules

| Disorder | Detection Logic |
|----------|----------------|
| **ALL (Leukemia)** | With subtypes: lymphocytes ≥20% of WBCs + WBC fraction ≥15% + ≥5 WBCs detected + nuclear irregularity >1.4. Without subtypes: WBC >30% of cells + ≥3 blast-like contours |
| **Leukocytosis** | WBC > 50% of total cells (morphology-only fallback) |
| **Eosinophilia** | Eosinophils ≥10% of WBCs (requires ≥5 WBCs) |
| **Microcytic Anemia** | Mean RBC segmentation area < 150px |
| **Sickle Cell Disease** | ≥30% of RBC contours with circularity < 0.20 |
| **Thrombocytopenia** | Platelet fraction < 2% AND <3 platelets detected |

---

## Technologies

- **Python 3.11** | **YOLOv8 (Ultralytics)** | **OpenCV 4.8** | **PyTorch**
- **scikit-image** (Watershed, GLCM, LBP) | **Streamlit** | **ReportLab**
