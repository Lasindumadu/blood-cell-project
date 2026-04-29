"""Evaluate disorder detection thresholds across test dataset."""
import sys
from pathlib import Path
import json

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from src.pipeline import analyze_image
from src.disorder import detect_all_disorders

# Test images
test_dir = Path('data/yolo_dataset/images/test')
test_images = sorted(test_dir.glob('*.jpeg'))[:5]  # test first 5 images

# Threshold combinations to test
thresholds = [
    {'lymph_fraction': 0.15, 'blast_min': 2, 'name': 'Lenient'},
    {'lymph_fraction': 0.20, 'blast_min': 3, 'name': 'Default'},
    {'lymph_fraction': 0.25, 'blast_min': 3, 'name': 'Moderate'},
    {'lymph_fraction': 0.30, 'blast_min': 4, 'name': 'Strict'},
    {'lymph_fraction': 0.40, 'blast_min': 5, 'name': 'Very Strict'},
]

model = 'runs/detect/test_yolo_dataset_stage12/weights/best.pt'

print("=" * 80)
print("THRESHOLD EVALUATION REPORT")
print("=" * 80)
print(f"Images tested: {len(test_images)}")
print(f"Model: {model}")
print()

for t in thresholds:
    print(f"\n--- {t['name']} (lymph>{t['lymph_fraction']}, blast≥{t['blast_min']}) ---")
    all_count = 0
    anemia_count = 0
    details = []
    
    for img in test_images:
        result = analyze_image(str(img), model_path=model, conf=0.25)
        summary = result['summary']
        
        # Manually check ALL criteria
        counts = summary.get('counts', {})
        boxes = summary.get('boxes', [])
        total_wbc = sum(v for k, v in counts.items() if k != '0')  # exclude RBC class 0
        
        lymph = counts.get('1', 0)  # class 1 = lymphocytes
        lymph_frac = lymph / total_wbc if total_wbc > 0 else 0
        
        blast_like = 0
        for b in boxes:
            if str(b.get('class')) == '1':
                for f in b.get('features', []):
                    if f.get('circularity', 1.0) < 0.6 and f.get('area', 0) > 200:
                        blast_like += 1
        
        has_all = lymph_frac > t['lymph_fraction'] and blast_like >= t['blast_min']
        if has_all:
            all_count += 1
            
        details.append({
            'image': img.name,
            'wbc': total_wbc,
            'lymph': lymph,
            'lymph_frac': f"{lymph_frac:.2%}",
            'blast': blast_like,
            'all': has_all
        })
    
    print(f"  Images flagged for ALL: {all_count}/{len(test_images)} ({all_count/len(test_images)*100:.0f}%)")
    print(f"  {'Image':<20} {'WBC':>5} {'Lymph':>6} {'Lymph%':>8} {'Blast':>6} {'ALL?':>5}")
    print(f"  {'-'*20} {'-'*5} {'-'*6} {'-'*8} {'-'*6} {'-'*5}")
    for d in details:
        print(f"  {d['image']:<20} {d['wbc']:>5} {d['lymph']:>6} {d['lymph_frac']:>8} {d['blast']:>6} {str(d['all']):>5}")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print("""
Without ground-truth labels, choose thresholds based on:
1. MEDICAL LITERATURE: ALL typically shows >20-30% lymphoblasts in blood
2. YOUR DATASET: If you KNOW some images are healthy, pick threshold that 
   flags ~0% of healthy images (avoids false positives)
3. CONSERVATIVE: Use 'Strict' (0.30, 4) for fewer false alarms
4. SENSITIVE: Use 'Lenient' (0.15, 2) to catch more potential cases

For a student project, the Default (0.20, 3) is reasonable.
""")
