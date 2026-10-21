import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def inspect_merge_nb(path):
    print(f"=== {path} ===")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for i, c in enumerate(nb['cells']):
        src = "".join(c.get('source', []))
        if any(kw in src for kw in ['crop_fundus_roi', 'compute_fov_mask', 'parse_patient_and_side', 'ben_graham', 'create_balanced_train_pool', 'GroupShuffleSplit', 'tolerance =']):
            print(f"--- Cell {i} ---")
            for line in src.split('\n'):
                if any(kw in line for kw in ['def ', 'tolerance', 'threshold', 'cv2.', 'npz', 'h5', 'cache', 'GroupShuffleSplit', 'quota', 'balance', 'leakage', 'patient']):
                    print(line[:100])

inspect_merge_nb('notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb')
inspect_merge_nb('notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge_dataset_lodo.ipynb')
