import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_23 = nb['cells'][23]
lines = cell_23['source']
for i, line in enumerate(lines):
    if any(k in line for k in ['read_csv', 'train_df', 'val_df', 'test_df', 'LODO_FOLDS_DIR', 'fold_', 'test_csv', 'train_csv', 'val_csv', 'Dataset(']):
        print(f"L{i}: {line.rstrip()}")
