import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def print_cell_code(nb_path, cell_indices):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"=== {nb_path} ===")
    for idx in cell_indices:
        if idx < len(nb['cells']):
            print(f"\n--- Cell {idx} ({nb['cells'][idx]['cell_type']}) ---")
            print("".join(nb['cells'][idx].get('source', [])))

# Let's inspect hyperparameters and core classes in main_model (6).ipynb
print_cell_code('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb', [4, 5, 6, 23, 38, 40, 42, 44, 47, 57, 68])
