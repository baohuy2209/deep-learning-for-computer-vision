import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

lodo_nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', encoding='utf-8'))
merge_nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb', encoding='utf-8'))
nb6 = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb', encoding='utf-8'))

def get_src(nb, idx):
    return ''.join(nb['cells'][idx]['source'])

print("=== CHECKING MODULE A & B: DATASET MERGING & PATIENT PARSING ===")
# Let's inspect cells 5, 6, 7, 8 of lodo
print("LODO Cell 6 (cohort paths):")
print(get_src(lodo_nb, 6)[:1000])

print("\nLODO Cell 7 (parse_patient_and_side):")
print(get_src(lodo_nb, 7)[:1000])

print("\nLODO Cell 8 (create_balanced_training_pool):")
print(get_src(lodo_nb, 8)[:1000])
