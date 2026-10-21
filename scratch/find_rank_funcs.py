import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = "".join(cell.get('source', []))
    if 'extract_rank_probs' in src:
        print(f"extract_rank_probs in cell {i}")
    if 'rank_monotonicity_violations' in src:
        print(f"rank_monotonicity_violations in cell {i}")
