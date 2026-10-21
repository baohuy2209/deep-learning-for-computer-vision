import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    for l_idx, line in enumerate(src.split('\n')):
        if any(w in line for w in ['np.zeros', 'np.full', 'torch.zeros', 'torch.full', 'zeros_like']):
            print(f"Cell {idx:2d}, line {l_idx:2d}: {line.strip()}")
