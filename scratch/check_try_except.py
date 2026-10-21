import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if 'except' in src:
        print(f"==================================================")
        print(f"Cell {idx}")
        print(f"==================================================")
        lines = src.split('\n')
        for i, line in enumerate(lines):
            if 'try:' in line or 'except' in line:
                # print 3 lines before and after
                start = max(0, i - 2)
                end = min(len(lines), i + 5)
                for j in range(start, end):
                    print(f"  {j:3d}: {lines[j]}")
                print("  ---")
