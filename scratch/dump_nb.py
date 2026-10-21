import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scratch/head_ablation_cells.txt', 'w', encoding='utf-8') as out:
    cells = nb.get('cells', [])
    out.write(f'Total cells: {len(cells)}\n')
    for idx, cell in enumerate(cells):
        ctype = cell['cell_type']
        src = ''.join(cell['source'])
        first_line = src.strip().split('\n')[0] if src.strip() else ''
        if ctype == 'markdown':
            lines = [l for l in src.split('\n') if l.strip().startswith('#')]
            title = lines[0] if lines else first_line[:80]
            out.write(f'Cell {idx:2d} [MD]  : {title}\n')
        else:
            out.write(f'Cell {idx:2d} [CODE]: {first_line[:100]}\n')
print('Done dumping cells')
