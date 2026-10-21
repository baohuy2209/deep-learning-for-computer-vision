import json

nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', encoding='utf-8'))

for idx, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    src = ''.join(cell.get('source', []))
    lines = [l.strip() for l in src.split('\n') if l.strip()]
    first_line = lines[0] if lines else ''
    
    # print headers or notable cells
    if ctype == 'markdown':
        if first_line.startswith('#'):
            print(f"Cell {idx} [MD]: {first_line}")
    else:
        # Check keywords
        keywords = ['train', 'test', 'eval', 'save', 'load', 'plot', 'visual', 'metric', 'confusion', 'roc', 'gradcam', 'cam']
        found = [k for k in keywords if k in src.lower()]
        if any(w in first_line.lower() for w in ['def ', 'class ', 'import ', 'plt.', 'sns.', 'save', 'torch.load', 'best_model']):
            print(f"Cell {idx} [CODE] ({','.join(found[:3])}): {first_line[:90]}")
        elif idx > 60: # Look at latter half of notebook where evaluation/visualization usually happens
            print(f"Cell {idx} [CODE] ({','.join(found[:3])}): {first_line[:90]}")
