import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', encoding='utf-8'))

for idx, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'class IDRiDXAIEvalDataset' in src or 'def compute_quantitative_xai' in src or 'class DRTrainer' in src:
        print(f"Cell {idx} ({cell['cell_type']}):")
        lines = src.split('\n')
        for l in lines[:30]:
            print("  ", l)
        print("  ...")
        print("\n" + "="*40)
