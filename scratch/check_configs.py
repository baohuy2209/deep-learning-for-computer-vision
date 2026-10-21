import json

def get_config_cells(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"\n{'='*20} CONFIG in {path} {'='*20}")
    for i, cell in enumerate(nb['cells']):
        src = "".join(cell.get('source', []))
        if 'CONFIG' in src or 'EPOCHS' in src or 'BATCH_SIZE' in src or 'LEARNING_RATE' in src:
            if cell['cell_type'] == 'code' and ('LEARNING_RATE' in src or 'EPOCHS' in src):
                print(f"--- Cell {i} ---")
                for line in src.splitlines():
                    if any(k in line for k in ['EPOCHS', 'BATCH_SIZE', 'LR', 'WEIGHT_DECAY', 'PATIENCE', 'WARMUP', 'GRAD_CLIP', 'T_MAX']):
                        print("  ", line.strip())

get_config_cells('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb')
get_config_cells('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb')
get_config_cells('notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb')
