import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def inspect_nb_keywords(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"\n==================== {nb_path} ====================")
    for i, c in enumerate(nb['cells']):
        src = "".join(c.get('source', []))
        if any(kw in src for kw in ['HEAD_TYPES =', 'LEARNING_RATE =', 'EPOCHS =', 'BATCH_SIZE =', 'class ResNet50AblationModel', 'def train_epoch', 'HEAD_METRICS', 'def evaluate']):
            print(f"--- Cell {i} ({c['cell_type']}) ---")
            lines = [l for l in src.split('\n') if any(kw in l for kw in [
                'HEAD_TYPES', 'LEARNING_RATE', 'EPOCHS', 'BATCH_SIZE', 'WEIGHT_DECAY', 
                'TARGET_QWK_WEIGHT', 'QWK_WARMUP', 'PATIENCE', 'T_MAX', 'class ResNet50AblationModel',
                'head_type', 'backbone', 'Linear'
            ])]
            print("\n".join(lines[:10]))

inspect_nb_keywords('notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb')
