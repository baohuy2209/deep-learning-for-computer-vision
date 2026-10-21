import json

def inspect_cells():
    # 1. main_model_lodo.ipynb
    with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'r', encoding='utf-8') as f:
        lodo_nb = json.load(f)
    print("=== LODO Cell 22 (DRTrainer definition) ===")
    print("".join(lodo_nb['cells'][22]['source'])[:1500])
    print("...\n")
    
    print("=== LODO Cell 25 (LODO training execution) ===")
    print("".join(lodo_nb['cells'][25]['source'])[:1500])
    print("...\n")

    # 2. main_model (6).ipynb
    with open('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb', 'r', encoding='utf-8') as f:
        main_nb = json.load(f)
    print("=== main_model (6) Cell 68 (DRTrainer definition) ===")
    print("".join(main_nb['cells'][68]['source'])[:1500])
    print("...\n")

    print("=== main_model (6) Cell 70 (training execution) ===")
    print("".join(main_nb['cells'][70]['source'])[:1500])
    print("...\n")

    # 3. main_model_head_ablation (3).ipynb
    with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb', 'r', encoding='utf-8') as f:
        head_nb = json.load(f)
    print("=== head_ablation (3) Cell 27 (DRHeadTrainer definition) ===")
    print("".join(head_nb['cells'][27]['source'])[:1500])
    print("...\n")

    print("=== head_ablation (3) Cell 29 (head ablation training execution) ===")
    print("".join(head_nb['cells'][29]['source'])[:1500])
    print("...\n")

if __name__ == '__main__':
    inspect_cells()
