import json
import difflib

def compare_head_ablation(path_a, path_b, name_a, name_b):
    print(f"==================================================")
    print(f"COMPARING: {name_a} vs {name_b}")
    print(f"==================================================")
    with open(path_a, "r", encoding="utf-8") as f:
        nb_a = json.load(f)
    with open(path_b, "r", encoding="utf-8") as f:
        nb_b = json.load(f)
    
    cells_a = nb_a["cells"]
    cells_b = nb_b["cells"]
    print(f"Cell count: {len(cells_a)} vs {len(cells_b)}")

if __name__ == "__main__":
    compare_head_ablation(
        r"notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb",
        r"notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (2).ipynb",
        "head_ablation (3)",
        "head_ablation (2)"
    )
    compare_head_ablation(
        r"notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb",
        r"notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation.ipynb",
        "head_ablation (3)",
        "head_ablation"
    )
