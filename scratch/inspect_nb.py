import json
from pathlib import Path

def inspect_notebook(path):
    print(f"=== {path} ===")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if any(term in src for term in [".csv", "fold_", "read_csv", "lodo_folds", "splits_dir", "train_df"]):
            matches = [l.strip() for l in src.split("\n") if any(k in l for k in [".csv", "fold_", "read_csv", "pd.", "lodo_folds"])]
            if matches:
                print(f"--- Cell {i} ---")
                for m in matches[:6]:
                    print("  ", m)

inspect_notebook("notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb")
inspect_notebook("notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge_dataset_lodo.ipynb")
