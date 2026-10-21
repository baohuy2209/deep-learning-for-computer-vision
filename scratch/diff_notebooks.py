import json
import difflib

def compare_notebooks(path_a, path_b, name_a, name_b):
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

    diff_count = 0
    for idx in range(min(len(cells_a), len(cells_b))):
        src_a = "".join(cells_a[idx].get("source", []))
        src_b = "".join(cells_b[idx].get("source", []))
        type_a = cells_a[idx].get("cell_type")
        type_b = cells_b[idx].get("cell_type")

        if src_a != src_b or type_a != type_b:
            diff_count += 1
            print(f"\n--- Difference at Cell {idx} (types: {type_a} vs {type_b}) ---")
            diff = list(difflib.unified_diff(
                src_a.splitlines(),
                src_b.splitlines(),
                fromfile=name_a,
                tofile=name_b,
                lineterm=""
            ))
            for line in diff[:25]:
                print(line)
            if len(diff) > 25:
                print(f"... ({len(diff)-25} more lines)")
    print(f"\nTotal differing cells: {diff_count}")

if __name__ == "__main__":
    compare_notebooks(
        r"d:\Downloads\main_model (8).ipynb",
        r"notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb",
        "Downloads/main_model (8)",
        "main_model (4)"
    )
    compare_notebooks(
        r"d:\Downloads\main_model (8).ipynb",
        r"notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb",
        "Downloads/main_model (8)",
        "main_model"
    )
