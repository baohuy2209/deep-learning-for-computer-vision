import json

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = "".join(cell.get('source', []))
    if "plot_fig7_referable_dr_roc" in src:
        print(f"Cell {idx} has plot_fig7_referable_dr_roc")
        for line in src.splitlines():
            if "ax.scatter" in line or "ax.axvline" in line or "Specificity" in line or "FPR" in line:
                print("  ", line)
