import json, ast, os

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Initial cell count: {len(nb['cells'])}")

# 1. Identify cells to remove: Cells 5, 6, 7, 8
# Cell 5: markdown "## 2. Master Dataset Harmonization..."
# Cell 6: code parse_patient_and_side
# Cell 7: code load_lodo_fold_splits
# Cell 8: code audit loop
print("Cell 5 first line:", "".join(nb["cells"][5]["source"])[:60])
print("Cell 6 first line:", "".join(nb["cells"][6]["source"])[:60])
print("Cell 7 first line:", "".join(nb["cells"][7]["source"])[:60])
print("Cell 8 first line:", "".join(nb["cells"][8]["source"])[:60])

# Delete cells 5, 6, 7, 8
del nb["cells"][5:9]
print(f"Cell count after removing 5..8: {len(nb['cells'])}")

# Now check new cell indices
for idx, c in enumerate(nb["cells"]):
    first = "".join(c["source"]).splitlines()[0] if c["source"] else "EMPTY"
    print(f"Cell {idx:02d} [{c['cell_type']}]: {first[:70]}")
