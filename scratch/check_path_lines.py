import json

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Inspect precompute_dataset_cache in cell 10
c10_src = "".join(nb["cells"][10]["source"])
print("Cell 10 path lines:")
for l in c10_src.splitlines():
    if "img_path" in l or "native_path" in l:
        print(" ", l)

# Inspect FundusDualBranchDataset in cell 11
c11_src = "".join(nb["cells"][11]["source"])
print("Cell 11 path lines:")
for l in c11_src.splitlines():
    if "img_path" in l or "native_path" in l:
        print(" ", l)
