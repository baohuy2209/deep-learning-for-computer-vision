import json

def find_symbols(path, keywords):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"=== {path} ===")
    all_code = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code")
    for kw in keywords:
        found = [line.strip() for line in all_code.split("\n") if kw.lower() in line.lower() and ("def " in line or "class " in line)]
        print(f"Keyword '{kw}':", found[:3])

find_symbols("notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb", ["preprocess", "crop", "save", "clahe", "balance"])
find_symbols("notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb", ["coral", "corn", "clm", "head", "softmax"])
