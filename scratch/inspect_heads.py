import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    for h in ["SoftmaxCEHead", "SoftmaxQWKHead", "CORALHead", "CORNHead", "CumulativeLinkModelQWKHead", "build_model", "QWKLoss"]:
        if f"class {h}" in src or f"def {h}" in src:
            print(f"\n{'='*70}")
            print(f"CELL {i}: {h}")
            print(f"{'='*70}")
            print(src)
