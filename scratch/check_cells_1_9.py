import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb", "r", encoding="utf-8") as f:
    nb4 = json.load(f)

for i in range(1, 10):
    src = "".join(nb4["cells"][i].get("source", []))
    print(f"\n--- Cell {i} ({nb4['cells'][i]['cell_type']}) ---")
    print(src[:400])
