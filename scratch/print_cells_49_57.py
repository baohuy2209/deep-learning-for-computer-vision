import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in range(49, 58):
    c = nb["cells"][i]
    src = "".join(c.get("source", []))
    print(f"\n{'='*70}")
    print(f"CELL {i} [{c.get('cell_type')}]:")
    print(f"{'='*70}")
    print(src)
