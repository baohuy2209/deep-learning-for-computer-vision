import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

src = "".join(nb["cells"][68].get("source", []))
for line in src.splitlines():
    if any(k in line.lower() for k in ["early", "stopping", "patience", "best_val", "no_improve"]):
        print(line)
