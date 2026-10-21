import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if any(k in src for k in ["cache_file", "CACHE_DIR", "warm_cache", "np.savez", "np.load"]):
        print(f"\n--- Cell {i} [{c.get('cell_type')}] ---")
        for line in src.splitlines():
            if any(k in line.lower() for k in ["cache", "npz", "np.save", "np.load", "anchors"]):
                print(" ", line[:120])
