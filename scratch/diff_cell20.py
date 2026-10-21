import json
import sys
import difflib

sys.stdout.reconfigure(encoding='utf-8')

def load_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

nb4 = load_nb("notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb")
nb8 = load_nb("notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb")

s4 = "".join(nb4["cells"][20].get("source", []))
s8 = "".join(nb8["cells"][20].get("source", []))

lines4 = s4.splitlines(keepends=True)
lines8 = s8.splitlines(keepends=True)

diff = list(difflib.unified_diff(lines4, lines8, fromfile='(4)[20]', tofile='(8)[20]'))
print("".join(diff[:100]))
if len(diff) > 100:
    print(f"... [{len(diff)-100} more diff lines]")
