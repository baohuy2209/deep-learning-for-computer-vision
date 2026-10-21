import json
import sys
import difflib

sys.stdout.reconfigure(encoding='utf-8')

def load_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

nb4 = load_nb("notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb")
nb8 = load_nb("notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb")

cells4 = nb4["cells"]
cells8 = nb8["cells"]

targets = [8, 10, 20, 23, 25, 39, 42, 44, 56, 64, 68, 70]

for idx in targets:
    s4 = "".join(cells4[idx].get("source", [])) if idx < len(cells4) else ""
    s8 = "".join(cells8[idx].get("source", [])) if idx < len(cells8) else ""
    
    print(f"\n{'='*70}")
    print(f"CELL {idx} COMPARISON")
    print(f"{'='*70}")
    lines4 = s4.splitlines(keepends=True)
    lines8 = s8.splitlines(keepends=True)
    
    diff = list(difflib.unified_diff(lines4, lines8, fromfile=f'(4)[{idx}]', tofile=f'(8)[{idx}]'))
    if diff:
        print("".join(diff[:50]))
        if len(diff) > 50:
            print(f"... [{len(diff)-50} more lines in diff]")
    else:
        print("IDENTICAL")
