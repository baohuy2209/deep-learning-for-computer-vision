import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def load_cells(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    return [(i, c.get("cell_type"), "".join(c.get("source", []))) for i, c in enumerate(nb.get("cells", []))]

c4 = load_cells("notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb")
c8 = load_cells("notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb")

print(f"{'Cell':<6} | {'Status':<10} | {'Summary in (4)':<45} | {'Summary in (8)':<45}")
print("-" * 115)

max_len = max(len(c4), len(c8))
for i in range(max_len):
    i4, t4, s4 = c4[i] if i < len(c4) else (None, None, "")
    i8, t8, s8 = c8[i] if i < len(c8) else (None, None, "")
    
    first_line_4 = s4.strip().split("\n")[0][:43] if s4.strip() else "[EMPTY]"
    first_line_8 = s8.strip().split("\n")[0][:43] if s8.strip() else "[EMPTY]"
    
    if s4.strip() == s8.strip():
        # identical
        continue
    else:
        print(f"{i:<6} | {'DIFFERENT':<10} | {first_line_4:<45} | {first_line_8:<45}")
