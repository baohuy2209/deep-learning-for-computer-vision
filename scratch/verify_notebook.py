import json
import ast

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

errors = []
code_cells = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code_cells += 1
        src = "".join(cell["source"])
        lines = [l for l in src.split("\n") if not l.strip().startswith("%") and not l.strip().startswith("!")]
        clean_src = "\n".join(lines)
        try:
            ast.parse(clean_src)
        except Exception as e:
            errors.append((i, str(e)))

if errors:
    print("Syntax errors found:")
    for idx, err in errors:
        print(f"Cell {idx}: {err}")
    exit(1)
else:
    print(f"Success! All {code_cells} code cells out of {len(nb['cells'])} total cells verified with 0 syntax errors.")
