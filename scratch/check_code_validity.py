import ast
import json
from pathlib import Path

nb_path = Path("notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Checking AST syntax of all code cells...")
syntax_errors = []
for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        code = "".join(cell.get("source", []))
        try:
            ast.parse(code)
        except SyntaxError as e:
            syntax_errors.append((idx, str(e)))

if syntax_errors:
    print(f"Found {len(syntax_errors)} syntax errors:")
    for idx, err in syntax_errors:
        print(f"  Cell {idx}: {err}")
else:
    print("All code cells pass AST syntax validation!")
