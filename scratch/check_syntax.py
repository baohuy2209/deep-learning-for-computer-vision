import json
import ast
import sys

sys.stdout.reconfigure(encoding='utf-8')

def test_ast(nb_path):
    print(f"\n--- Checking Python AST in {nb_path} ---")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    errors = []
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code":
            src = "".join(c.get("source", []))
            # filter out ipython magic lines like ! or %
            py_lines = []
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith("!") or stripped.startswith("%"):
                    py_lines.append(f"# {line}")
                else:
                    py_lines.append(line)
            py_code = "\n".join(py_lines)
            try:
                ast.parse(py_code)
            except SyntaxError as e:
                errors.append((i, str(e)))
    if errors:
        for cell_idx, err in errors:
            print(f"SyntaxError in Cell {cell_idx}: {err}")
    else:
        print("No Python syntax errors found!")

test_ast("notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb")
test_ast("notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb")
test_ast("notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb")
