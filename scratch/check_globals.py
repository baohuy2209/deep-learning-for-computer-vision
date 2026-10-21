import ast
import json
import builtins

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Collect all global definitions and imported names
defined_names = set(dir(builtins))
undefined_usages = []

for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    code = "".join(cell.get("source", []))
    tree = ast.parse(code)
    
    # First find all assigned or imported names in this cell
    cell_defs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store)):
            cell_defs.add(node.id)
        elif isinstance(node, ast.FunctionDef):
            cell_defs.add(node.name)
        elif isinstance(node, ast.ClassDef):
            cell_defs.add(node.name)
        elif isinstance(node, ast.Import):
            for n in node.names:
                cell_defs.add(n.asname if n.asname else n.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for n in node.names:
                cell_defs.add(n.asname if n.asname else n.name)

    # Now find all loaded names in this cell that are not in defined_names and not in cell_defs
    # Note: local variables in functions shouldn't be flagged as global undefined
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            # skip deep checking for now or check top-level calls
            continue
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Name) and isinstance(subnode.ctx, ast.Load):
                if subnode.id not in defined_names and subnode.id not in cell_defs:
                    undefined_usages.append((idx, subnode.id))
                    
    defined_names.update(cell_defs)

print("Top-level undefined usages:")
for cell_idx, name in set(undefined_usages):
    print(f"  Cell {cell_idx}: undefined '{name}'")
