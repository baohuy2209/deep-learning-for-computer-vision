import ast
import json
import builtins

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Collect all imports and definitions across cells
defined = set(dir(builtins))
undefined_calls = []

for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    code = "".join(cell.get("source", []))
    tree = ast.parse(code)
    
    # Track definitions in this cell
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for n in node.names:
                defined.add(n.asname if n.asname else n.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for n in node.names:
                defined.add(n.asname if n.asname else n.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)

# Now in a second pass, check inside functions / classes / top-level for any Name that is never defined anywhere!
missing_symbols = set()
for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    code = "".join(cell.get("source", []))
    tree = ast.parse(code)
    
    # We want to find names loaded that are neither in defined nor local in scope
    # A simple approach: names loaded in any function/class that are not in defined and not arguments/locals
    class ScopeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.scopes = [set()]
            
        def visit_FunctionDef(self, node):
            # new scope
            local_vars = {arg.arg for arg in node.args.args}
            local_vars.update({arg.arg for arg in node.args.kwonlyargs})
            if node.args.vararg: local_vars.add(node.args.vararg.arg)
            if node.args.kwarg: local_vars.add(node.args.kwarg.arg)
            self.scopes.append(local_vars)
            
            # walk children
            for child in node.body:
                self.visit(child)
            self.scopes.pop()

        def visit_ClassDef(self, node):
            self.scopes.append(set())
            for child in node.body:
                self.visit(child)
            self.scopes.pop()

        def visit_Assign(self, node):
            for target in node.targets:
                for t in ast.walk(target):
                    if isinstance(t, ast.Name):
                        self.scopes[-1].add(t.id)
            self.generic_visit(node)

        def visit_For(self, node):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    self.scopes[-1].add(t.id)
            self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                name = node.id
                # check if in any scope or defined
                in_scope = any(name in s for s in self.scopes)
                if not in_scope and name not in defined:
                    missing_symbols.add((idx, name))

    ScopeVisitor().visit(tree)

print("ALL MISSING/UNDEFINED SYMBOLS:")
for cell_idx, name in sorted(missing_symbols):
    print(f"  Cell {cell_idx:2d}: '{name}'")
