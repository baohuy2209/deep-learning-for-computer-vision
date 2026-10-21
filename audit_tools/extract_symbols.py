"""Static symbol and definition extractor for golden reference notebooks and candidate package."""

import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def extract_code_from_notebook(nb_path: Path) -> str:
    """Extract code from all code cells in execution order."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    code_cells = []
    for idx, cell in enumerate(nb_data.get("cells", [])):
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            # Remove IPython magic commands that break standard AST parsing
            clean_lines = []
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith("%") or stripped.startswith("!"):
                    clean_lines.append(f"# {line}")
                else:
                    clean_lines.append(line)
            code_cells.append(f"# CELL {idx}\n" + "\n".join(clean_lines))
    return "\n\n".join(code_cells)


def parse_symbols(code_str: str, source_name: str) -> List[Dict[str, Any]]:
    """Parse AST to extract classes, functions, and top-level constants."""
    symbols = []
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        symbols.append({
            "name": "__syntax_error__",
            "type": "error",
            "source": source_name,
            "lineno": e.lineno,
            "details": str(e),
        })
        return symbols

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            symbols.append({
                "name": node.name,
                "type": "function",
                "source": source_name,
                "lineno": node.lineno,
                "args": args,
            })
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            bases = [ast.unparse(b) for b in node.bases]
            symbols.append({
                "name": node.name,
                "type": "class",
                "source": source_name,
                "lineno": node.lineno,
                "bases": bases,
                "methods": methods,
            })
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    val_str = ""
                    try:
                        val_str = ast.unparse(node.value)
                        if len(val_str) > 100:
                            val_str = val_str[:97] + "..."
                    except Exception:
                        pass
                    symbols.append({
                        "name": target.id,
                        "type": "constant_or_var",
                        "source": source_name,
                        "lineno": target.lineno,
                        "value_preview": val_str,
                    })
    return symbols


def main():
    root = Path(".")
    golden_files = [
        root / "notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb",
        root / "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb",
        root / "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb",
        root / "notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb",
        root / "notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge_dataset_lodo.ipynb",
        root / "src/get_sample_data/resize_image.py",
        root / "src/get_sample_data/total_aptos.py",
        root / "src/get_sample_data/total_deepdrid.py",
        root / "src/get_sample_data/total_idrid.py",
        root / "src/get_sample_data/total_idrid-segmentation.py",
        root / "src/download-dataset/ddr.py",
        root / "src/download-dataset/deep-drid.py",
        root / "src/download-dataset/idrid.py",
        root / "src/download-dataset/idrid-segmentation.py",
        root / "src/download-dataset/messidor.py",
    ]

    original_symbols = {}
    for p in golden_files:
        if not p.exists():
            print(f"Warning: {p} does not exist")
            continue
        rel_path = str(p.relative_to(root)).replace("\\", "/")
        if p.suffix == ".ipynb":
            code = extract_code_from_notebook(p)
        else:
            code = p.read_text(encoding="utf-8", errors="ignore")
        syms = parse_symbols(code, rel_path)
        original_symbols[rel_path] = syms

    out_orig = root / "audit_artifacts/original_symbols.json"
    with open(out_orig, "w", encoding="utf-8") as f:
        json.dump(original_symbols, f, indent=2)
    print(f"Extracted {sum(len(v) for v in original_symbols.values())} original symbols -> {out_orig}")

    # Extract symbols from refactored candidate src/drdg/
    refactored_symbols = {}
    drdg_dir = root / "src/drdg"
    for py_path in sorted(drdg_dir.rglob("*.py")):
        rel_path = str(py_path.relative_to(root)).replace("\\", "/")
        code = py_path.read_text(encoding="utf-8", errors="ignore")
        syms = parse_symbols(code, rel_path)
        refactored_symbols[rel_path] = syms

    out_ref = root / "audit_artifacts/refactored_symbols.json"
    with open(out_ref, "w", encoding="utf-8") as f:
        json.dump(refactored_symbols, f, indent=2)
    print(f"Extracted {sum(len(v) for v in refactored_symbols.values())} refactored symbols -> {out_ref}")


if __name__ == "__main__":
    main()
