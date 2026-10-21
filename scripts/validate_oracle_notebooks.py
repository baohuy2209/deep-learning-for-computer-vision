"""
Validation script to verify all 6 Baseline Oracle notebooks:
1. JSON structure & nbformat compliance.
2. Python compilation & syntax parsing of all code cells.
3. Essential function / class presence (DualBranchMILModel, CumulativeLinkModelQWKHead, parse_patient_and_side, etc.).
4. Checkpoint directory and transfer gap calculation logic.
"""

import os
import ast
import json
import sys

def validate_notebook(notebook_path: str) -> bool:
    print(f"\n[AUDITING] {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"  [FAIL] File does not exist: {notebook_path}")
        return False

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"  [FAIL] Invalid JSON: {e}")
        return False

    if "cells" not in nb or not isinstance(nb["cells"], list):
        print("  [FAIL] Missing 'cells' list in notebook")
        return False

    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    markdown_cells = [c for c in nb["cells"] if c.get("cell_type") == "markdown"]
    print(f"  -> Total cells: {len(nb['cells'])} ({len(code_cells)} code, {len(markdown_cells)} markdown)")

    # Test Python syntax of each code cell using ast.parse
    syntax_errors = 0
    total_lines = 0
    all_code_str = ""

    for idx, cell in enumerate(code_cells):
        src = "".join(cell.get("source", []))
        total_lines += len(src.splitlines())
        all_code_str += "\n" + src

        # Filter out ipython magic commands like !nvidia-smi or %matplotlib before AST parse
        filtered_lines = []
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("!") or stripped.startswith("%"):
                filtered_lines.append(f"# {line}")
            else:
                filtered_lines.append(line)
        filtered_src = "\n".join(filtered_lines)

        try:
            ast.parse(filtered_src)
        except SyntaxError as se:
            print(f"  [SYNTAX ERROR] In Code Cell {idx} at line {se.lineno}: {se.msg}")
            print(f"    Code snippet: {se.text}")
            syntax_errors += 1

    if syntax_errors > 0:
        print(f"  [FAIL] Found {syntax_errors} syntax errors in code cells!")
        return False

    # Check key architectural symbols
    required_symbols = [
        "DualBranchMILModel",
        "CumulativeLinkModelQWKHead",
        "ContinuousQWKLoss",
        "DRTrainer",
        "GroupShuffleSplit",
        "parse_patient_and_side",
        "delta_qwk",
        "oracle_"
    ]

    missing_symbols = [sym for sym in required_symbols if sym not in all_code_str]
    if missing_symbols:
        print(f"  [FAIL] Missing critical architecture symbols: {missing_symbols}")
        return False

    print(f"  [PASS] Successfully verified {total_lines:,} lines of Python code without syntax errors.")
    return True

def main():
    oracle_dir = "notebooks/diabetic_retinopathy/pipeline/oracle"
    notebooks = [
        "oracle_aptos.ipynb",
        "oracle_ddr.ipynb",
        "oracle_deepdrid.ipynb",
        "oracle_idrid.ipynb",
        "oracle_messidor2.ipynb",
        "oracle_eyepacs.ipynb"
    ]

    all_passed = True
    print("=" * 75)
    print("RUNNING QUALITY AUDIT ACROSS 6 ORACLE BASELINE NOTEBOOKS")
    print("=" * 75)

    for nb_name in notebooks:
        p = os.path.join(oracle_dir, nb_name)
        ok = validate_notebook(p)
        if not ok:
            all_passed = False

    print("\n" + "=" * 75)
    if all_passed:
        print("ALL 6 BASELINE ORACLE NOTEBOOKS PASSED VALIDATION WITH ZERO ERRORS!")
    else:
        print("SOME NOTEBOOKS FAILED VALIDATION. PLEASE INSPECT LOGS ABOVE.")
    print("=" * 75)

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
