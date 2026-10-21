import ast
import json
import os
import sys

def verify_notebook_ast(path):
    print(f"\n[AST CHECK] Verifying: {path}")
    if not os.path.exists(path):
        print(f"  [FAIL] File not found: {path}")
        return False, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"  [FAIL] Invalid JSON: {e}")
        return False, None

    syntax_errors = []
    code_cells = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            code_cells.append((idx, source))
            # Filter out IPython magic commands (!, %) before parsing AST
            clean_lines = []
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("!") or stripped.startswith("%"):
                    clean_lines.append(f"# {line}")
                else:
                    clean_lines.append(line)
            clean_source = "\n".join(clean_lines)
            try:
                ast.parse(clean_source)
            except SyntaxError as e:
                syntax_errors.append((idx, str(e)))

    if syntax_errors:
        print(f"  [FAIL] Found {len(syntax_errors)} syntax errors in code cells:")
        for c_idx, err in syntax_errors[:5]:
            print(f"    Cell {c_idx}: {err}")
        return False, nb
    else:
        print(f"  [PASS] All {len(code_cells)} code cells passed AST syntax check.")
        return True, nb

def verify_main_model_training_parity(nb):
    print("\n[PARITY CHECK] Checking main_model (6).ipynb training parity...")
    all_code = "\n".join(["".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"])
    
    checks = {
        "DRTrainer defined": "class DRTrainer:" in all_code,
        "lr_scheduler steps on val_qwk": "self.lr_scheduler.step(val_qwk)" in all_code,
        "epochs_no_improve_es tracked": "epochs_no_improve_es" in all_code,
        "epochs_no_improve_lr tracked": "epochs_no_improve_lr" in all_code,
        "checkpoint saves best_val_qwk": "best_val_qwk" in all_code,
        "checkpoint saves best_val_loss": "best_val_loss" in all_code,
        "patience_reduce_lr passed": "patience_reduce_lr=" in all_code,
        "reduce_lr_factor passed": "reduce_lr_factor=" in all_code,
        "min_lr passed": "min_lr=" in all_code,
    }
    
    passed = True
    for desc, res in checks.items():
        status = "PASS" if res else "FAIL"
        if not res:
            passed = False
        print(f"  [{status}] {desc}")
    return passed

def verify_head_ablation_training_parity(nb):
    print("\n[PARITY CHECK] Checking main_model_head_ablation (3).ipynb training parity...")
    all_code = "\n".join(["".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"])
    
    checks = {
        "DRHeadTrainer defined": "class DRHeadTrainer:" in all_code,
        "epoch-level QWK warmup": "min(1.0, epoch / max(1, self.qwk_warmup_epochs))" in all_code or "min(1.0, epoch / max(1, qwk_warmup_epochs))" in all_code,
        "epochs_no_improve_es tracked": "epochs_no_improve_es" in all_code,
        "epochs_no_improve_lr tracked": "epochs_no_improve_lr" in all_code,
        "checkpoint saves best_val_qwk": "best_val_qwk" in all_code,
        "checkpoint saves best_val_loss": "best_val_loss" in all_code,
        "GPU memory cleanup preserved": "torch.cuda.empty_cache()" in all_code and "gc.collect()" in all_code,
    }
    passed = True
    for desc, res in checks.items():
        status = "PASS" if res else "FAIL"
        if not res:
            passed = False
        print(f"  [{status}] {desc}")
    return passed

def main():
    print("=" * 70)
    print("NOTEBOOK TRAINING MECHANISM PARITY & INTEGRITY AUDITOR")
    print("=" * 70)
    
    lodo_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
    main_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb"
    head_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb"
    
    lodo_ok, _ = verify_notebook_ast(lodo_path)
    main_ok, main_nb = verify_notebook_ast(main_path)
    head_ok, head_nb = verify_notebook_ast(head_path)
    
    if not (lodo_ok and main_ok and head_ok):
        print("\n[ERROR] AST verification failed for one or more notebooks.")
        return 1
        
    main_parity = verify_main_model_training_parity(main_nb)
    head_parity = verify_head_ablation_training_parity(head_nb)
    
    print("\n" + "=" * 70)
    if main_parity and head_parity:
        print("[SUMMARY] ALL AUDITS PASSED - Complete Training Parity Achieved!")
        print("=" * 70)
        return 0
    else:
        print("[SUMMARY] Some parity checks are currently incomplete.")
        print("=" * 70)
        return 2

if __name__ == "__main__":
    sys.exit(main())
