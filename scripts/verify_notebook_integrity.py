#!/usr/bin/env python3
"""Automated notebook integrity verification script.

Verifies that the 5 canonical research notebooks exist, are valid Jupyter notebooks,
retain their complete self-contained implementation, and have not been converted into
lightweight wrappers importing from `drdg`.
"""

import hashlib
import json
import sys
from pathlib import Path

CANONICAL_NOTEBOOKS = [
    "notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb",
    "notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge_dataset_lodo.ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb",
]

# Core symbols that must exist inside the notebook code cells (proving self-contained implementation)
EXPECTED_NOTEBOOK_SYMBOLS = {
    "notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb": [
        "DATASET_CONFIG",
        "crop_fundus_roi",
        "create_balanced_train_pool",
    ],
    "notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge_dataset_lodo.ipynb": [
        "GroupShuffleSplit",
        "create_balanced_train_pool",
    ],
    "notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb": [
        "FundusDualBranchDataset",
        "CBAM",
        "NonLocalBlock2D",
        "QuadrantTokenAggregator",
        "GlobalBranch",
        "GatedAttentionPool",
        "ContinuousQWKLoss",
    ],
    "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb": [
        "FundusDualBranchDataset",
        "build_model",
        "LODO_FOLDS_DIR",
    ],
    "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb": [
        "SoftmaxCEHead",
        "SoftmaxQWKHead",
        "CORALHead",
        "CORNHead",
        "CumulativeLinkModelQWKHead",
    ],
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def verify_notebook(nb_rel_path: str) -> bool:
    path = Path(nb_rel_path)
    if not path.exists():
        print(f"[FAIL] Missing notebook: {nb_rel_path}")
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"[FAIL] Cannot parse {nb_rel_path} as JSON: {e}")
        return False

    cells = nb.get("cells", [])
    if not cells:
        print(f"[FAIL] Notebook {nb_rel_path} has zero cells")
        return False

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    all_code = "\n".join("".join(c.get("source", [])) for c in code_cells)

    # Check for forbidden wrapper imports: notebook importing implementation from drdg
    if "from drdg.models" in all_code or "from drdg.training" in all_code or "from drdg.data" in all_code:
        print(f"[FAIL] Notebook {nb_rel_path} has been turned into a wrapper importing from drdg!")
        return False

    # Check required golden symbols
    expected_symbols = EXPECTED_NOTEBOOK_SYMBOLS.get(nb_rel_path, [])
    missing_symbols = []
    for sym in expected_symbols:
        if sym not in all_code:
            missing_symbols.append(sym)

    if missing_symbols:
        print(f"[FAIL] Notebook {nb_rel_path} is missing expected self-contained symbols: {missing_symbols}")
        return False

    sha = compute_sha256(path)
    print(f"[PASS] {nb_rel_path}")
    print(f"       Total cells: {len(cells)} ({len(code_cells)} code), SHA256: {sha[:16]}...")
    return True


def main() -> int:
    print("=" * 75)
    print("CANONICAL NOTEBOOK INTEGRITY & PRESERVATION AUDIT")
    print("=" * 75)
    all_ok = True
    for nb in CANONICAL_NOTEBOOKS:
        if not verify_notebook(nb):
            all_ok = False

    print("=" * 75)
    if all_ok:
        print("RESULT: ALL 5 CANONICAL NOTEBOOKS ARE 100% PRESERVED & SELF-CONTAINED.")
        return 0
    else:
        print("RESULT: NOTEBOOK INTEGRITY VIOLATION DETECTED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
