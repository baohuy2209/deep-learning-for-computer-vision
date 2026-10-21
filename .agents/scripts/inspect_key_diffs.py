import json
import sys
import difflib

sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb', 'r', encoding='utf-8') as f:
    nb_local = json.load(f)

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb', 'r', encoding='utf-8') as f:
    nb_colab = json.load(f)

key_pairs = [
    (0, 0, "Title & Overview Markdown"),
    (26, 31, "Section 3 Header Markdown"),
    (31, 36, "Section 3.1.5 Markdown (Non-local & Quadrant Tokens)"),
    (32, 37, "NonLocalBlock2D & QuadrantTokenAggregator"),
    (33, 38, "Section 3.2 Markdown (Global Branch)"),
    (34, 39, "GlobalBranch"),
    (37, 42, "Section 3.4 Markdown (Fusion)"),
    (38, 43, "DualBranchMILModel"),
    (53, 58, "DRFullModel"),
    (55, 60, "build_model"),
    (59, 64, "calculate_dr_metrics"),
    (62, 67, "DRTrainer"),
    (64, 69, "Training Run Cell"),
]

for l_idx, c_idx, title in key_pairs:
    l_src = ''.join(nb_local['cells'][l_idx]['source'])
    c_src = ''.join(nb_colab['cells'][c_idx]['source'])
    print(f"\n{'='*30} {title} (L{l_idx} vs C{c_idx}) {'='*30}")
    diff = list(difflib.unified_diff(
        l_src.splitlines(keepends=True),
        c_src.splitlines(keepends=True),
        fromfile=f"Local_L{l_idx}",
        tofile=f"Colab_C{c_idx}",
        n=2
    ))
    if not diff:
        print("EXACT MATCH")
    else:
        for line in diff:
            print(line, end='')
