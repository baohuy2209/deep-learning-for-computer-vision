import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', 'r', encoding='utf-8') as f:
    nb_repo = json.load(f)

with open('d:/Downloads/main_model (8).ipynb', 'r', encoding='utf-8') as f:
    nb_dl = json.load(f)

diff_cells = [6, 8, 62, 64, 65, 68, 69, 81, 88]

for idx in diff_cells:
    src_repo = "".join(nb_repo['cells'][idx].get('source', []))
    src_dl = "".join(nb_dl['cells'][idx].get('source', []))
    print(f"\n==================== CELL {idx} ({nb_repo['cells'][idx]['cell_type']}) ====================")
    import difflib
    diff = list(difflib.unified_diff(
        src_repo.splitlines(keepends=True),
        src_dl.splitlines(keepends=True),
        fromfile=f"repo_cell_{idx}",
        tofile=f"dl_cell_{idx}"
    ))
    print("".join(diff[:30]))
