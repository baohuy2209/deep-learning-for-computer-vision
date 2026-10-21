import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', 'r', encoding='utf-8') as f:
    nb_repo = json.load(f)

with open('d:/Downloads/main_model (8).ipynb', 'r', encoding='utf-8') as f:
    nb_dl = json.load(f)

diff_cells = []
for i in range(92):
    src_repo = "".join(nb_repo['cells'][i].get('source', []))
    src_dl = "".join(nb_dl['cells'][i].get('source', []))
    if src_repo != src_dl:
        diff_cells.append(i)

print(f"Source diff cells: {diff_cells}")
