import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', 'r', encoding='utf-8') as f:
    nb_repo = json.load(f)

with open('d:/Downloads/main_model (8).ipynb', 'r', encoding='utf-8') as f:
    nb_dl = json.load(f)

# Check cell 68 outputs (training output)
print("Repo cell 68 outputs count:", len(nb_repo['cells'][68].get('outputs', [])))
print("DL   cell 68 outputs count:", len(nb_dl['cells'][68].get('outputs', [])))

# Check cell 70 or whichever has training output
for i in range(len(nb_repo['cells'])):
    out_repo = nb_repo['cells'][i].get('outputs', [])
    out_dl = nb_dl['cells'][i].get('outputs', [])
    if len(out_repo) > 0 or len(out_dl) > 0:
        if len(out_repo) != len(out_dl):
            print(f"Cell {i} outputs count diff: Repo {len(out_repo)} vs DL {len(out_dl)}")
