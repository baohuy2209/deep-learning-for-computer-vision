import json

with open('d:/Downloads/main_model (8).ipynb', 'r', encoding='utf-8') as f:
    nb_dl = json.load(f)

src_dl = "".join(nb_dl['cells'][68]['source'])
for line in src_dl.split("\n"):
    if "val_ref_sens" in line or "Ref_Sensitivity_at_Spec" in line or "log_msg" in line or "Ref-Sens" in line:
        print(line)
