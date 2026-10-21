import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][22]['source'])
src = src.replace('ax.set_xlabel("1D Latent', 'ax.set_xlabel(r"1D Latent')
src = src.replace('ax.set_title("Figure 7: Clinical ROC Curve for Referable DR (Grade $\\geq$', 'ax.set_title(r"Figure 7: Clinical ROC Curve for Referable DR (Grade $\\geq$')
src = src.replace('ax.set_title("Figure 7: Clinical ROC', 'ax.set_title(r"Figure 7: Clinical ROC')

lines = [l + '\n' for l in src.split('\n')]
if lines and lines[-1] == '\n':
    lines.pop()
nb['cells'][22]['source'] = lines

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully fixed raw strings in cell 22!")
