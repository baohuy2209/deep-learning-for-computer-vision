import json

NOTEBOOK_PATH = 'notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb'
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for c in nb['cells']:
    if c['cell_type'] == 'code':
        src = c['source']
        new_src = []
        for line in src:
            if '1D Latent Severity Score' in line and not line.strip().startswith('r"'):
                line = line.replace('ax.set_xlabel("1D Latent Severity', 'ax.set_xlabel(r"1D Latent Severity')
            new_src.append(line)
        c['source'] = new_src

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Fixed raw string cleanly.')
