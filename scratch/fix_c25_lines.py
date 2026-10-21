import json, ast

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

lines = nb['cells'][25]['source']

lines[125] = '            ann_text = f"Δ: {val:.3f}\\n({marginal:+.3f})\\n[{tag}]"\n'
lines[127] = '            ann_text = f"Δ: {val:.3f}\\n[{tag}]"\n'

nb['cells'][25]['source'] = lines

src = ''.join(lines)
ast.parse(src)
print("Cell 25 ast.parse() passed successfully!")

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Saved cleanly!")
