import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell25_lines = nb['cells'][25]['source']

new_lines = []
for line in cell25_lines:
    if 'ax.set_ylabel("Generalization Drop $\\Delta' in line:
        new_lines.append('    ax.set_ylabel(r"Generalization Drop $\\Delta_{\\mathrm{LODO}} = (\\mathrm{Val} - \\mathrm{OOD\\,Test})$", fontweight=\'bold\')\n')
    elif 'ax.set_title("Figure 4: Waterfall Generalization Gap (\\Delta' in line:
        new_lines.append('    ax.set_title(r"Figure 4: Waterfall Generalization Gap ($\Delta$ QWK) Decomposition Across Architectural Ladder", fontsize=12.5, fontweight=\'bold\', pad=15)\n')
    elif 'ann_text = f"\\Delta:' in line:
        new_lines.append('            ann_text = rf"$\Delta$: {val:.3f}\n({marginal:+.3f})\n[{tag}]"\n' if 'marginal' in line else '            ann_text = rf"$\Delta$: {val:.3f}\n[{tag}]"\n')
    else:
        new_lines.append(line)

nb['cells'][25]['source'] = new_lines

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Cleaned Cell 25 warnings in notebook successfully!")
