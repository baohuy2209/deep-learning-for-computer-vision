import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Let's check:
# 1. Early stopping in Cell 21
# 2. QWK warmup in ContinuousQWKLoss / CumulativeLinkModelQWKHead
# 3. LODO loop in Cell 23
# 4. Evaluation and figures in Cell 25-29

cell21 = "".join(nb['cells'][21]['source'])
cell23 = "".join(nb['cells'][23]['source'])
cell25 = "".join(nb['cells'][25]['source'])
cell27 = "".join(nb['cells'][27]['source'])
cell28 = "".join(nb['cells'][28]['source'])
cell29 = "".join(nb['cells'][29]['source'])

report = []

# Check EarlyStopping in Cell 21:
if "val_qwk" in cell21 and "best_val_qwk" in cell21:
    report.append("Cell 21 EarlyStopping: Tracks val_qwk (PASSED)")
else:
    report.append("Cell 21 EarlyStopping: WARNING - might not be tracking val_qwk!")

# Check QWK warmup in Cell 21 / 23:
if "qwk_warmup_epochs" in cell23 and "TARGET_QWK_WEIGHT" in cell23:
    report.append("Cell 23 QWK Warmup: Uses TARGET_QWK_WEIGHT & QWK_WARMUP_EPOCHS (PASSED)")
else:
    report.append("Cell 23 QWK Warmup: WARNING")

# Check batch size in eval in Cell 23:
report.append("Cell 23 eval batch size lines:")
for line in cell23.splitlines():
    if "eval_batch_size" in line or "batch_size" in line:
        report.append(f"  {line.strip()}")

# Check Fig 7 in Cell 28:
report.append("Cell 28 Fig 7 ROC lines:")
for line in cell28.splitlines():
    if "fpr" in line or "plot_fig7" in line:
        report.append(f"  {line.strip()}")

# Check Fig 4 in Cell 27:
report.append("Cell 27 Fig 4 Waterfall lines:")
for line in cell27.splitlines():
    if "plot_fig4" in line or "ood_qwk" in line:
        report.append(f"  {line.strip()}")

with open('scratch/deep_audit_report.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(report))

print("Audit check done.")
