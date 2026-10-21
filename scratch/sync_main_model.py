import json
import ast

notebook_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 68: DRTrainer
cell_68 = nb["cells"][68]
src_68 = "".join(cell_68["source"])

# Replace self.lr_scheduler.step(val_loss) with self.lr_scheduler.step(val_qwk)
assert "self.lr_scheduler.step(val_loss)" in src_68, "Expected self.lr_scheduler.step(val_loss) in cell 68"
src_68 = src_68.replace("self.lr_scheduler.step(val_loss)", "self.lr_scheduler.step(val_qwk)")

# Also ensure default qwk_warmup_epochs in __init__ is 15
src_68 = src_68.replace("qwk_warmup_epochs: int = 5", "qwk_warmup_epochs: int = 15")

cell_68["source"] = [line + "\n" for line in src_68.splitlines()]
# remove trailing newline on the last line if necessary to preserve format
if cell_68["source"]:
    cell_68["source"][-1] = cell_68["source"][-1].rstrip("\n")

# 2. Update Cell 70: trainer.fit(...) call
cell_70 = nb["cells"][70]
src_70 = "".join(cell_70["source"])

old_fit_call = """main_history = main_trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    epochs=EPOCHS,
    model_name=model_tag,
    patience_early_stopping=PATIENCE_EARLY_STOPPING,
    resume_checkpoint_path=resume_ckpt_path
)"""

new_fit_call = """main_history = main_trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    epochs=EPOCHS,
    model_name=model_tag,
    patience_early_stopping=PATIENCE_EARLY_STOPPING,
    patience_reduce_lr=PATIENCE_REDUCE_LR,
    reduce_lr_factor=REDUCE_LR_FACTOR,
    min_lr=MIN_LR,
    resume_checkpoint_path=resume_ckpt_path
)"""

assert old_fit_call in src_70, "Expected old_fit_call in cell 70"
src_70 = src_70.replace(old_fit_call, new_fit_call)

cell_70["source"] = [line + "\n" for line in src_70.splitlines()]
if cell_70["source"]:
    cell_70["source"][-1] = cell_70["source"][-1].rstrip("\n")

# 3. Update Cell 81: Evaluation & completion marking
cell_81 = nb["cells"][81]
src_81 = "".join(cell_81["source"])

old_cell_81_tail = """    calib_save_path = os.path.join(CHECKPOINT_DIR, f"{model_tag}_screening_calibration.csv")
    df_screening_calib.to_csv(calib_save_path, index=False)
    print(f"-> Saved screening calibration report to: '{calib_save_path}'\\n")"""

new_cell_81_tail = """    calib_save_path = os.path.join(CHECKPOINT_DIR, f"{model_tag}_screening_calibration.csv")
    df_screening_calib.to_csv(calib_save_path, index=False)
    print(f"-> Saved screening calibration report to: '{calib_save_path}'\\n")

# 6. Mark checkpoint as completed (is_completed=True)
if os.path.exists(best_ckpt_path):
    try:
        ckpt_meta = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        ckpt_meta["is_completed"] = True
        ckpt_meta["test_metrics"] = test_metrics
        torch.save(ckpt_meta, best_ckpt_path)
        print(f"-> Marked checkpoint '{best_ckpt_path}' as completed (is_completed=True).")
    except Exception as e:
        print(f"[WARN] Failed to write is_completed flag to {best_ckpt_path}: {e}")

val_qwk_score = float(val_metrics.get("QWK", 0.0))
test_qwk_score = float(test_metrics.get("QWK", 0.0))
print(f"\\n>>> [FINAL CHAMPION MODEL EVALUATION SUMMARY]:")
print(f"    In-Domain Val QWK : {val_qwk_score:.4f}")
print(f"    Held-Out Test QWK : {test_qwk_score:.4f}")
print(f"    Generalization Gap (Delta QWK): {val_qwk_score - test_qwk_score:.4f}")"""

assert old_cell_81_tail in src_81, "Expected old_cell_81_tail in cell 81"
src_81 = src_81.replace(old_cell_81_tail, new_cell_81_tail)

cell_81["source"] = [line + "\n" for line in src_81.splitlines()]
if cell_81["source"]:
    cell_81["source"][-1] = cell_81["source"][-1].rstrip("\n")

# Save updated notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("[SUCCESS] Synchronized main_model (6).ipynb cells 68, 70, and 81.")
