import json
import ast

notebook_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 27: DRHeadTrainer
cell_27 = nb["cells"][27]
src_27 = "".join(cell_27["source"])

# Replace step-level warmup in train_epoch with clean epoch-level warmup in fit()
old_warmup_code = """        pbar = tqdm(dataloader, desc=f"Training [{self.head_name}] (Epoch {epoch}/{epochs})", leave=False, unit="batch")
        for batch_idx, batch in enumerate(pbar):
            # Step-level continuous dynamic linear warmup for QWK lambda
            if head is not None and hasattr(head, "set_qwk_weight") and hasattr(head, "target_qwk_weight"):
                current_step = (epoch - 1) * len(dataloader) + batch_idx
                step_progress = min(1.0, current_step / total_warmup_steps)
                current_weight = step_progress * head.target_qwk_weight
                head.set_qwk_weight(current_weight)"""

new_warmup_code = """        pbar = tqdm(dataloader, desc=f"Training [{self.head_name}] (Epoch {epoch}/{epochs})", leave=False, unit="batch")
        for batch_idx, batch in enumerate(pbar):"""

assert old_warmup_code in src_27, "Expected old_warmup_code in Cell 27"
src_27 = src_27.replace(old_warmup_code, new_warmup_code)

# Add epoch-level dynamic warmup in fit() loop matching DRTrainer
old_fit_loop_start = """        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            train_loss, train_qwk = self.train_epoch(train_loader, epoch, epochs)"""

new_fit_loop_start = """        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            head = getattr(self.model, "head", self.model)
            if hasattr(head, "set_qwk_weight") and hasattr(head, "target_qwk_weight"):
                progress = min(1.0, epoch / max(1, self.qwk_warmup_epochs))
                head.set_qwk_weight(progress * head.target_qwk_weight)

            train_loss, train_qwk = self.train_epoch(train_loader, epoch, epochs)"""

assert old_fit_loop_start in src_27, "Expected old_fit_loop_start in Cell 27"
src_27 = src_27.replace(old_fit_loop_start, new_fit_loop_start)

# Change self.lr_scheduler.step(val_loss) to self.lr_scheduler.step(val_qwk)
assert "self.lr_scheduler.step(val_loss)" in src_27, "Expected self.lr_scheduler.step(val_loss) in Cell 27"
src_27 = src_27.replace("self.lr_scheduler.step(val_loss)", "self.lr_scheduler.step(val_qwk)")

cell_27["source"] = [line + "\n" for line in src_27.splitlines()]
if cell_27["source"]:
    cell_27["source"][-1] = cell_27["source"][-1].rstrip("\n")

# 2. Update Cell 29: 5-round ablation loop
cell_29 = nb["cells"][29]
src_29 = "".join(cell_29["source"])

old_ablation_fit = """    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=ABLATION_EPOCHS,
        patience_early_stopping=PATIENCE_EARLY_STOP,
        resume_checkpoint_path=resume_path
    )"""

new_ablation_fit = """    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=ABLATION_EPOCHS,
        patience_early_stopping=PATIENCE_EARLY_STOP,
        patience_reduce_lr=4,
        reduce_lr_factor=0.5,
        min_lr=MIN_LR,
        resume_checkpoint_path=resume_path
    )"""

assert old_ablation_fit in src_29, "Expected old_ablation_fit in Cell 29"
src_29 = src_29.replace(old_ablation_fit, new_ablation_fit)

# Add completion flag saving to checkpoint in Cell 29
old_calib_export = """    # Save per-head screening calibration CSV to disk
    if "screening_calibration" in t_metrics and isinstance(t_metrics["screening_calibration"], pd.DataFrame):
        calib_csv_path = os.path.join(RESULTS_DIR_HEAD, f"screening_calibration_{head_name}.csv")
        t_metrics["screening_calibration"].to_csv(calib_csv_path, index=False)
        print(f"  [SAVED] Screening calibration table exported to: '{calib_csv_path}'")"""

new_calib_export = """    # Save per-head screening calibration CSV to disk
    if "screening_calibration" in t_metrics and isinstance(t_metrics["screening_calibration"], pd.DataFrame):
        calib_csv_path = os.path.join(RESULTS_DIR_HEAD, f"screening_calibration_{head_name}.csv")
        t_metrics["screening_calibration"].to_csv(calib_csv_path, index=False)
        print(f"  [SAVED] Screening calibration table exported to: '{calib_csv_path}'")

    # Mark checkpoint as completed to protect compute units
    if os.path.exists(best_ckpt_path):
        try:
            ckpt_meta = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            ckpt_meta["is_completed"] = True
            ckpt_meta["test_metrics"] = t_metrics
            torch.save(ckpt_meta, best_ckpt_path)
            print(f"  [COMPLETED] Marked '{best_ckpt_path}' as completed (is_completed=True).")
        except Exception as e:
            print(f"  [WARN] Failed to write is_completed flag to {best_ckpt_path}: {e}")"""

assert old_calib_export in src_29, "Expected old_calib_export in Cell 29"
src_29 = src_29.replace(old_calib_export, new_calib_export)

cell_29["source"] = [line + "\n" for line in src_29.splitlines()]
if cell_29["source"]:
    cell_29["source"][-1] = cell_29["source"][-1].rstrip("\n")

# Save updated notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("[SUCCESS] Synchronized main_model_head_ablation (3).ipynb cells 27 and 29.")
