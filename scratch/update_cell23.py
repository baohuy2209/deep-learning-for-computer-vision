import json
import ast

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

src23 = "".join(nb["cells"][23]["source"])

target = """        if is_fold_completed:
            print(f"[SKIP COMPLETED] Fold {fold_idx + 1} ({held_out_name}) for {variant_name} ALREADY COMPLETED!")
            if saved_metrics:
                print(f"  -> In-Domain Val QWK: {saved_metrics.get('in_domain_val_qwk', 'N/A')} | OOD Test QWK: {saved_metrics.get('ood_test_qwk', 'N/A')}")
                lodo_results[variant_name][held_out_key] = saved_metrics
            continue"""

replacement = """        if is_fold_completed:
            print(f"[SKIP COMPLETED] Fold {fold_idx + 1} ({held_out_name}) for {variant_name} ALREADY COMPLETED!")
            if saved_metrics:
                print(f"  -> In-Domain Val QWK: {saved_metrics.get('in_domain_val_qwk', 'N/A')} | OOD Test QWK: {saved_metrics.get('ood_test_qwk', 'N/A')}")
                lodo_results[variant_name][held_out_key] = saved_metrics
            
            # Preserve model & loaders in eval_fold_artifacts for publication visualization & XAI
            if (variant_name not in eval_fold_artifacts or held_out_key not in eval_fold_artifacts[variant_name]) and os.path.exists(best_checkpoint_path):
                try:
                    ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
                    fold_model.load_state_dict(ckpt["model_state_dict"])
                    eval_fold_artifacts[variant_name][held_out_key] = {
                        "model": fold_model,
                        "val_loader": val_loader,
                        "test_loader": test_loader
                    }
                except Exception as e:
                    pass
            continue"""

if target in src23:
    src23 = src23.replace(target, replacement)
    ast.parse(src23)
    nb["cells"][23]["source"] = [l + "\n" for l in src23.split("\n")[:-1]] + [src23.split("\n")[-1]]
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Cell 23 updated successfully with artifact preservation on skipped folds!")
else:
    print("Target block not found in Cell 23.")
