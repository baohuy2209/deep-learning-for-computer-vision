"""
================================================================================
DIABETIC RETINOPATHY (DR) PIPELINE - TRAINER & METRICS EVALUATION MODULE
================================================================================
Core Research Focus: Domain Shift Quantification & Cross-Domain Generalization Benchmarking.
Key Metrics Tracked:
  - Quadratic Weighted Kappa (QWK): Primary evaluation metric for ordinal agreement.
  - Delta_QWK: Domain Shift Drop = (In-Domain Val QWK) - (OOD Test QWK).
  - Within-1-Grade Accuracy: Proportion of predictions within +/-1 ordinal severity level.
  - Referable DR Sensitivity & Specificity: Clinical screening metric (Grades 0-1 vs Grades 2-4).
  - Per-Grade Recall & Confusion Matrix.
  - Rank Monotonicity Violations: Measures logical ordinal ranking consistency on cumulative rank probabilities.
  - FundusGradCAM: Explainable AI with memory-safe hook management.
================================================================================
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score, recall_score

# Setup module logger
logger = logging.getLogger("DRTrainer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s | %(asctime)s | DRTrainer] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ==============================================================================
# 1. COMPREHENSIVE MEDICAL & ORDINAL METRICS CALCULATOR
# ==============================================================================

def calculate_dr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5
) -> Dict[str, Any]:
    """
    Computes clinical and statistical evaluation metrics for 5-grade DR classification.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    # 1. Quadratic Weighted Kappa (QWK)
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    if np.isnan(qwk):
        qwk = 0.0
        
    # 2. Accuracy
    acc = float(accuracy_score(y_true, y_pred))
    
    # 3. Within-1-Grade Accuracy: Tolerance metric for clinical viability
    within_1 = float(np.mean(np.abs(y_true - y_pred) <= 1))
    
    # 4. Confusion Matrix & Per-grade Recalls
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_grade_recall = {}
    for c in range(num_classes):
        denom = cm[c, :].sum()
        recall = float(cm[c, c] / denom) if denom > 0 else 0.0
        per_grade_recall[f"Recall_Grade_{c}"] = recall
        
    # 5. Referable DR Metrics (Non-referable: 0,1 vs Referable: 2,3,4)
    y_true_ref = (y_true >= 2).astype(int)
    y_pred_ref = (y_pred >= 2).astype(int)
    
    cm_ref = confusion_matrix(y_true_ref, y_pred_ref, labels=[0, 1])
    tn, fp, fn, tp = cm_ref.ravel() if cm_ref.size == 4 else (0, 0, 0, 0)
    
    ref_sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    ref_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    return {
        "QWK": qwk,
        "Accuracy": acc,
        "Within_1_Grade_Acc": within_1,
        "Referable_Sensitivity": ref_sensitivity,
        "Referable_Specificity": ref_specificity,
        "Confusion_Matrix": cm.tolist(),
        **per_grade_recall
    }


def rank_monotonicity_violations(rank_probs: np.ndarray) -> float:
    """
    Computes the proportion of patient samples that violate the strict clinical
    monotonicity constraint on cumulative rank probabilities P(y > k).
    
    Constraint: P(y > 0) >= P(y > 1) >= P(y > 2) >= P(y > 3) (Strictly non-increasing)
    """
    if rank_probs is None or len(rank_probs) == 0:
        return 0.0
    rank_probs = np.asarray(rank_probs, dtype=np.float64)
    if rank_probs.ndim == 1:
        rank_probs = rank_probs.reshape(1, -1)
        
    diffs = np.diff(rank_probs, axis=1)
    violations = np.any(diffs > 1e-4, axis=1)
    return float(np.mean(violations))


# ==============================================================================
# 2. GRAD-CAM WITH MEMORY-SAFE HOOK MANAGEMENT
# ==============================================================================

class FundusGradCAM:
    """
    Grad-CAM implementation with explicit hook registration and removal handles
    to prevent memory leaks across multiple evaluation batches.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hook_handles = []

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h_f = self.target_layer.register_forward_hook(forward_hook)
        h_b = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h_f, h_b])

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def generate_cam(
        self,
        global_img: torch.Tensor,
        local_patches: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        self._register_hooks()
        try:
            self.model.eval()
            self.model.zero_grad()
            
            outputs = self.model(global_img, local_patches)
            probs = outputs.get("probs", None)
            
            if target_class is None:
                if probs is not None:
                    target_class = int(torch.argmax(probs, dim=1).item())
                else:
                    target_class = int(outputs["preds"].item())
                    
            if "logits" in outputs:
                score = outputs["logits"][0, target_class]
            elif "severity_score" in outputs:
                score = outputs["severity_score"][0]
            else:
                score = probs[0, target_class]
                
            score.backward(retain_graph=False)
            
            if self.gradients is None or self.activations is None:
                cam = np.zeros((300, 300), dtype=np.float32)
                return cam, target_class
                
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            activations = self.activations[0]
            
            for i in range(len(pooled_gradients)):
                activations[i, :, :] *= pooled_gradients[i]
                
            heatmap = torch.mean(activations, dim=0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            max_val = np.max(heatmap)
            if max_val > 0:
                heatmap /= max_val
                
            return heatmap, target_class
        finally:
            self.remove_hooks()


# ==============================================================================
# 3. PYTORCH TRAINING & EVALUATION ENGINE
# ==============================================================================

class DRTrainer:
    """
    High-Performance PyTorch Training & Evaluation Engine for DR Multi-Scale Pipeline.
    Supports Mixed Precision (AMP), QWK Loss Warmup, Gradient Clipping, and LODO Evaluation.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        grad_clip: float = 1.0,
        warmup_epochs: int = 5,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        device_str = str(device)
        self.device_type = "cuda" if "cuda" in device_str else "cpu"
        self.use_amp = use_amp and (self.device_type == "cuda")
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_amp)

    def train_epoch(self, dataloader: DataLoader, epoch: int = 1) -> Dict[str, float]:
        """Runs one full training epoch over the dataset with QWK warmup."""
        self.model.train()
        
        # Apply QWK loss linear warmup
        if hasattr(self.model, "set_qwk_warmup"):
            self.model.set_qwk_warmup(epoch, total_warmup_epochs=self.warmup_epochs)
            
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch in dataloader:
            global_img = batch["global_img"].to(self.device, non_blocking=True)
            local_patches = batch["local_patches"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)
            
            # Skip degenerate batch size 1 in training mode
            if len(targets) <= 1:
                continue
                
            self.optimizer.zero_grad()
            
            # Forward pass with Automatic Mixed Precision (AMP)
            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                outputs = self.model(global_img, local_patches)
                loss = self.model.compute_loss(outputs, targets, class_weights=self.class_weights)
                
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * len(targets)
            all_preds.extend(outputs["preds"].detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
            
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        n_samples = len(all_targets)
        epoch_loss = total_loss / max(1, n_samples)
        metrics = calculate_dr_metrics(all_targets, all_preds)
        metrics["Loss"] = epoch_loss
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """Evaluates the model on validation or out-of-domain test sets."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_rank_probs = []
        
        for batch in dataloader:
            global_img = batch["global_img"].to(self.device, non_blocking=True)
            local_patches = batch["local_patches"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)
            
            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                outputs = self.model(global_img, local_patches)
                loss = self.model.compute_loss(outputs, targets, class_weights=self.class_weights)
                
            total_loss += loss.item() * len(targets)
            all_preds.extend(outputs["preds"].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            if "rank_probs" in outputs:
                all_rank_probs.extend(outputs["rank_probs"].cpu().numpy())
                
        n_samples = len(all_targets)
        eval_loss = total_loss / max(1, n_samples)
        metrics = calculate_dr_metrics(all_targets, all_preds)
        metrics["Loss"] = eval_loss
        
        if all_rank_probs:
            rank_probs_np = np.array(all_rank_probs)
            metrics["Monotonicity_Violations"] = rank_monotonicity_violations(rank_probs_np)
            
        return metrics, np.array(all_targets), np.array(all_preds)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 20,
        model_name: str = "dr_model"
    ) -> Dict[str, Any]:
        """Executes multi-epoch training with checkpointing on validation QWK."""
        best_val_qwk = -1.0
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_qwk": [], "val_qwk": [],
            "val_within_1": [], "val_ref_sens": []
        }
        
        best_checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pth")
        logger.info(f"Beginning Training for {epochs} Epochs on Device: {self.device}")
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader, epoch=epoch)
            val_metrics, _, _ = self.evaluate(val_loader)
            elapsed = time.time() - t0
            
            history["train_loss"].append(train_metrics["Loss"])
            history["val_loss"].append(val_metrics["Loss"])
            history["train_qwk"].append(train_metrics["QWK"])
            history["val_qwk"].append(val_metrics["QWK"])
            history["val_within_1"].append(val_metrics["Within_1_Grade_Acc"])
            history["val_ref_sens"].append(val_metrics["Referable_Sensitivity"])
            
            logger.info(
                f"[Epoch {epoch:02d}/{epochs:02d} - {elapsed:.1f}s] "
                f"Train Loss: {train_metrics['Loss']:.4f} | Train QWK: {train_metrics['QWK']:.4f} || "
                f"Val Loss: {val_metrics['Loss']:.4f} | Val QWK: {val_metrics['QWK']:.4f} | "
                f"Val Within-1: {val_metrics['Within_1_Grade_Acc']*100:.1f}% | "
                f"Val Ref-Sens: {val_metrics['Referable_Sensitivity']*100:.1f}%"
            )
            
            if val_metrics["QWK"] > best_val_qwk:
                best_val_qwk = val_metrics["QWK"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_qwk": best_val_qwk,
                    "val_metrics": val_metrics
                }, best_checkpoint_path)
                logger.info(f"  -> Saved new best model checkpoint to '{best_checkpoint_path}' (Val QWK: {best_val_qwk:.4f})")
                
        test_metrics = {}
        if test_loader is not None and os.path.exists(best_checkpoint_path):
            logger.info("Loading best model weights for Out-Of-Domain (OOD) Test evaluation...")
            ckpt = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            test_metrics, _, _ = self.evaluate(test_loader)
            
            delta_qwk = best_val_qwk - test_metrics["QWK"]
            test_metrics["Delta_QWK"] = delta_qwk
            
            logger.info(
                f"=== OOD TEST RESULTS ===\n"
                f"OOD Test QWK: {test_metrics['QWK']:.4f} | "
                f"In-Domain Val QWK: {best_val_qwk:.4f} | "
                f"Delta_QWK: {delta_qwk:.4f} | "
                f"OOD Within-1: {test_metrics['Within_1_Grade_Acc']*100:.1f}% | "
                f"OOD Ref-Sens: {test_metrics['Referable_Sensitivity']*100:.1f}%"
            )
            
        return {
            "best_val_qwk": best_val_qwk,
            "best_checkpoint": best_checkpoint_path,
            "history": history,
            "ood_test_metrics": test_metrics
        }
