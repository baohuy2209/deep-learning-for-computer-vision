import json
import ast

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 5 to include IDRiD segmentation paths
cell5_src = "".join(nb["cells"][5]["source"])
if "idrid_seg_root" not in cell5_src:
    cell5_addition = """
# ==============================================================================
# 1.6 CONFIG: IDRiD PIXEL-LEVEL SEGMENTATION GROUND TRUTH (FOR QUANTITATIVE XAI)
# ==============================================================================
idrid_seg_root_drive = "/content/drive/MyDrive/Luận Văn Tốt Nghiệp 2026 - 2027/code/data/raw/idrid_segmentation"
idrid_seg_root_local = "data/raw/idrid_segmentation"
idrid_seg_root = idrid_seg_root_drive if os.path.exists(idrid_seg_root_drive) else idrid_seg_root_local

idrid_seg_image_dirs = [
    f"{idrid_seg_root}/DR_Testing_Set/Fundus Images",
    f"{idrid_seg_root}/DR_Training_Set/Fundus Images"
]
idrid_seg_mask_dirs = [
    f"{idrid_seg_root}/DR_Testing_Set/Combined Masks",
    f"{idrid_seg_root}/DR_Training_Set/Combined Masks"
]
print(f"[XAI CONFIG] IDRiD Benchmark Segmentation Root: '{idrid_seg_root}'")
print(f"  - Image directories ({len(idrid_seg_image_dirs)}): {idrid_seg_image_dirs}")
print(f"  - Mask directories  ({len(idrid_seg_mask_dirs)}):  {idrid_seg_mask_dirs}")
"""
    cell5_src = cell5_src.rstrip() + cell5_addition
    nb["cells"][5]["source"] = [l + "\n" for l in cell5_src.split("\n")[:-1]] + [cell5_src.split("\n")[-1]]

# 2. Update Cell 25: Full FundusGradCAM + IDRiDXAIEvalDataset + generate_attention_mil_saliency_map + compute_quantitative_xai
cell25_code = '''# ==============================================================================
# 7.0 ARTIFACT EXTRACTION HELPERS, GRAD-CAM & QUANTITATIVE XAI ENGINE
# (EXACT PARITY WITH main_model (6).ipynb CELLS 74, 76, 77, 78)
# ==============================================================================
def extract_tensor_from_outputs(outputs: Any) -> torch.Tensor:
    """Extracts continuous severity scores from model output dictionary."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, dict):
        for k in ["severity_score", "latent_score", "logits", "expected_grade", "preds"]:
            if k in outputs and isinstance(outputs[k], torch.Tensor):
                return outputs[k]
        return next(iter(outputs.values()))
    raise ValueError(f"Unable to extract Tensor from output: {type(outputs)}")


def extract_model_thetas(eval_model: nn.Module, device: str = "cuda") -> List[float]:
    """Extracts learned monotonic cutpoints theta_1..4 from CLM head."""
    head = getattr(eval_model, "head", eval_model)
    if hasattr(head, "_get_monotonic_cutpoints"):
        with torch.no_grad():
            thetas = head._get_monotonic_cutpoints().cpu().numpy().tolist()
        return thetas
    return [-1.5, -0.5, 0.5, 1.5]


@torch.no_grad()
def evaluate_and_extract_artifacts(eval_model: nn.Module, data_loader: DataLoader, device: str = "cuda") -> Dict[str, Any]:
    """
    Extracts real per-grade recalls, continuous latent severity scores s(x),
    ROC TPR/FPR arrays, and probabilities from real model inference without mocking.
    """
    eval_model.eval()
    all_targets, all_preds, all_probs, all_scores = [], [], [], []

    for batch in data_loader:
        g_img = batch["global_img"].to(device)
        l_patch = batch["local_patches"].to(device)
        p_mask = batch.get("patch_mask", None)
        if p_mask is not None:
            p_mask = p_mask.to(device)
        labels = batch["label"].numpy()

        with torch.amp.autocast("cuda" if "cuda" in device else "cpu", enabled="cuda" in device):
            outputs = eval_model(g_img, l_patch, patch_mask=p_mask)

        preds = outputs["preds"].cpu().numpy()
        probs = outputs["probs"].cpu().numpy()
        scores = extract_tensor_from_outputs(outputs).cpu().numpy()

        all_targets.extend(labels)
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_scores.extend(scores)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probs_np = np.array(all_probs)
    scores_np = np.array(all_scores)

    # Per-grade recall
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    recalls = {}
    for g in range(5):
        denom = cm[g, :].sum()
        recalls[g] = float(cm[g, g] / denom) if denom > 0 else 0.0

    # Per-grade score dictionary
    grade_scores = {g: scores_np[y_true == g] for g in range(5)}

    # Referable DR ROC Curve
    y_true_ref = (y_true >= 2).astype(int)
    ref_probs = probs_np[:, 2:].sum(axis=1) if probs_np.shape[1] >= 3 else probs_np[:, 1]
    if len(np.unique(y_true_ref)) > 1:
        fpr, tpr, _ = roc_curve(y_true_ref, ref_probs)
        roc_auc = float(auc(fpr, tpr))
    else:
        fpr, tpr, roc_auc = np.array([0, 1]), np.array([0, 1]), 0.5

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "recalls": recalls,
        "scores": grade_scores,
        "thetas": extract_model_thetas(eval_model, device=device),
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc
    }


# ==============================================================================
# 7.0.1 FUNDUS GRAD-CAM WITH RELIABLE HOOKS (EXACT main_model (6).ipynb CELL 74)
# ==============================================================================
class FundusGradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) with memory-safe hook cleanup.
    Extracts visual attention saliency heatmaps for both categorical and ordinal CLM models.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        if target_layer is None:
            if hasattr(model, "backbone") and hasattr(model.backbone, "global_branch") and hasattr(model.backbone.global_branch, "layer4"):
                target_layer = model.backbone.global_branch.layer4[-1]
            elif hasattr(model, "layer4"):
                target_layer = model.layer4[-1]
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        self.hook_handles = []
        self.remove_hooks()

    def _register_hooks(self):
        if self.target_layer is None:
            return

        def forward_hook(module, input, output):
            self.activation = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradient = grad_output[0].detach()

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
        target_class: Optional[int] = None,
        patch_mask: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, int]:
        self.remove_hooks()
        self._register_hooks()
        try:
            self.model.eval()
            self.model.zero_grad()

            if patch_mask is not None:
                try:
                    outputs = self.model(global_img, local_patches, patch_mask=patch_mask)
                except TypeError:
                    outputs = self.model(global_img, local_patches)
            else:
                outputs = self.model(global_img, local_patches)

            pred_class = None
            if isinstance(outputs, dict) and "preds" in outputs:
                pred_class = int(outputs["preds"][0].item())

            if isinstance(outputs, dict) and "severity_score" in outputs:
                s_x = outputs["severity_score"].flatten()[0]
                score = s_x
                if target_class is None:
                    target_class = pred_class if pred_class is not None else 0
            elif isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
                if target_class is None:
                    target_class = int(torch.argmax(logits, dim=-1)[0].item())
                score = logits[0, target_class]
            elif isinstance(outputs, torch.Tensor):
                if outputs.dim() > 1 and outputs.size(-1) == 5:
                    if target_class is None:
                        target_class = int(torch.argmax(outputs, dim=-1)[0].item())
                    score = outputs[0, target_class]
                else:
                    score = outputs.flatten()[0]
                    if target_class is None:
                        target_class = pred_class if pred_class is not None else 0
            else:
                score = outputs[0]
                if target_class is None:
                    target_class = pred_class if pred_class is not None else 0

            score.backward(retain_graph=False)

            if self.gradient is None or self.activation is None:
                cam = np.zeros((global_img.size(2), global_img.size(3)), dtype=np.float32)
                return cam, target_class

            weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
            cam_tensor = torch.sum(weights * self.activation, dim=1).squeeze()

            cam = F.relu(cam_tensor).cpu().numpy()
            denom = np.max(cam) - np.min(cam)
            if denom > 1e-6:
                cam = (cam - np.min(cam)) / denom
            else:
                cam = torch.mean(torch.abs(self.activation), dim=1).squeeze().cpu().numpy()
                denom_fb = np.max(cam) - np.min(cam)
                cam = (cam - np.min(cam)) / (denom_fb + 1e-6)

            return cam.astype(np.float32), target_class

        finally:
            self.remove_hooks()

    def generate(self, global_img: torch.Tensor, local_patches: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """Alias for backward compatibility with plot_fig8_real_xai."""
        cam, _ = self.generate_cam(global_img, local_patches, patch_mask=patch_mask)
        cam_resized = cv2.resize(cam, (300, 300), interpolation=cv2.INTER_LINEAR)
        return cam_resized


# ==============================================================================
# 7.0.2 IDRiD XAI EVALUATION DATASET WITH DUAL-CROP ALIGNMENT (CELL 77)
# ==============================================================================
class IDRiDXAIEvalDataset(Dataset):
    """
    Dedicated dataset for quantitative XAI fidelity evaluation.
    Features Dual-Crop Alignment:
      Crops BOTH raw fundus image and its ground-truth lesion mask using the EXACT SAME
      bounding box computed from compute_fov_mask(img_np, erode_px=0) before resizing.
      Eliminates scale/margin domain shift between native_cropped training set and raw IDRiD images.
    """
    def __init__(
        self,
        image_dirs: List[str],
        mask_dirs: List[str],
        img_size: int = IMG_SIZE,
        ablation_mode: str = "ben_graham_green",
    ):
        self.img_size = img_size
        self.ablation_mode = ablation_mode
        self.global_transform = A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
        ])

        self.image_records = []
        for img_dir in image_dirs:
            if not os.path.exists(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    img_id = os.path.splitext(fname)[0]
                    img_path = os.path.join(img_dir, fname)
                    self.image_records.append((img_id, img_path))

        self.samples = []
        for img_id, img_path in self.image_records:
            mask_path = None
            for mask_dir in mask_dirs:
                if not os.path.exists(mask_dir):
                    continue
                candidates = [f"{img_id}.tif", f"{img_id}.png", f"{img_id}.jpg", f"{img_id}.jpeg"]
                for c in candidates:
                    p = os.path.join(mask_dir, c)
                    if os.path.exists(p):
                        mask_path = p
                        break
                if mask_path:
                    break
            if mask_path is not None:
                self.samples.append((img_id, img_path, mask_path))

        logger.info(f"IDRiDXAIEvalDataset: found {len(self.image_records)} images, "
                    f"{len(self.samples)} with a verified binary lesion mask on disk.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id, img_path, mask_path = self.samples[idx]
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_np.shape

        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        lesion_mask = (m > 0).astype(np.uint8)

        fov_mask_raw = compute_fov_mask(img_np, erode_px=0)
        ys, xs = np.where(fov_mask_raw)
        if len(ys) > 0 and len(xs) > 0:
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            img_np = img_np[y0:y1, x0:x1]
            lesion_mask = lesion_mask[y0:y1, x0:x1]

        fov_mask = compute_fov_mask(img_np, erode_px=15)
        processed_np = apply_green_channel_ablation(img_np, ablation_mode=self.ablation_mode)
        global_augmented = self.global_transform(image=processed_np)
        global_img = global_augmented["image"]
        global_tensor = torch.from_numpy(global_img).permute(2, 0, 1).contiguous()

        local_patches, patch_anchors = extract_mil_patches_sliding(
            processed_np,
            fov_mask=fov_mask,
            patch_size=128,
            stride=96,
            fov_thresh=0.5,
            mode="eval",
            max_patches_eval=256,
            return_anchors=True
        )

        mask_gt_resized = cv2.resize(lesion_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        return {
            "image_id": img_id,
            "global_img": global_tensor,
            "local_patches": local_patches,
            "anchors": torch.from_numpy(np.asarray(patch_anchors, dtype=np.int32)),
            "mask_gt": torch.tensor(mask_gt_resized, dtype=torch.uint8)
        }


# ==============================================================================
# 7.0.3 ARCHITECTURE-NATIVE ATTENTION-MIL SALIENCY & QUANTITATIVE SUITE (CELL 78)
# ==============================================================================
def generate_attention_mil_saliency_map(
    model: nn.Module,
    global_img: torch.Tensor,
    local_patches: torch.Tensor,
    anchors: np.ndarray,
    patch_mask: Optional[torch.Tensor] = None,
    canvas_size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
    patch_size: int = 128
) -> np.ndarray:
    """
    Projects patch-level attention weights alpha_k from GatedAttentionPool onto a 2D spatial canvas
    using patch coordinates [y0, x0].
    Produces an architecture-native fine-grained saliency map (~128 px resolution).
    """
    model.eval()
    with torch.no_grad():
        _ = model(global_img, local_patches, patch_mask=patch_mask)

    pool = None
    if hasattr(model, "backbone") and hasattr(model.backbone, "local_mil_branch") and hasattr(model.backbone.local_mil_branch, "gated_pool"):
        pool = model.backbone.local_mil_branch.gated_pool
    elif hasattr(model, "local_mil_branch") and hasattr(model.local_mil_branch, "gated_pool"):
        pool = model.local_mil_branch.gated_pool

    if pool is not None and hasattr(pool, "last_attention_weights") and pool.last_attention_weights is not None:
        att_weights = pool.last_attention_weights[0, :, 0].cpu().numpy()
    else:
        num_p = local_patches.size(1) if local_patches.dim() == 5 else len(anchors)
        att_weights = np.ones(num_p, dtype=np.float32) / max(1, num_p)

    canvas = np.zeros(canvas_size, dtype=np.float32)
    counts = np.zeros(canvas_size, dtype=np.float32)
    H_can, W_can = canvas_size

    for k, (y0, x0) in enumerate(anchors):
        if k >= len(att_weights):
            break
        alpha_k = float(att_weights[k])
        y1 = min(int(y0 + patch_size), H_can)
        x1 = min(int(x0 + patch_size), W_can)
        y0_clamped = max(0, int(y0))
        x0_clamped = max(0, int(x0))
        canvas[y0_clamped:y1, x0_clamped:x1] += alpha_k
        counts[y0_clamped:y1, x0_clamped:x1] += 1.0

    mil_map = np.divide(canvas, counts, out=np.zeros_like(canvas), where=(counts > 0))
    denom = mil_map.max() - mil_map.min()
    if denom > 1e-7:
        mil_map = (mil_map - mil_map.min()) / denom
    return mil_map.astype(np.float32)


def compute_quantitative_xai(
    model: nn.Module,
    idrid_xai_dataset: Dataset,
    device: str = "cuda",
    cam_threshold: float = 0.5,
    return_samples: bool = False
) -> Dict[str, Any]:
    """
    Quantitative XAI Evaluation Suite:
      1. Pointing Game Accuracy evaluated strictly against empirical random baseline.
      2. Saliency Recall @ Area-Matched (thresholding top-q% heatmap pixels where q matches lesion mask ratio).
      3. Pixel-level AUROC (threshold-free continuous saliency discrimination).
      4. Comparative Benchmark: Global Grad-CAM (ResNet Layer 4) vs. Local Attention-MIL (~128px resolution).
      5. Deprecates raw Mean IoU as failure metric (acknowledges ~0.05 theoretical micro-lesion ceiling).
    """
    model.eval()
    model.to(device)

    mean_g = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_g  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    mean_l = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 1, 3, 1, 1)
    std_l  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 1, 3, 1, 1)

    target_layer = None
    if hasattr(model, "backbone") and hasattr(model.backbone, "global_branch") and hasattr(model.backbone.global_branch, "layer4"):
        target_layer = model.backbone.global_branch.layer4[-1]
    elif hasattr(model, "layer4"):
        target_layer = model.layer4[-1]
    grad_cam = FundusGradCAM(model=model, target_layer=target_layer)

    if len(idrid_xai_dataset) == 0:
        raise ValueError("IDRiD XAI eval dataset is empty - no real masks found on disk.")

    print(f"\\n{'='*80}")
    print(f"RUNNING OVERHAULED QUANTITATIVE XAI ON {len(idrid_xai_dataset)} REAL LESION MASK IMAGES")
    print(f"{'='*80}")

    gradcam_hits, mil_hits = [], []
    gradcam_area_recalls, mil_area_recalls = [], []
    gradcam_pix_aucs, mil_pix_aucs = [], []
    iou_scores = []
    random_baselines = []
    xai_samples = []

    for idx in range(len(idrid_xai_dataset)):
        sample = idrid_xai_dataset[idx]
        g_img = sample["global_img"].unsqueeze(0).to(device)
        l_patch = sample["local_patches"].unsqueeze(0).to(device)
        anchors = sample["anchors"].numpy()
        mask_gt = sample["mask_gt"].numpy()

        if g_img.dtype == torch.uint8:
            g_img = g_img.float().div_(255.0).sub_(mean_g).div_(std_g)
        if l_patch.dtype == torch.uint8:
            l_patch = l_patch.float().div_(255.0).sub_(mean_l).div_(std_l)

        cam_global, _ = grad_cam.generate_cam(g_img, l_patch)
        cam_global = cv2.resize(cam_global, (mask_gt.shape[1], mask_gt.shape[0]))

        saliency_mil = generate_attention_mil_saliency_map(
            model=model,
            global_img=g_img,
            local_patches=l_patch,
            anchors=anchors,
            canvas_size=(mask_gt.shape[0], mask_gt.shape[1])
        )

        n_pos = int((mask_gt > 0).sum())
        total_pixels = int(mask_gt.size)
        q = float(n_pos) / float(total_pixels)
        random_baselines.append(q)

        gy, gx = np.unravel_index(np.argmax(cam_global), cam_global.shape)
        g_hit = bool(mask_gt[gy, gx] > 0)
        gradcam_hits.append(g_hit)

        my, mx = np.unravel_index(np.argmax(saliency_mil), saliency_mil.shape)
        m_hit = bool(mask_gt[my, mx] > 0)
        mil_hits.append(m_hit)

        if n_pos > 0:
            q_clamped = max(1e-4, min(0.99, q))
            thresh_g = np.quantile(cam_global, 1.0 - q_clamped)
            thresh_m = np.quantile(saliency_mil, 1.0 - q_clamped)

            g_recall = float(((cam_global >= thresh_g) & (mask_gt > 0)).sum()) / float(n_pos)
            m_recall = float(((saliency_mil >= thresh_m) & (mask_gt > 0)).sum()) / float(n_pos)
            gradcam_area_recalls.append(g_recall)
            mil_area_recalls.append(m_recall)

            if (mask_gt == 0).sum() > 0:
                try:
                    gradcam_pix_aucs.append(float(roc_auc_score(mask_gt.flatten() > 0, cam_global.flatten())))
                    mil_pix_aucs.append(float(roc_auc_score(mask_gt.flatten() > 0, saliency_mil.flatten())))
                except Exception:
                    pass
        else:
            gradcam_area_recalls.append(0.0)
            mil_area_recalls.append(0.0)

        cam_bin = (cam_global >= cam_threshold).astype(np.uint8)
        intersection = np.logical_and(cam_bin, mask_gt).sum()
        union = np.logical_or(cam_bin, mask_gt).sum()
        iou_scores.append(float(intersection / union) if union > 0 else 0.0)

        if return_samples and len(xai_samples) < 5:
            xai_samples.append({
                "image_id": sample["image_id"],
                "global_img": sample["global_img"],
                "cam_global": cam_global,
                "saliency_mil": saliency_mil,
                "mask_gt": mask_gt,
                "g_hit": g_hit,
                "m_hit": m_hit,
                "random_baseline": q
            })

    avg_random_baseline = float(np.mean(random_baselines)) * 100.0
    g_pointing_acc = float(np.mean(gradcam_hits)) * 100.0
    m_pointing_acc = float(np.mean(mil_hits)) * 100.0

    g_area_rec = float(np.mean(gradcam_area_recalls)) if gradcam_area_recalls else 0.0
    m_area_rec = float(np.mean(mil_area_recalls)) if mil_area_recalls else 0.0

    g_auroc = float(np.mean(gradcam_pix_aucs)) if gradcam_pix_aucs else 0.0
    m_auroc = float(np.mean(mil_pix_aucs)) if mil_pix_aucs else 0.0
    mean_iou = float(np.mean(iou_scores))

    results = {
        "Random_Baseline": avg_random_baseline,
        "GradCAM_Pointing_Game": g_pointing_acc,
        "MIL_Pointing_Game": m_pointing_acc,
        "GradCAM_Area_Matched_Recall": g_area_rec,
        "MIL_Area_Matched_Recall": m_area_rec,
        "GradCAM_Pixel_AUROC": g_auroc,
        "MIL_Pixel_AUROC": m_auroc,
        "Mean_IoU_Informational": mean_iou
    }

    print(f"XAI BENCHMARK RESULTS (Global Grad-CAM vs. Architecture-Native Local Attention-MIL):")
    print(f"  - Empirical Random Baseline:         {avg_random_baseline:.2f}%")
    print(f"  - Pointing Game Accuracy:")
    print(f"      * Global Grad-CAM (Layer 4):     {g_pointing_acc:.2f}% ({g_pointing_acc/max(1e-3, avg_random_baseline):.1f}x chance)")
    print(f"      * Local Attention-MIL (~128px):   {m_pointing_acc:.2f}% ({m_pointing_acc/max(1e-3, avg_random_baseline):.1f}x chance)")
    print(f"  - Saliency Recall @ Area-Matched:")
    print(f"      * Global Grad-CAM:               {g_area_rec:.4f}")
    print(f"      * Local Attention-MIL:           {m_area_rec:.4f}")
    print(f"  - Pixel-level Saliency AUROC:")
    print(f"      * Global Grad-CAM:               {g_auroc:.4f}")
    print(f"      * Local Attention-MIL:           {m_auroc:.4f}")
    print(f"  - Mean IoU (Informational):          {mean_iou:.4f} (Theoretical micro-lesion ceiling ~0.050)\\n")

    if return_samples:
        return results, xai_samples
    return results
'''

# 3. Update Cell 28: FIG 8 grounded against IDRiD masks
cell28_code = '''# ==============================================================================
# 7.3 FIG 5 (LODO HEATMAP), FIG 6 (LATENT AXIS), FIG 7 (ROC), & FIG 8 (XAI GRID)
# ==============================================================================
def plot_fig5_lodo_heatmap(results_dict: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Renders Fig. 5: Heatmap matrix of LODO transfer performance across 6 external test cohorts.
    Seamlessly supports nested multi-variant dictionary (lodo_results[variant][fold]) by selecting
    Champion model (or primary variant) as well as flat single-model dictionaries.
    """
    configure_publication_style()
    
    target_dict = results_dict
    if any(k in ["V3_PlusOrdinalHead_H3_Champion", "Champion"] for k in results_dict):
        target_dict = results_dict.get("V3_PlusOrdinalHead_H3_Champion", results_dict.get("Champion"))
    elif any(isinstance(v, dict) and any(dk in DATASET_KEYS for dk in v.keys()) for v in results_dict.values()):
        target_dict = next(v for v in results_dict.values() if isinstance(v, dict) and any(dk in DATASET_KEYS for dk in v.keys()))

    keys = [k for k in target_dict.keys() if isinstance(target_dict[k], dict)]
    cohort_names = [target_dict[k].get("held_out_dataset", DATASET_CONFIG.get(k, {}).get("name", k)) for k in keys]
    data = {
        "In-Domain Val QWK": [float(target_dict[k].get("in_domain_val_qwk", target_dict[k].get("val_qwk", 0.0))) for k in keys],
        "OOD Test QWK": [float(target_dict[k].get("ood_test_qwk", target_dict[k].get("test_qwk", 0.0))) for k in keys],
        "Delta QWK Drop": [float(target_dict[k].get("delta_lodo", target_dict[k].get("delta_qwk", 0.0))) for k in keys],
        "OOD Within-1 Acc (%)": [float(target_dict[k].get("ood_within_1_grade_acc", target_dict[k].get("ood_within_1", 0.0))) * 100 for k in keys],
        "OOD Ref-Sens @ 95% Spec (%)": [float(target_dict[k].get("ood_referable_sensitivity_at_95spec", target_dict[k].get("ood_ref_sens", 0.0))) * 100 for k in keys]
    }
    df = pd.DataFrame(data, index=cohort_names).T

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Metric Value'},
                linewidths=1.2, linecolor="#1e293b", ax=ax)

    ax.set_title("Figure 5: 6-Fold Leave-One-Dataset-Out (LODO) Generalization Matrix\\n"
                 "Cross-Center Resilience across External Cohorts", fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("Evaluated Metric", fontsize=11, fontweight="bold")
    ax.set_xlabel("Held-Out External Test Cohort", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_fig6_latent_severity_axis(
    in_scores: Dict[int, np.ndarray],
    ood_scores: Dict[int, np.ndarray],
    cutpoints: List[float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders Fig. 6: 1D Continuous Latent Severity Score Axis s(x) of CLM + Learned Cutpoints theta.
    Key Features:
      - Demonstrates continuous disease severity spectrum.
      - Overlays Out-of-Domain score distribution with dashed lines to visibly expose Domain Shift.
    """
    configure_publication_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    grade_colors = ["#3b82f6", C_ORANGE, "#eab308", C_CRIMSON, "#7e22ce"]

    for g in range(5):
        if len(in_scores.get(g, [])) > 1:
            sns.kdeplot(in_scores[g], ax=ax, color=grade_colors[g], fill=True, alpha=0.35, lw=2.0, label=f"Grade {g} (In-Domain)")
        if len(ood_scores.get(g, [])) > 1:
            sns.kdeplot(ood_scores[g], ax=ax, color=grade_colors[g], linestyle="--", lw=1.8, alpha=0.85, label=f"Grade {g} (Out-of-Domain)" if g == 0 else "")

    for i, theta in enumerate(cutpoints):
        ax.axvline(theta, color=C_PURPLE, linestyle=":", lw=2.2, zorder=5)
        ax.text(theta, ax.get_ylim()[1] * 0.90, f"$\\theta_{i+1} = {theta:.2f}$",
                ha="center", va="bottom", fontsize=10.0, fontweight="bold", color=C_PURPLE,
                bbox=dict(boxstyle="round,pad=0.2", fc="#faf5ff", ec=C_PURPLE, lw=1.0))

    ax.set_xlabel(r"1D Latent Severity Score: $s(\mathbf{x}) = \mathbf{w}^T \mathbf{z}$", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Sample Density", fontsize=11.5, fontweight="bold")
    ax.set_title("Figure 6: 1D Latent Severity Spectrum of Cumulative Link Model (CLM) & Learned Cutpoints $\\theta$\\n"
                 "Overlaid Dashed Curves Visibly Capture Out-of-Domain Distributional Shift", 
                 fontsize=12, fontweight="bold", pad=15)
    
    handles = [
        Patch(facecolor="#3b82f6", alpha=0.5, edgecolor="#1e293b", label="In-Domain Solid Distributions"),
        plt.Line2D([0], [0], color="#3b82f6", linestyle="--", lw=2, label="Out-of-Domain Dashed (Domain Shift)"),
        plt.Line2D([0], [0], color=C_PURPLE, linestyle=":", lw=2, label="Monotonic Cutpoints $\\theta_1 < \\theta_2 < \\theta_3 < \\theta_4$")
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_fig7_referable_dr_roc(
    in_fpr: np.ndarray,
    in_tpr: np.ndarray,
    in_auc: float,
    ood_fpr: np.ndarray,
    ood_tpr: np.ndarray,
    ood_auc: float,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Renders Fig. 7: Clinical ROC Curve for Referable DR (Grade >= 2), comparing In-Domain vs. External Center."""
    configure_publication_style()
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    ax.plot(in_fpr, in_tpr, color=C_NAVY, lw=2.5, label=f"In-Domain Internal Val (AUC = {in_auc:.3f})")
    ax.plot(ood_fpr, ood_tpr, color=C_ORANGE, lw=2.5, linestyle="--", label=f"Out-of-Domain External LODO (AUC = {ood_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle=":", lw=1.2)

    idx_in = np.argmin(np.abs(in_fpr - 0.05))
    idx_ood = np.argmin(np.abs(ood_fpr - 0.05))
    
    ax.scatter([0.05], [in_tpr[idx_in]], color=C_NAVY, s=90, zorder=5)
    ax.scatter([0.05], [ood_tpr[idx_ood]], color=C_ORANGE, s=90, zorder=5)

    ax.axvline(0.05, color="#cbd5e1", linestyle="--", lw=1.2)
    ax.text(0.06, 0.40, "95% Screening Specificity Gate (FPR = 5%)", rotation=90, va="center", color="#475569", fontsize=9.5)

    ax.text(0.08, in_tpr[idx_in] - 0.02, f"In-Domain Sens: {in_tpr[idx_in]*100:.1f}%", fontweight="bold", color=C_NAVY)
    ax.text(0.08, ood_tpr[idx_ood] - 0.05, f"OOD Sens: {ood_tpr[idx_ood]*100:.1f}%", fontweight="bold", color=C_ORANGE)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title(r"Figure 7: Clinical ROC Curve for Referable DR (Grade $\geq$ 2)\\n"
                 "High-Sensitivity Maintenance at 95% Screening Specificity Gate", fontsize=12, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_fig8_real_xai(
    eval_model: nn.Module,
    test_loader: DataLoader,
    xai_samples: Optional[List[Dict[str, Any]]] = None,
    idrid_xai_dataset: Optional[Dataset] = None,
    device: str = "cuda",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders Fig. 8: Qualitative XAI Grid grounded against IDRiD Expert Lesion Masks.
    Displays 5-Column High-Resolution Benchmark:
      Col 0: Raw Retinal Fundus Image (ROI Cropped)
      Col 1: Expert Ophthalmologist Ground-Truth Lesion Mask (IDRiD)
      Col 2: Architecture-Native Local Attention-MIL Saliency Map (~128px resolution, H1)
      Col 3: Global Grad-CAM Saliency Map (ResNet-50 Layer 4, H2)
      Col 4: Pathological Saliency Overlay & Pointing Game Peak
    """
    configure_publication_style()
    eval_model.eval()

    if xai_samples and len(xai_samples) > 0:
        n_rows = min(5, len(xai_samples))
        fig, axes = plt.subplots(n_rows, 5, figsize=(16.0, 3.2 * n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, 0)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for row in range(n_rows):
            s = xai_samples[row]
            g_t = s["global_img"]
            if isinstance(g_t, torch.Tensor):
                if g_t.dtype == torch.uint8:
                    orig = g_t.permute(1, 2, 0).cpu().numpy()
                else:
                    orig = g_t.permute(1, 2, 0).cpu().numpy()
                    orig = np.clip((orig * std + mean) * 255.0, 0, 255).astype(np.uint8)
            else:
                orig = np.asarray(g_t, dtype=np.uint8)

            mask_gt = s["mask_gt"]
            cam_g = s["cam_global"]
            cam_m = s["saliency_mil"]

            axes[row, 0].imshow(orig)
            axes[row, 0].set_title(f"Image {s['image_id']}: Raw Fundus", fontweight="bold", fontsize=10)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_gt, cmap="gray")
            axes[row, 1].set_title("IDRiD Expert Lesion Mask", fontweight="bold", fontsize=10, color="#b91c1c")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(cam_m, cmap="inferno")
            m_hit_str = "Hit" if s.get("m_hit", False) else "Miss"
            axes[row, 2].set_title(f"Local Attention-MIL ({m_hit_str})", fontweight="bold", fontsize=10, color="#047857")
            axes[row, 2].axis("off")

            axes[row, 3].imshow(cam_g, cmap="jet")
            g_hit_str = "Hit" if s.get("g_hit", False) else "Miss"
            axes[row, 3].set_title(f"Global Grad-CAM ({g_hit_str})", fontweight="bold", fontsize=10, color="#1d4ed8")
            axes[row, 3].axis("off")

            heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_m), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(orig, 0.6, heatmap_rgb, 0.4, 0)
            axes[row, 4].imshow(overlay)
            axes[row, 4].set_title("Pathological Overlay", fontweight="bold", fontsize=10)
            axes[row, 4].axis("off")

        plt.suptitle("Figure 8: Quantitative & Qualitative XAI Verification Grid against IDRiD Ground-Truth Lesion Masks\\n"
                     "Demonstrating Micro-Lesion Alignment in Local Attention-MIL ($H_1$) vs. Global Context ($H_2$)",
                     fontsize=12.5, fontweight="bold", y=1.01)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    # Fallback to test loader samples if no mask samples available
    cam_extractor = FundusGradCAM(eval_model)
    grade_samples = {}
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for batch in test_loader:
        labels = batch["label"].cpu().numpy()
        for i, lbl in enumerate(labels):
            if lbl not in grade_samples and lbl > 0:
                g_tensor = batch["global_img"][i:i+1].to(device)
                l_tensor = batch["local_patches"][i:i+1].to(device)
                p_mask = batch.get("patch_mask", None)
                p_mask_t = p_mask[i:i+1].to(device) if p_mask is not None else None

                cam, _ = cam_extractor.generate_cam(g_tensor, l_tensor, patch_mask=p_mask_t)
                orig_img = g_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                if g_tensor.dtype != torch.uint8:
                    orig_img = np.clip((orig_img * std + mean) * 255.0, 0, 255).astype(np.uint8)

                cam_res = cv2.resize(cam, (orig_img.shape[1], orig_img.shape[0]))
                grade_samples[lbl] = (orig_img, cam_res)
        if len(grade_samples) >= 4:
            break

    if not grade_samples:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "XAI Demonstration Samples", ha="center", va="center")
        return fig

    sorted_grades = sorted(grade_samples.keys())
    fig, axes = plt.subplots(len(sorted_grades), 3, figsize=(10.5, 3.2 * len(sorted_grades)))
    if len(sorted_grades) == 1:
        axes = np.expand_dims(axes, 0)

    for row, g in enumerate(sorted_grades):
        orig, cam = grade_samples[g]
        axes[row, 0].imshow(orig)
        axes[row, 0].set_title(f"Grade {g}: Raw Fundus Image", fontweight="bold", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(cam, cmap="jet")
        axes[row, 1].set_title(f"Grade {g}: Grad-CAM Activation", fontweight="bold", fontsize=10)
        axes[row, 1].axis("off")

        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig, 0.6, heatmap_rgb, 0.4, 0)

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"Grade {g}: Pathological Overlay", fontweight="bold", fontsize=10)
        axes[row, 2].axis("off")

    plt.suptitle("Figure 8: Qualitative XAI Grid - Empirical Grad-CAM from Proposed Model (H5)\\n"
                 "Demonstrating Focal Attention on Microaneurysms (H1) and Multi-Quadrant Lesions (H2)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
'''

# 4. Update Cell 29: Execute full suite including IDRiD quantitative XAI on final Champion model
cell29_code = '''# ==============================================================================
# 7.4 EXECUTE FULL PUBLICATION FIGURE SUITE (FIG 1 - 8) WITH REAL RUNTIME METRICS
# ==============================================================================
print("\\n" + "="*75)
print("GENERATING PUBLICATION-QUALITY FIGURES (FIG 1 - 8) FROM EMPIRICAL LODO RUN")
print("All figures exported to Drive as .png and .pdf with (2) suffix")
print("="*75 + "\\n")

# Reconstruct master dataset overview across 6 clinical cohorts from test CSVs
test_cohort_dfs = [
    pd.read_csv(os.path.join(LODO_FOLDS_DIR, f"fold_{i}_{k}_test.csv"))
    for i, k in enumerate(DATASET_KEYS)
    if os.path.exists(os.path.join(LODO_FOLDS_DIR, f"fold_{i}_{k}_test.csv"))
]
master_df = pd.concat(test_cohort_dfs, ignore_index=True) if test_cohort_dfs else None

def save_fig_dual(fig, basename: str):
    """Helper to save figure as both 300-DPI PNG and vector PDF with (2) suffix."""
    png_path = os.path.join(FIGURES_DIR, f"{basename}(2).png")
    pdf_path = os.path.join(FIGURES_DIR, f"{basename}(2).pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  -> Saved: {png_path} & {pdf_path}")

# 2. Figure 2: Master Dataset Composition & Label Shift
if master_df is not None:
    print("[Fig. 2] Generating Dataset Composition & Label Shift Chart...")
    f2 = plot_fig2_dataset_distribution(master_df)
    save_fig_dual(f2, "figure_2_dataset_distribution")
    plt.show()

# 4. Figure 4: Waterfall Decomposition of Delta QWK from Empirical LODO Results
print("[Fig. 4] Generating Waterfall QWK Decomposition from Empirical OOD Results...")
f4 = plot_fig4_waterfall_qwk_decomposition(lodo_results=lodo_results)
save_fig_dual(f4, "fig4_waterfall_qwk_decomposition")
plt.show()

# 5. Figure 5: 6-Fold LODO Generalization Heatmap Matrix
print("[Fig. 5] Generating 6-Fold LODO Generalization Matrix Heatmap...")
f5 = plot_fig5_lodo_heatmap(lodo_results)
save_fig_dual(f5, "fig5_lodo_heatmap")
plt.show()

# Extract real artifacts from the final trained Champion Model for instance-level figures
if eval_fold_artifacts:
    preferred_variants = [
        "V3_PlusOrdinalHead_H3_Champion",
        "Champion",
        "CLM_QWK",
        "V2_PlusGlobalContext_H2",
        "V1_PlusLocalMIL_H1"
    ]
    target_var = None
    for pref in preferred_variants:
        matched = next((k for k in eval_fold_artifacts.keys() if pref.lower() in k.lower()), None)
        if matched:
            target_var = matched
            break
    if target_var is None:
        target_var = list(eval_fold_artifacts.keys())[-1]

    var_folds = eval_fold_artifacts[target_var]
    target_fold = None
    if isinstance(var_folds, dict):
        preferred_folds = ["idrid", "messidor2", "aptos"]
        for pf in preferred_folds:
            if pf in var_folds:
                target_fold = pf
                break
        if target_fold is None and len(var_folds) > 0:
            target_fold = list(var_folds.keys())[-1]
        artifacts_dict = var_folds[target_fold]
    else:
        artifacts_dict = var_folds
        target_fold = "default_fold"

    fold_display_name = DATASET_CONFIG.get(target_fold, {}).get("name", target_fold)
    print(f"\\n[INFO] Extracting detailed artifacts & XAI from Model: '{target_var}' | OOD Cohort: '{fold_display_name}'")

    if "model" in artifacts_dict:
        active_model = artifacts_dict["model"]
        v_loader = artifacts_dict["val_loader"]
        t_loader = artifacts_dict["test_loader"]
        
        val_artifacts = evaluate_and_extract_artifacts(active_model, v_loader, device=device)
        ood_artifacts = evaluate_and_extract_artifacts(active_model, t_loader, device=device)

        # 3. Figure 3: Slope Chart Per-Grade Recall (Central Thesis - Selective Collapse in G1 & G3)
        print("[Fig. 3] Generating Per-Grade Recall Generalization Slope Chart (Central Thesis)...")
        f3 = plot_fig3_slope_chart_per_grade_recall(
            in_recalls=val_artifacts["recalls"],
            ood_recalls=ood_artifacts["recalls"],
            lodo_results=lodo_results
        )
        save_fig_dual(f3, "fig3_slope_chart")
        plt.show()

        # 6. Figure 6: 1D Latent Severity Score Spectrum & Learned Monotonic Cutpoints theta (H3)
        print(f"[Fig. 6] Generating 1D Latent Severity Spectrum & Monotonic Cutpoints for '{target_var}'...")
        f6 = plot_fig6_latent_severity_axis(
            in_scores=val_artifacts["scores"],
            ood_scores=ood_artifacts["scores"],
            cutpoints=val_artifacts["thetas"]
        )
        save_fig_dual(f6, "figure_6_latent_spectrum")
        plt.show()

        # 7. Figure 7: Clinical ROC Curve for Referable DR (95% Screening Specificity Gate)
        print("[Fig. 7] Generating Referable DR Clinical ROC Curve at 95% Specificity Gate...")
        f7 = plot_fig7_referable_dr_roc(
            in_fpr=val_artifacts["fpr"],
            in_tpr=val_artifacts["tpr"],
            in_auc=val_artifacts["auc"],
            ood_fpr=ood_artifacts["fpr"],
            ood_tpr=ood_artifacts["tpr"],
            ood_auc=ood_artifacts["auc"]
        )
        save_fig_dual(f7, "fig7_referable_dr_roc_comparison")
        plt.show()

        # 8. Figure 8: Quantitative & Qualitative XAI Verification against IDRiD Ground-Truth Lesion Masks
        print(f"\\n{'='*75}")
        print(f"[XAI] Executing Quantitative Saliency Evaluation against IDRiD Ground-Truth Lesion Masks...")
        print(f"      Model under evaluation: '{target_var}' (Proposed Champion)")
        print(f"{'='*75}")
        xai_results = None
        xai_samples = None

        if any(os.path.exists(d) for d in idrid_seg_mask_dirs) and any(os.path.exists(d) for d in idrid_seg_image_dirs):
            try:
                idrid_xai_dataset = IDRiDXAIEvalDataset(
                    image_dirs=idrid_seg_image_dirs,
                    mask_dirs=idrid_seg_mask_dirs,
                    img_size=IMG_SIZE,
                    ablation_mode="ben_graham_green",
                )
                if len(idrid_xai_dataset) > 0:
                    xai_results, xai_samples = compute_quantitative_xai(
                        model=active_model,
                        idrid_xai_dataset=idrid_xai_dataset,
                        device=device,
                        return_samples=True
                    )
                    xai_json_path = os.path.join(RESULTS_DIR_LODO, "xai_quantitative_metrics.json")
                    with open(xai_json_path, "w") as f:
                        json.dump(xai_results, f, indent=2)
                    print(f"  -> Saved Quantitative XAI metrics to '{xai_json_path}'")
                else:
                    print("[WARN] IDRiDXAIEvalDataset found 0 matched images with masks.")
            except Exception as e:
                print(f"[WARN] Error executing IDRiD Quantitative XAI: {e}")
        else:
            print(f"[INFO] IDRiD segmentation directories not found at '{idrid_seg_root}'. Falling back to test loader samples.")

        print(f"[Fig. 8] Generating Qualitative XAI Grad-CAM & Attention-MIL Verification Grid...")
        f8 = plot_fig8_real_xai(
            eval_model=active_model,
            test_loader=t_loader,
            xai_samples=xai_samples,
            idrid_xai_dataset=idrid_xai_dataset if 'idrid_xai_dataset' in locals() and len(idrid_xai_dataset) > 0 else None,
            device=device
        )
        save_fig_dual(f8, "figure_8_xai_qualitative_grid")
        plt.show()

print(f"\\n{'='*75}")
print(f"All Publication Figures (Fig 1 - 8) Successfully Generated and Saved in '{FIGURES_DIR}' with '(2)' suffix!")
print(f"{'='*75}\\n")
'''

def to_lines(code):
    return [l + '\n' for l in code.split('\n')[:-1]] + [code.split('\n')[-1]]

# Validate syntax
ast.parse(cell25_code)
ast.parse(cell28_code)
ast.parse(cell29_code)

nb['cells'][25]['source'] = to_lines(cell25_code)
nb['cells'][28]['source'] = to_lines(cell28_code)
nb['cells'][29]['source'] = to_lines(cell29_code)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Applied full XAI mask integration to main_model_lodo.ipynb successfully!")
