"""
================================================================================
DIABETIC RETINOPATHY (DR) PIPELINE - PREPROCESSING MODULE
================================================================================
Core Research Focus: Deconstructing Domain Shift & Preprocessing Ablation.
Supported Preprocessing Modes:
  - Auto-crop ROI: Automatic circular fundus boundary segmentation to strip black margins.
  - Mode 1 (Baseline): ROI Auto-crop + Standard Resizing + Denoise.
  - Mode 2 (Ben Graham): ROI Auto-crop + Local Background Subtraction (Vascular enhancement).
  - Mode 3 (Green Channel + CLAHE): ROI Auto-crop + Green/Luminance Contrast Limited
    Adaptive Histogram Equalization (Microaneurysm enhancement - NO Gaussian blur at the end).
  - PyTorch torchvision transform builders for Train (heavy augmentations) and Eval.
================================================================================
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, Union, Optional, Callable


# ==============================================================================
# 1. AUTO-CROP REGION OF INTEREST (ROI)
# ==============================================================================

def crop_fundus_roi(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Automatically detects and crops the fundus region of interest (ROI)
    to remove redundant black background margins.
    
    Args:
        img: Input image as RGB numpy array (H, W, 3) or grayscale (H, W).
        tol: Grayscale intensity threshold tolerance for dark background (default 7).
        
    Returns:
        np.ndarray: Cropped bounding box region enclosing the retinal disk.
    """
    if img.ndim == 2:
        mask = img > tol
        if not np.any(mask):
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]
        
    # Multi-channel color image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img
    mask = gray > tol
    
    # Check if mask is completely empty (black image)
    if not np.any(mask):
        return img
        
    # Check channel-wise masks
    check_r = img[:, :, 0] > tol
    check_g = img[:, :, 1] > tol
    check_b = img[:, :, 2] > tol
    combined_mask = check_r | check_g | check_b
    
    if not np.any(combined_mask):
        return img
        
    row_mask = combined_mask.any(axis=1)
    col_mask = combined_mask.any(axis=0)
    
    cropped = img[np.ix_(row_mask, col_mask)]
    if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
        return img
        
    return cropped


# ==============================================================================
# 2. PREPROCESSING MODES (ABLATION STUDY)
# ==============================================================================

def preprocess_baseline(
    img: Union[np.ndarray, Image.Image],
    target_size: Optional[Tuple[int, int]] = (300, 300)
) -> np.ndarray:
    """
    Mode 1 (Baseline): Auto-crop ROI + Denoise + Standard Resize.
    
    Args:
        img: Input image as numpy array or PIL Image.
        target_size: Optional (width, height) to resize.
        
    Returns:
        np.ndarray: Preprocessed RGB image (uint8).
    """
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img.copy()
        
    # 1. Auto-crop black border
    cropped = crop_fundus_roi(img_np)
    
    # 2. Light median/bilateral denoising
    denoised = cv2.bilateralFilter(cropped, d=5, sigmaColor=30, sigmaSpace=30)
    
    # 3. Resize if target_size specified
    if target_size is not None:
        denoised = cv2.resize(denoised, target_size, interpolation=cv2.INTER_AREA)
        
    return denoised


def preprocess_ben_graham(
    img: Union[np.ndarray, Image.Image],
    target_size: Optional[Tuple[int, int]] = (300, 300),
    sigma_ratio: float = 30.0
) -> np.ndarray:
    """
    Mode 2 (Ben Graham): Auto-crop ROI + Local Background Subtraction.
    Highlights vasculature, hemorrhages, and microaneurysms by subtracting local Gaussian blur:
      I_out = 4 * I - 4 * GaussianBlur(I, sigma) + 128
      
    Args:
        img: Input image as numpy array or PIL Image.
        target_size: Optional (width, height) to resize.
        sigma_ratio: Ratio to determine Gaussian blur sigma from image dimension.
        
    Returns:
        np.ndarray: Preprocessed RGB image with normalized background (uint8).
    """
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img.copy()
        
    # 1. Auto-crop ROI
    cropped = crop_fundus_roi(img_np)
    
    if target_size is not None:
        cropped = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        
    # 2. Local background subtraction
    sigma = max(1, int(cropped.shape[1] / sigma_ratio))
    # Gaussian blur kernel size must be odd
    ksize = int(2 * np.ceil(2 * sigma) + 1)
    blurred = cv2.GaussianBlur(cropped, (ksize, ksize), sigmaX=sigma)
    
    # Weighted difference
    enhanced = cv2.addWeighted(cropped, 4.0, blurred, -4.0, 128.0)
    
    # Mask out outer black border circle cleanly
    h, w = enhanced.shape[:2]
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(w / 2), int(h / 2))
    radius = int(min(center[0], center[1]) * 0.98)
    cv2.circle(circle_mask, center, radius, 255, -1)
    
    result = np.zeros_like(enhanced)
    result[circle_mask == 255] = enhanced[circle_mask == 255]
    
    return np.clip(result, 0, 255).astype(np.uint8)


def preprocess_clahe_green(
    img: Union[np.ndarray, Image.Image],
    target_size: Optional[Tuple[int, int]] = (300, 300),
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Mode 3 (Green Channel + CLAHE):
    Extracts high-contrast green channel / LAB luminance and applies Contrast Limited
    Adaptive Histogram Equalization (CLAHE) to amplify subtle microaneurysms and exudates.
    CRITICAL RULE: Does NOT apply Gaussian blur at the end to preserve sharp microaneurysm edges.
    
    Args:
        img: Input image as numpy array or PIL Image.
        target_size: Optional (width, height) to resize.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Grid size for local histogram equalization (default 8x8).
        
    Returns:
        np.ndarray: Sharp, contrast-enhanced RGB fundus image (uint8).
    """
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img.copy()
        
    # 1. Auto-crop ROI
    cropped = crop_fundus_roi(img_np)
    
    if target_size is not None:
        cropped = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        
    # 2. Convert to LAB color space to equalize L channel while preserving chromatic balance
    lab = cv2.cvtColor(cropped, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 3. Apply CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl_channel = clahe.apply(l_channel)
    
    # 4. Merge back and convert to RGB
    limg = cv2.merge((cl_channel, a_channel, b_channel))
    rgb_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # 5. Extract and enhance Green channel contrast directly
    # Green channel has the highest optical absorption difference for retinal micro-lesions
    r, g, b = cv2.split(rgb_clahe)
    g_clahe = clahe.apply(g)
    
    # Reassemble balanced multi-channel representation
    enhanced = cv2.merge((r, g_clahe, b))
    
    # Explicitly NO Gaussian blur applied to retain edge sharpness of microaneurysms
    return np.clip(enhanced, 0, 255).astype(np.uint8)


# ==============================================================================
# 3. CALLABLE PREPROCESSING TRANSFORMS (FOR PYTORCH DATASET)
# ==============================================================================

class FundusPreprocessorTransform:
    """
    Callable transform wrapper integrating selectable Preprocessing Ablation modes
    into PyTorch torchvision dataset workflows.
    """
    def __init__(
        self,
        mode: str = "clahe",
        target_size: Optional[Tuple[int, int]] = (300, 300)
    ):
        """
        Args:
            mode: 'baseline', 'ben_graham', or 'clahe' / 'green_clahe'.
            target_size: Target (width, height) resolution tuple.
        """
        self.mode = mode.lower()
        self.target_size = target_size

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        if self.mode in ["ben_graham", "bengraham", "mode2"]:
            proc_np = preprocess_ben_graham(img, self.target_size)
        elif self.mode in ["clahe", "green_clahe", "clahe_green", "mode3"]:
            proc_np = preprocess_clahe_green(img, self.target_size)
        else:
            proc_np = preprocess_baseline(img, self.target_size)
            
        return Image.fromarray(proc_np)


# ==============================================================================
# 4. PYTORCH AUGMENTATION & EVALUATION PIPELINES
# ==============================================================================

def get_torch_transforms(
    mode: str = "clahe",
    global_size: int = 300,
    patch_size: int = 128,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Tuple[Callable, Callable, Callable, Callable]:
    """
    Builds torchvision transform pipelines for both Global and Local MIL branches.
    
    Args:
        mode: Preprocessing ablation mode ('baseline', 'ben_graham', 'clahe').
        global_size: Resolution for global full-image context (default 300).
        patch_size: Resolution for local MIL patches (default 128).
        mean: Normalization channel means.
        std: Normalization channel standard deviations.
        
    Returns:
        Tuple: (global_train_tf, global_eval_tf, patch_train_tf, patch_eval_tf)
    """
    preproc_global = FundusPreprocessorTransform(mode=mode, target_size=(global_size, global_size))
    preproc_patch = FundusPreprocessorTransform(mode=mode, target_size=(patch_size, patch_size))
    
    # 1. Global Training Transforms (Heavy spatial + color augmentation)
    global_train_tf = transforms.Compose([
        preproc_global,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=35),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 2. Global Evaluation Transforms (Deterministic)
    global_eval_tf = transforms.Compose([
        preproc_global,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 3. Patch Training Transforms (Moderate patch-level augmentations)
    patch_train_tf = transforms.Compose([
        preproc_patch,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 4. Patch Evaluation Transforms (Deterministic)
    patch_eval_tf = transforms.Compose([
        preproc_patch,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return global_train_tf, global_eval_tf, patch_train_tf, patch_eval_tf
