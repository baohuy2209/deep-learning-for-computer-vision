import os
import cv2
import torch
import numpy as np
import albumentations as A

# Test local IDRiD segmentation paths
idrid_seg_root_local = "data/raw/idrid_segmentation"
idrid_seg_image_dirs = [
    f"{idrid_seg_root_local}/DR_Testing_Set/Fundus Images",
    f"{idrid_seg_root_local}/DR_Training_Set/Fundus Images"
]
idrid_seg_mask_dirs = [
    f"{idrid_seg_root_local}/DR_Testing_Set/Combined Masks",
    f"{idrid_seg_root_local}/DR_Training_Set/Combined Masks"
]

print("Verifying files:")
for img_dir in idrid_seg_image_dirs:
    print(f"Img dir: {img_dir} - {len(os.listdir(img_dir))} files")
for mask_dir in idrid_seg_mask_dirs:
    print(f"Mask dir: {mask_dir} - {len(os.listdir(mask_dir))} files")
