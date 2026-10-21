import json
import re

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Concatenate all code cells up to cell 26
all_code = []
for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        code = "".join(cell.get("source", []))
        all_code.append((idx, code))

# Check for PureWindowsPath
for idx, code in all_code:
    if "PureWindowsPath" in code:
        print(f"Cell {idx} uses PureWindowsPath")
    if "apply_green_channel_ablation" in code:
        print(f"Cell {idx} uses or defines apply_green_channel_ablation")
    if "tqdm" in code:
        print(f"Cell {idx} uses or defines tqdm")
    if "build_model" in code:
        print(f"Cell {idx} uses or defines build_model")
    if "num_workers" in code:
        print(f"Cell {idx} uses or defines num_workers")
    if "ChannelAttention" in code:
        print(f"Cell {idx} uses or defines ChannelAttention")
    if "SpatialAttention" in code:
        print(f"Cell {idx} uses or defines SpatialAttention")
    if "NonLocalBlock2D" in code:
        print(f"Cell {idx} uses or defines NonLocalBlock2D")
    if "QuadrantTokenAggregator" in code:
        print(f"Cell {idx} uses or defines QuadrantTokenAggregator")
    if "LocalMILBranch" in code:
        print(f"Cell {idx} uses or defines LocalMILBranch")
    if "C_NAVY" in code:
        print(f"Cell {idx} uses or defines C_NAVY")
    if "C_PURPLE" in code:
        print(f"Cell {idx} uses or defines C_PURPLE")
