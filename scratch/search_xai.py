import json
import glob

print("Searching for XAI and mask references across notebooks...")
notebooks = glob.glob("notebooks/**/*.ipynb", recursive=True)
for nb_path in notebooks:
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        for i, cell in enumerate(nb.get("cells", [])):
            src = "".join(cell.get("source", []))
            for kw in ["idrid_mask", "ground_truth", "pointing_game", "iou", "gradcam", "grad_cam", "mask_dir", "segmentation", "xai"]:
                if kw in src.lower():
                    print(f"[{nb_path}] Cell {i} matched '{kw}':")
                    matching_lines = [l.strip() for l in src.split("\n") if kw in l.lower()]
                    for ml in matching_lines[:3]:
                        print(f"    {ml[:100]}")
                    break
    except Exception as e:
        print(f"Error reading {nb_path}: {e}")
