import json

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if "idrid_seg_root" in src or "idrid_seg_mask_dirs" in src:
        print(f"Cell {i} has idrid_seg_root:")
        for line in src.split("\n"):
            if any(k in line for k in ["idrid_seg", "mask_dir", "image_dir"]):
                print("  ", line.encode("ascii", "backslashreplace").decode())
