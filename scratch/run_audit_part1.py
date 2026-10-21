import json
import re

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

output = []
def log(msg):
    output.append(msg)

cells = nb["cells"]

# 1. Inspect Cell 4: Settings & Hyperparameters
log("### Cell 4: Config & Paths")
log("".join(cells[4].get("source", [])))

# 2. Inspect Cell 6: parse_patient_and_side
log("\n### Cell 6: Patient Parsing")
log("".join(cells[6].get("source", [])))

# 3. Inspect Cell 7: Dataset loading & Harmonization
log("\n### Cell 7: Dataset Loading")
log("".join(cells[7].get("source", [])))

# 4. Inspect Cell 8: create_balanced_training_pool
log("\n### Cell 8: Balancing")
log("".join(cells[8].get("source", [])))

# 5. Inspect Cell 10: compute_fov_mask & extract_mil_patches
log("\n### Cell 10: FOV & MIL Patches")
log("".join(cells[10].get("source", [])))

# 6. Inspect Cell 11: Dataset & DataLoader collate
log("\n### Cell 11: Dataset & Collate")
log("".join(cells[11].get("source", [])))

with open("scratch/audit_part1.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Part 1 written")
