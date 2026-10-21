import json

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
output = []

for idx in [13, 14, 15, 17, 18, 20]:
    if idx < len(cells):
        output.append(f"\n{'='*30} CELL {idx} ({cells[idx]['cell_type']}) {'='*30}\n")
        output.append("".join(cells[idx].get("source", [])))

with open("scratch/audit_part2.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Part 2 written")
