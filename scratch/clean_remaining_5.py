import json

targets = [
    "notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb"
]

for path in targets:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 37 line 1
    cell37 = nb["cells"][37]["source"]
    for i, line in enumerate(cell37):
        if "Added per advisor review:" in line:
            cell37[i] = line.replace("Added per advisor review:", "Architectural Context:")

    # Cell 40 line 8
    cell40 = nb["cells"][40]["source"]
    for i, line in enumerate(cell40):
        if "added per advisor" in line:
            cell40[i] = line.replace(" - added per advisor", "")

    # Cell 64 line 110
    cell64 = nb["cells"][64]["source"]
    for i, line in enumerate(cell64):
        if "(HD_4 Part 1):" in line:
            cell64[i] = line.replace(" (HD_4 Part 1):", ":")

    # Cell 78 line 80
    cell78 = nb["cells"][78]["source"]
    for i, line in enumerate(cell78):
        if "(HD_4)" in line:
            cell78[i] = line.replace(" (HD_4)", "")

    # Cell 81 line 30
    cell81 = nb["cells"][81]["source"]
    for i, line in enumerate(cell81):
        if "(HD_4)..." in line:
            cell81[i] = line.replace(" (HD_4)...", "...")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Updated remaining lines in {path}!")
