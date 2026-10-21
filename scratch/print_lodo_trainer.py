import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'r', encoding='utf-8') as f:
    lodo_nb = json.load(f)

with open('scratch/lodo_trainer.txt', 'w', encoding='utf-8') as out:
    out.write("=== CELL 22 of main_model_lodo.ipynb ===\n")
    out.write("".join(lodo_nb['cells'][22]['source']))
    out.write("\n\n=== CELL 25 of main_model_lodo.ipynb ===\n")
    out.write("".join(lodo_nb['cells'][25]['source']))

print("Successfully written to scratch/lodo_trainer.txt")
