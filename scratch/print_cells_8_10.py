import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def load_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

nb4 = load_nb("notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb")
nb8 = load_nb("notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb")

print("--- Cell 8 in (4) ---")
print("".join(nb4["cells"][8].get("source", [])))

print("\n--- Cell 8 in (8) ---")
print("".join(nb8["cells"][8].get("source", [])))

print("\n--- Cell 10 in (4) ---")
print("".join(nb4["cells"][10].get("source", [])))

print("\n--- Cell 10 in (8) ---")
print("".join(nb8["cells"][10].get("source", [])))
