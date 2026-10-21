import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb', 'r', encoding='utf-8') as f:
    nb_local = json.load(f)

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb', 'r', encoding='utf-8') as f:
    nb_colab = json.load(f)

print("=== CELL 0 (Overview Markdown) ===")
print("--- Local Cell 0 ---")
print(''.join(nb_local['cells'][0]['source']))
print("\n--- Colab Cell 0 ---")
print(''.join(nb_colab['cells'][0]['source']))

print("\n" + "="*50 + "\n")
print("=== Section 3 Header Markdown ===")
print("--- Local Cell 26 ---")
print(''.join(nb_local['cells'][26]['source']))
print("\n--- Colab Cell 31 ---")
print(''.join(nb_colab['cells'][31]['source']))

print("\n" + "="*50 + "\n")
print("=== Section 3.1.5 & NonLocal/Quadrant (Local 31, 32 vs Colab 36, 37) ===")
print("--- Local Cell 31 ---")
print(''.join(nb_local['cells'][31]['source']))
print("--- Local Cell 32 ---")
print(''.join(nb_local['cells'][32]['source']))
print("\n--- Colab Cell 36 ---")
print(''.join(nb_colab['cells'][36]['source']))
print("--- Colab Cell 37 ---")
print(''.join(nb_colab['cells'][37]['source']))

print("\n" + "="*50 + "\n")
print("=== GlobalBranch (Local 33, 34 vs Colab 38, 39) ===")
print("--- Local Cell 33 ---")
print(''.join(nb_local['cells'][33]['source']))
print("--- Local Cell 34 ---")
print(''.join(nb_local['cells'][34]['source']))
print("\n--- Colab Cell 38 ---")
print(''.join(nb_colab['cells'][38]['source']))
print("--- Colab Cell 39 ---")
print(''.join(nb_colab['cells'][39]['source']))

