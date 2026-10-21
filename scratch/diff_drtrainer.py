import json
import difflib

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'r', encoding='utf-8') as f:
    lodo_nb = json.load(f)

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb', 'r', encoding='utf-8') as f:
    main_nb = json.load(f)

lodo_trainer_src = "".join(lodo_nb['cells'][22]['source']).splitlines()
main_trainer_src = "".join(main_nb['cells'][68]['source']).splitlines()

diff = list(difflib.unified_diff(main_trainer_src, lodo_trainer_src, fromfile='main_model(6)_DRTrainer', tofile='main_model_lodo_DRTrainer', lineterm=''))

print(f"Diff lines count: {len(diff)}")
for line in diff[:100]:
    print(line)
if len(diff) > 100:
    print(f"... and {len(diff) - 100} more lines")
