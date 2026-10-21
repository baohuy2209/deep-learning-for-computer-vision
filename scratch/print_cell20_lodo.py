import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

lodo_nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', encoding='utf-8'))
src20 = ''.join(lodo_nb['cells'][20]['source'])
print(src20)
