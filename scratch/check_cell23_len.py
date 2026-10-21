import json

nb6 = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb', encoding='utf-8'))
nb8 = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', encoding='utf-8'))

src6 = ''.join(nb6['cells'][23]['source'])
src8 = ''.join(nb8['cells'][23]['source'])

print(f"nb6 cell 23 len: {len(src6)}")
print(f"nb8 cell 23 len: {len(src8)}")
print("Does nb6 have cand_anchors:", "cand_anchors" in src6)
print("Does nb8 have cand_anchors:", "cand_anchors" in src8)
