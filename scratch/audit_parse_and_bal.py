import json
import sys
import difflib

sys.stdout.reconfigure(encoding='utf-8')

lodo_nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', encoding='utf-8'))
merge_nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb', encoding='utf-8'))
nb6 = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb', encoding='utf-8'))

def get_src(nb, idx):
    return ''.join(nb['cells'][idx]['source'])

print("="*80)
print("1. COMPARING parse_patient_and_side: LODO cell 6 vs MERGE cell 5")
print("="*80)
lodo_p = get_src(lodo_nb, 6)
merge_p = get_src(merge_nb, 5)

# Extract parse_patient_and_side definition from both
def extract_func(src, name):
    start = src.find(f"def {name}")
    if start == -1: return "NOT FOUND"
    end = src.find("\ndef ", start+1)
    if end == -1: end = src.find("\nclass ", start+1)
    if end == -1: end = len(src)
    return src[start:end].strip()

f_lodo_parse = extract_func(lodo_p, "parse_patient_and_side")
f_merge_parse = extract_func(merge_p, "parse_patient_and_side")

diff = list(difflib.unified_diff(
    f_merge_parse.splitlines(keepends=True),
    f_lodo_parse.splitlines(keepends=True),
    fromfile='merge-dataset.ipynb',
    tofile='main_model_lodo.ipynb'
))
print("Diff lines in parse_patient_and_side:", len(diff))
for l in diff:
    print(l, end='')

print("\n" + "="*80)
print("2. COMPARING balancing logic: LODO cell 8 vs MERGE cell 11")
print("="*80)
f_lodo_bal = extract_func(get_src(lodo_nb, 8), "create_balanced_training_pool")
f_merge_bal = extract_func(get_src(merge_nb, 11), "create_balanced_train_pool")
print("LODO balancing signature:", f_lodo_bal.split('\n')[0])
print("MERGE balancing signature:", f_merge_bal.split('\n')[0])

diff_bal = list(difflib.unified_diff(
    f_merge_bal.splitlines(keepends=True),
    f_lodo_bal.splitlines(keepends=True),
    fromfile='merge-dataset.ipynb',
    tofile='main_model_lodo.ipynb'
))
print("Diff lines in balancing:", len(diff_bal))
for l in diff_bal[:40]:
    print(l, end='')
