import json
import sys
import difflib

sys.stdout.reconfigure(encoding='utf-8')

nb6 = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb', encoding='utf-8'))
nb8 = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', encoding='utf-8'))

src6 = ''.join(nb6['cells'][23]['source'])
src8 = ''.join(nb8['cells'][23]['source'])

def extract_func(src, func_name):
    start = src.find(f'def {func_name}')
    if start == -1: return 'NOT FOUND'
    end = src.find('\ndef ', start + 1)
    if end == -1: end = src.find('\nclass ', start + 1)
    if end == -1: end = len(src)
    return src[start:end]

f6 = extract_func(src6, 'extract_mil_patches_sliding')
f8 = extract_func(src8, 'extract_mil_patches_sliding')

print("=== FUNCTION IN main_model (6).ipynb ===")
print(f6)

print("\n=== DIFF (f8 -> f6) ===")
diff = difflib.unified_diff(
    f8.splitlines(keepends=True),
    f6.splitlines(keepends=True),
    fromfile='main_model (8)',
    tofile='main_model (6)'
)
sys.stdout.writelines(diff)
