import json
import difflib

nb6 = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb', encoding='utf-8'))
nb8 = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', encoding='utf-8'))

src6_77 = ''.join(nb6['cells'][77]['source'])
src8_77 = ''.join(nb8['cells'][77]['source'])

diff = list(difflib.unified_diff(
    src8_77.splitlines(keepends=True),
    src6_77.splitlines(keepends=True),
    fromfile='cell77_nb8',
    tofile='cell77_nb6'
))

print(f"Cell 77 diff lines: {len(diff)}")
for line in diff:
    print(line, end='')
