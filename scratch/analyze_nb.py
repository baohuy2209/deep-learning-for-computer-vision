import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    src = "".join(cell.get('source', []))
    lines = src.splitlines()
    print(f"--- Cell {idx} ({ctype}) [{len(lines)} lines] ---")
    if lines:
        print("  First line:", lines[0][:80])
        # print some key lines or functions defined
        for l in lines:
            ls = l.strip()
            if ls.startswith('def ') or ls.startswith('class ') or 'LODO_' in ls or 'DATA_ROOT' in ls or 'BATCH_SIZE' in ls or 'QWK_WARMUP' in ls or 'TARGET_QWK' in ls or 'early_stopping' in ls or 'fit(' in ls:
                print("    *", ls[:90])
