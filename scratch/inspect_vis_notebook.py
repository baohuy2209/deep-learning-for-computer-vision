import json

with open('notebooks/diabetic_retinopathy/pipeline/model/visualize_main_model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    lines = src.splitlines()
    # print headers or key lines
    headers = [l for l in lines if l.startswith('#')]
    found = [l for l in lines if any(k in l.lower() for k in ['batch_size', 'backbone', 'image_size', 'device', 'grad_cam', 'checkpoint', 'vram', 'dataloader', 'model =', 'model('])]
    if headers or found:
        print(f"--- Cell {i} ({cell.get('cell_type')}) ---")
        if headers:
            print("  Header:", headers[0][:80])
        for l in found[:8]:
            print("    ", l.strip()[:100])
