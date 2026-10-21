import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', encoding='utf-8'))

def find_cell(query):
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        if query in src:
            return i, src
    return -1, ""

queries = [
    "def compute_fov_mask",
    "def dynamic_collate_fn",
    "def calibrate_and_evaluate_screening",
    "class FundusGradCAM",
    "def compute_quantitative_xai",
    "def visualize_champion_model_results",
    "def plot_fig3_slope_chart",
    "def plot_fig6_latent_axis",
    "def plot_fig7_roc",
    "def plot_fig8_real_xai",
]

for q in queries:
    idx, src = find_cell(q)
    print(f"Query '{q}' found in cell {idx} (length {len(src)})")
