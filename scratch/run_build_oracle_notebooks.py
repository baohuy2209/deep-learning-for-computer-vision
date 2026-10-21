"""
Execution script to generate the 6 oracle notebooks.
"""

import os
import json
import sys

from scratch.generate_oracle_notebooks import build_oracle_notebook

def main():
    # Load common cells from main_model_lodo
    lodo_path = 'notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb'
    with open(lodo_path, 'r', encoding='utf-8') as f:
        lodo_nb = json.load(f)

    common_cells = {
        'cell_env': ''.join(lodo_nb['cells'][2]['source']),
        'cell_plot_config': ''.join(lodo_nb['cells'][3]['source']),
        'cell_ablation_green': ''.join(lodo_nb['cells'][7]['source']),
        'cell_global_aug': ''.join(lodo_nb['cells'][8]['source']),
        'cell_dual_branch_ds': ''.join(lodo_nb['cells'][9]['source']),
        'cell_extract_patches': ''.join(lodo_nb['cells'][10]['source']),
        'cell_cbam': ''.join(lodo_nb['cells'][12]['source']),
        'cell_nonlocal': ''.join(lodo_nb['cells'][13]['source']),
        'cell_global_branch': ''.join(lodo_nb['cells'][14]['source']),
        'cell_local_branch': ''.join(lodo_nb['cells'][15]['source']),
        'cell_fusion': ''.join(lodo_nb['cells'][16]['source']),
        'cell_clm_qwk_loss': ''.join(lodo_nb['cells'][17]['source']),
        'cell_model_builder': ''.join(lodo_nb['cells'][18]['source']),
        'cell_metrics': ''.join(lodo_nb['cells'][20]['source']),
        'cell_trainer': ''.join(lodo_nb['cells'][21]['source'])
    }

    cohorts = [
        {
            'cohort_key': 'APTOS',
            'cohort_name': 'APTOS 2019',
            'csv_rel_path': 'aptos/aptos_total.csv',
            'id_col': 'id_code',
            'label_col': 'diagnosis',
            'id_suffix': '.png',
            'total_images': 3662,
            'lodo_fold_idx': 0,
            'lodo_slug': 'aptos',
            'filename': 'oracle_aptos.ipynb'
        },
        {
            'cohort_key': 'DDR',
            'cohort_name': 'DDR Dataset',
            'csv_rel_path': 'ddr/DR_grading.csv',
            'id_col': 'id_code',
            'label_col': 'diagnosis',
            'id_suffix': '.jpg',
            'total_images': 12522,
            'lodo_fold_idx': 3,
            'lodo_slug': 'ddr',
            'filename': 'oracle_ddr.ipynb'
        },
        {
            'cohort_key': 'DeepDRiD',
            'cohort_name': 'DeepDRiD',
            'csv_rel_path': 'deepdrid/deepdrid_total.csv',
            'id_col': 'image_id',
            'label_col': 'patient_DR_Level',
            'id_suffix': '.jpg',
            'total_images': 2000,
            'lodo_fold_idx': 5,
            'lodo_slug': 'deepdrid',
            'filename': 'oracle_deepdrid.ipynb'
        },
        {
            'cohort_key': 'IDRiD',
            'cohort_name': 'IDRiD',
            'csv_rel_path': 'idrid/idrid_total.csv',
            'id_col': 'image_id',
            'label_col': 'diagnosis',
            'id_suffix': '.jpg',
            'total_images': 516,
            'lodo_fold_idx': 1,
            'lodo_slug': 'idrid',
            'filename': 'oracle_idrid.ipynb'
        },
        {
            'cohort_key': 'Messidor-2',
            'cohort_name': 'Messidor-2',
            'csv_rel_path': 'messidor2preprocess/messidor_data.csv',
            'id_col': 'id_code',
            'label_col': 'diagnosis',
            'id_suffix': '.png',
            'total_images': 1744,
            'lodo_fold_idx': 2,
            'lodo_slug': 'messidor2',
            'filename': 'oracle_messidor2.ipynb'
        },
        {
            'cohort_key': 'EyePACS',
            'cohort_name': 'EyePACS (Kaggle)',
            'csv_rel_path': 'zipEyepacs/trainLabels.csv/trainLabels.csv',
            'id_col': 'image',
            'label_col': 'level',
            'id_suffix': '.jpeg',
            'total_images': 35126,
            'lodo_fold_idx': 4,
            'lodo_slug': 'eyepacs',
            'filename': 'oracle_eyepacs.ipynb'
        }
    ]

    out_dir = 'notebooks/diabetic_retinopathy/pipeline/oracle'
    os.makedirs(out_dir, exist_ok=True)

    for c in cohorts:
        nb = build_oracle_notebook(
            cohort_key=c['cohort_key'],
            cohort_name=c['cohort_name'],
            csv_rel_path=c['csv_rel_path'],
            id_col=c['id_col'],
            label_col=c['label_col'],
            id_suffix=c['id_suffix'],
            total_images=c['total_images'],
            lodo_fold_idx=c['lodo_fold_idx'],
            lodo_slug=c['lodo_slug'],
            common_cells=common_cells
        )
        p = os.path.join(out_dir, c['filename'])
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        num_cells = len(nb["cells"])
        print(f"Generated: {p} ({num_cells} cells)")

if __name__ == '__main__':
    main()
