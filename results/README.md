# Experimental Results Directory

This directory stores all serialized outputs, machine-readable JSON metrics, summary CSV tables, and publication-ready figures generated across the benchmark experiments.

---

## Directory Organization

```
results/
├── lodo/                        # 6-Fold Leave-One-Dataset-Out cross-domain benchmark
│   ├── v0/                      # Standard ResNet-50 baseline
│   ├── v1/                      # Dual-branch MIL baseline
│   ├── v2/                      # Global Context attention model
│   ├── v3/                      # Proposed Cumulative Link Model (CLM) + QWK
│   │   ├── aptos/metrics.json   # Tripartite serialization for held-out APTOS
│   │   ├── ddr/metrics.json
│   │   ├── ...
│   │   ├── summary.json         # Complete 6-fold aggregation
│   │   └── summary.csv          # Flat tabular summary
│   └── significance.json        # Pairwise Wilcoxon signed-rank test results
├── head_ablation/               # Controlled 5-head ablation under ResNet-50 features
│   ├── summary.json             # Metrics comparing Softmax_CE, QWK, CORAL, CORN, CLM
│   └── summary.csv              # Flat tabular summary
├── xai/                         # Quantitative explainability against IDRiD lesion masks
│   └── metrics_summary.json     # Pointing Game, Saliency Recall, and Pixel AUROC
└── paper_artifacts/             # Final publication assets matching submitted manuscript
    ├── fig2_lodo_benchmark.pdf  # 300 DPI vector PDF
    ├── fig2_lodo_benchmark.png  # High-resolution raster PNG
    ├── fig3_head_ablation.pdf
    ├── fig3_head_ablation.png
    ├── fig4_rdr_screening.pdf
    ├── fig4_rdr_screening.png
    ├── fig5_xai_lesion_alignment.pdf
    ├── fig5_xai_lesion_alignment.png
    ├── table1_cohort_summary.tex# LaTeX cohort composition table
    ├── table2_lodo_results.tex  # LaTeX LODO 6-fold results table
    ├── table3_head_ablation.tex # LaTeX 5-head comparison table
    └── table4_xai_metrics.tex   # LaTeX quantitative XAI table
```

---

## Generating Results

To populate or refresh the artifacts in this directory:

- **Run LODO Benchmark**: `python scripts/run_lodo.py --variant v3 --held-out aptos`
- **Run Head Ablation**: `python scripts/run_head_ablation.py --all-heads`
- **Run XAI Benchmark**: `python scripts/run_xai.py --variant v3`
- **Export Publication Figures & Tables**: `python scripts/export_paper_artifacts.py`

For detailed schema definitions and field descriptions, consult [`docs/RESULTS_SCHEMA.md`](../docs/RESULTS_SCHEMA.md).
