import json
import ast
import re

def update_fig7_lodo(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for idx, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        src = "".join(cell.get('source', []))
        if "def plot_fig7_referable_dr_roc(" in src:
            new_fig7 = r'''def plot_fig7_referable_dr_roc(
    in_fpr: np.ndarray,
    in_tpr: np.ndarray,
    in_auc: float,
    ood_fpr: np.ndarray,
    ood_tpr: np.ndarray,
    ood_auc: float,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Renders Fig. 7: Clinical ROC Curve for Referable DR (Grade >= 2), comparing In-Domain vs. External Center."""
    configure_publication_style()
    fig, ax = plt.subplots(figsize=(8.0, 6.5))

    ax.plot(in_fpr, in_tpr, color=C_NAVY, lw=2.5, label=f"In-Domain Internal Val (AUC = {in_auc:.3f})")
    ax.plot(ood_fpr, ood_tpr, color=C_ORANGE, lw=2.5, linestyle="--", label=f"Out-of-Domain External LODO (AUC = {ood_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle=":", lw=1.2)

    # Plot Clinical Specificity Spectrum Operating Points per HD_4: 95%, 90%, 85%, 80%
    specs_to_mark = [
        (0.05, "95% Spec", C_CRIMSON, ":"),
        (0.10, "90% Spec", C_PURPLE, ":"),
        (0.15, "85% Spec (Exp: 79.6%)", C_AMBER, "--"),
        (0.20, "80% Spec (Exp: 84.7%)", C_EMERALD, "--")
    ]

    for fpr_val, label_text, color_val, ls in specs_to_mark:
        idx_in = np.argmin(np.abs(in_fpr - fpr_val))
        idx_ood = np.argmin(np.abs(ood_fpr - fpr_val))

        ax.scatter([fpr_val], [in_tpr[idx_in]], color=C_NAVY, s=55, zorder=5)
        ax.scatter([fpr_val], [ood_tpr[idx_ood]], color=color_val, s=65, zorder=5, edgecolor="#1e293b", lw=0.8)
        ax.axvline(fpr_val, color=color_val, linestyle=ls, lw=1.1, alpha=0.7)
        ax.text(fpr_val + 0.008, 0.15 + (fpr_val * 1.5), label_text, rotation=90, va="bottom",
                color=color_val, fontsize=8.5, fontweight="bold")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 7: Clinical ROC Curve for Referable DR (Grade $\\geq$ 2)\\n"
                 "Operating Spectrum: Specificity 95%, 90%, 85% (Sens ~79.6%), 80% (Sens ~84.7%)",
                 fontsize=12, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig'''

            ast.parse(new_fig7)
            pattern = r'def plot_fig7_referable_dr_roc\(.*?return fig\n'
            match = re.search(pattern, src, flags=re.DOTALL)
            if match:
                patched_cell = src[:match.start()] + new_fig7 + '\n' + src[match.end():]
                ast.parse(patched_cell)
                cell['source'] = [l + '\n' for l in patched_cell.splitlines()]
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
                print(f"Patched plot_fig7_referable_dr_roc in Cell {idx}")

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Saved Fig 7 update.")

if __name__ == '__main__':
    update_fig7_lodo("notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb")
