with open('scratch/update_ablation_waterfall.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the Figure 4 block cleanly
start_marker = '    x = np.arange(len(labels))\n    colors = ["#dc2626", "#ea580c", "#0d9488", "#1e3a8a"]'
end_marker = '    plt.tight_layout()'

target_block = content[content.find(start_marker):content.find(end_marker) + len(end_marker)]

clean_block = '''    x = np.arange(len(labels))
    colors = ["#dc2626", "#ea580c", "#0d9488", "#1e3a8a"]

    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.88, edgecolor="#1e293b", linewidth=1.5, width=0.55)

    ax.set_ylabel("Generalization Drop Δ_LODO = (Val - OOD Test)", fontweight=\'bold\')
    ax.set_title("Figure 4: Waterfall Generalization Gap (Δ QWK) Decomposition Across Architectural Ladder", fontsize=12.5, fontweight=\'bold\', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight=\'bold\', rotation=12, ha=\'right\')
    ax.grid(True, linestyle=\'--\', alpha=0.4, axis=\'y\')

    # Annotate bar heights and marginal reduction
    for i in range(len(means)):
        val = means[i]
        tag = hyp_tags[i]
        if i > 0:
            marginal = val - means[i - 1]
            ann_text = f"Δ: {val:.3f}\\n({marginal:+.3f})\\n[{tag}]"
        else:
            ann_text = f"Δ: {val:.3f}\\n[{tag}]"
        ax.annotate(ann_text, (x[i], val + (stds[i] if stds[i] > 0 else 0.01) + 0.012),
                    ha=\'center\', fontsize=9.5, fontweight=\'bold\', color="#1e293b")

    ax.set_ylim(0, max(means + [0.3]) * 1.35)
    plt.tight_layout()'''

assert start_marker in content, "start_marker not found"
content = content.replace(target_block, clean_block, 1)

with open('scratch/update_ablation_waterfall.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Cleaned Figure 4 block in update_ablation_waterfall.py successfully!")
