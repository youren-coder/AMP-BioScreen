import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open(r"D:\Research_AI_Bio\03_Datasets\Final\cdhit_xgb_results.json") as f:
    results = json.load(f)

labels = ["Original\n(90%, pre-split)", "90%\n(post-split)", "70%", "40%"]
aucs = [results["original"]["auc"], results["90"]["auc"], results["70"]["auc"], results["40"]["auc"]]
Ns = [results["original"]["N_total"], results["90"]["N_total"], results["70"]["N_total"], results["40"]["N_total"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: AUC bar chart
colors = ['#2E86AB', '#45A6C9', '#6CBFDD', '#A6D8F0']
bars = ax1.bar(labels, aucs, color=colors, edgecolor='white', linewidth=0.8)
ax1.set_ylabel('AUROCC', fontsize=12)
ax1.set_title('Robustness Across CD-HIT Thresholds', fontsize=13, fontweight='bold')
ax1.set_ylim(0.85, 1.01)
for bar, auc in zip(bars, aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{auc:.3f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
ax1.grid(axis='y', alpha=0.3)

# Right: Scatter (AUC vs N)
n_pos = [results["original"]["N_pos"], results["90"]["N_pos"], results["70"]["N_pos"], results["40"]["N_pos"]]
n_neg = [results["original"]["N_neg"], results["90"]["N_neg"], results["70"]["N_neg"], results["40"]["N_neg"]]

for i, (n, auc, l) in enumerate(zip(Ns, aucs, ["Original", "90%", "70%", "40%"])):
    ax2.scatter(n, auc, s=150, c=colors[i], edgecolors='white', linewidth=1.5, zorder=5)
    ax2.annotate(l, (n, auc), textcoords="offset points", xytext=(10, -5), fontsize=10)

ax2.set_xlabel('Number of Sequences (after CD-HIT)', fontsize=12)
ax2.set_ylabel('AUROCC', fontsize=12)
ax2.set_title('AUC vs Dataset Size', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'D:\Research_AI_Bio\06_Figures\cdhit_robustness.png', dpi=600, bbox_inches='tight')
print("CD-HIT figure saved.")
