"""
SHAP deep analysis v2: Use Tree SHAP (not interventional) for interaction support.
"""
import sys, os, warnings, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from pathlib import Path
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
_utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_utils_dir, '..'))
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR

PROCESSED = PROCESSED_DIR
FEATURES = PROCESSED / "features"
FIGS = FIGURE_DIR

# Load data
X_train = np.nan_to_num(np.load(FEATURES / "amp_train_X_150m_new_neg.npy").astype(np.float32), nan=0.0)
y_train = np.load(FEATURES / "amp_train_y_amp_new_neg.npy")
X_test = np.nan_to_num(np.load(FEATURES / "amp_test_X_150m_new_neg.npy").astype(np.float32), nan=0.0)
y_test = np.load(FEATURES / "amp_test_y_amp_new_neg.npy")

PHYS_NAMES = ["length", "mw", "charge", "pI", "hydrophob", "hmoment", "instability",
              "aliphatic", "boman", "entropy", "gravy", "pos_charge", "neg_charge",
              "charge_ratio", "pos_ratio", "neg_ratio"]
ESM_NAMES = [f"ESM_{i}" for i in range(640)]
feature_names = PHYS_NAMES + ESM_NAMES

charge_idx = PHYS_NAMES.index("charge")
hmoment_idx = PHYS_NAMES.index("hmoment")
hydrophob_idx = PHYS_NAMES.index("hydrophob")
gravy_idx = PHYS_NAMES.index("gravy")

# Train model
sw = (len(y_train) - y_train.sum()) / y_train.sum()
model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=sw, early_stopping_rounds=50,
    eval_metric="auc", random_state=42, n_jobs=-1,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(f"Model trained. Best iter: {model.best_iteration}")

# Use Tree SHAP (default, supports interactions)
print("Computing Tree SHAP...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print(f"SHAP shape: {shap_values.shape}")

# ================================================
# Fig 1: Charge dependence with HMoment interaction coloring
# ================================================
fig1, ax1 = plt.subplots(figsize=(7, 5))
shap.dependence_plot(
    charge_idx, shap_values, X_test,
    feature_names=feature_names,
    interaction_index=hmoment_idx,
    ax=ax1, show=False,
)
ax1.set_xlabel("Net Charge (pH 7)", fontsize=12)
ax1.set_ylabel("SHAP Value", fontsize=12)
ax1.set_title("SHAP Dependence: Net Charge\n(colored by Hydrophobic Moment)", fontsize=12, fontweight="bold")
ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
fig1.tight_layout()
fig1.savefig(str(FIGS / "shap_charge_vs_hmoment.png"), dpi=600, bbox_inches="tight")
print("Saved: shap_charge_vs_hmoment.png")

# ================================================
# Fig 2: HMoment dependence with charge coloring
# ================================================
fig2, ax2 = plt.subplots(figsize=(7, 5))
shap.dependence_plot(
    hmoment_idx, shap_values, X_test,
    feature_names=feature_names,
    interaction_index=charge_idx,
    ax=ax2, show=False,
)
ax2.set_xlabel("Hydrophobic Moment", fontsize=12)
ax2.set_ylabel("SHAP Value", fontsize=12)
ax2.set_title("SHAP Dependence: Hydrophobic Moment\n(colored by Net Charge)", fontsize=12, fontweight="bold")
ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
fig2.tight_layout()
fig2.savefig(str(FIGS / "shap_hmoment_vs_charge.png"), dpi=600, bbox_inches="tight")
print("Saved: shap_hmoment_vs_charge.png")

# ================================================
# Fig 3: Hydrophobicity dependence with charge
# ================================================
fig3, ax3 = plt.subplots(figsize=(7, 5))
shap.dependence_plot(
    hydrophob_idx, shap_values, X_test,
    feature_names=feature_names,
    interaction_index=charge_idx,
    ax=ax3, show=False,
)
ax3.set_xlabel("Hydrophobicity", fontsize=12)
ax3.set_ylabel("SHAP Value", fontsize=12)
ax3.set_title("SHAP Dependence: Hydrophobicity\n(colored by Net Charge)", fontsize=12, fontweight="bold")
ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
fig3.tight_layout()
fig3.savefig(str(FIGS / "shap_hydrophob_vs_charge.png"), dpi=600, bbox_inches="tight")
print("Saved: shap_hydrophob_vs_charge.png")

# ================================================
# Fig 4: 2D amphipathicity analysis
# ================================================
fig4, ax4 = plt.subplots(figsize=(7, 6))

# Compute amphipathic score = charge * hmoment (proxy for "both high")
charge_vals = X_test[:, charge_idx]
hmoment_vals = X_test[:, hmoment_idx]
amphi_score = charge_vals * hmoment_vals

# Color by total SHAP contribution from charge + hmoment
total_shap = shap_values[:, charge_idx] + shap_values[:, hmoment_idx]

sc = ax4.scatter(
    charge_vals, hmoment_vals,
    c=total_shap, cmap="RdBu_r", alpha=0.8, s=70,
    vmin=-0.12, vmax=0.12, edgecolors="grey", linewidth=0.3
)
charge_med = np.median(charge_vals)
hmoment_med = np.median(hmoment_vals)
ax4.axhline(y=hmoment_med, color="gray", linestyle="--", alpha=0.5, label=f"HM median={hmoment_med:.3f}")
ax4.axvline(x=charge_med, color="gray", linestyle="--", alpha=0.5, label=f"Charge median={charge_med:.2f}")

# Annotate quadrants
ax4.text(charge_vals.max()*0.8, hmoment_vals.max()*0.9, "Amphipathic\n(high charge + high HM)", 
         ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
ax4.text(charge_vals.min()*0.8, hmoment_vals.min()*0.9, "Non-amphipathic", 
         ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))

ax4.set_xlabel("Net Charge (pH 7)", fontsize=12)
ax4.set_ylabel("Hydrophobic Moment", fontsize=12)
ax4.set_title("Amphipathicity Analysis: Charge-HMoment SHAP Synergy", fontsize=13, fontweight="bold")
ax4.legend(fontsize=8)
cbar = plt.colorbar(sc, ax=ax4)
cbar.set_label("SHAP (Charge + HM)", fontsize=10)
fig4.tight_layout()
fig4.savefig(str(FIGS / "shap_amphipathicity_2d.png"), dpi=600, bbox_inches="tight")
print("Saved: shap_amphipathicity_2d.png")

# ================================================
# Fig 5: SHAP interaction matrix (physchem only)
# ================================================
print("Computing SHAP interaction values...")
shap_interaction = explainer.shap_interaction_values(X_test)

# Extract physchem subset
phys_interaction = np.abs(shap_interaction[:, :16, :16]).mean(axis=0)

fig5, ax5 = plt.subplots(figsize=(8, 7))
im = ax5.imshow(phys_interaction, cmap="YlOrRd", aspect="auto")
ax5.set_xticks(range(16))
ax5.set_yticks(range(16))
ax5.set_xticklabels(PHYS_NAMES, rotation=45, ha="right", fontsize=7)
ax5.set_yticklabels(PHYS_NAMES, fontsize=7)
ax5.set_title("SHAP Interaction Matrix: Physicochemical Features", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax5, label="mean(|SHAP interaction|)")
fig5.tight_layout()
fig5.savefig(str(FIGS / "shap_interaction_matrix.png"), dpi=600, bbox_inches="tight")
print("Saved: shap_interaction_matrix.png")

# ================================================
# Key numbers for the paper
# ================================================
print("\n" + "=" * 60)
print("KEY METRICS FOR PAPER")
print("=" * 60)

# Per-feature mean SHAP
phys_idx = list(range(16))
phys_shap = np.abs(shap_values[:, phys_idx]).mean(axis=0)
print("\nPhyschem SHAP importance (mean|SHAP|):")
for i in np.argsort(phys_shap)[::-1]:
    print(f"  {PHYS_NAMES[i]:<15} {phys_shap[i]:.6f}")

# Charge-HMoment interaction specifically
ch_hm_interaction = np.abs(shap_interaction[:, charge_idx, hmoment_idx]).mean()
print(f"\nCharge-HMoment interaction: mean|interaction| = {ch_hm_interaction:.6f}")

# Top ESM dimensions
esm_abs = np.abs(shap_values[:, 16:]).mean(axis=0)
top_esm_idx = np.argsort(esm_abs)[::-1][:15]
print("\nTop 15 ESM-2 dimensions by mean|SHAP|:")
for i, idx in enumerate(top_esm_idx):
    # Check correlation with charge and hmoment
    corr_charge = np.corrcoef(X_test[:, 16+idx], X_test[:, charge_idx])[0, 1]
    corr_hm = np.corrcoef(X_test[:, 16+idx], X_test[:, hmoment_idx])[0, 1]
    print(f"  {i+1:>2}. ESM_{idx:<4} SHAP={esm_abs[idx]:.4f}  corr(charge)={corr_charge:+.3f}  corr(HM)={corr_hm:+.3f}")

# Disentangle: what fraction of top ESM dims' SHAP is explained by charge/HM correlation?
print("\nDisentangling charge vs HM contributions in top ESM dimensions:")
for i, idx in enumerate(top_esm_idx[:5]):
    # Partial correlation: SHAP ~ charge + HM
    from sklearn.linear_model import LinearRegression
    X_pred = np.column_stack([X_test[:, charge_idx], X_test[:, hmoment_idx]])
    lr = LinearRegression().fit(X_pred, shap_values[:, 16+idx])
    r2 = lr.score(X_pred, shap_values[:, 16+idx])
    print(f"  ESM_{idx}: R^2(charge+HM -> SHAP) = {r2:.3f}")

print("\nDONE.")
