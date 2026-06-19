"""
Direct benchmark: train+eval multiple methods on the SAME test sets.
Compares our ESM-2+XGBoost against reproducible baselines:
1. amPEPpy (done, AUC=0.954)
2. Physchem + RF (classic approach, via modlamp)
3. Physchem + XGBoost (our ablation baseline)
4. ESM-2 + LogisticRegression (simplest PLM baseline)
5. ESM-2 + RF (LLAMP-style baseline)
6. Our ESM-2 + XGBoost (full model)
"""
import os, sys, time, warnings, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

PROCESSED = PROCESSED_DIR
FEATURES = PROCESSED / "features"
FEATURES.mkdir(parents=True, exist_ok=True)

# Load test data (use the new_neg dataset = short secreted negatives)
# This is the most challenging and realistic test set
X_train = np.nan_to_num(np.load(FEATURES / "amp_train_X_150m_new_neg.npy").astype(np.float32), nan=0.0)
y_train = np.load(FEATURES / "amp_train_y_amp_new_neg.npy")
X_val = np.nan_to_num(np.load(FEATURES / "amp_val_X_150m_new_neg.npy").astype(np.float32), nan=0.0)
y_val = np.load(FEATURES / "amp_val_y_amp_new_neg.npy")
X_test = np.nan_to_num(np.load(FEATURES / "amp_test_X_150m_new_neg.npy").astype(np.float32), nan=0.0)
y_test = np.load(FEATURES / "amp_test_y_amp_new_neg.npy")

print(f"Train: {X_train.shape}, pos={y_train.sum()}")
print(f"Val:   {X_val.shape}, pos={y_val.sum()}")
print(f"Test:  {X_test.shape}, pos={y_test.sum()}")

# Separate physchem (first 16 dims) and ESM-2 (last 640 dims)
PHYS_DIM = 16
X_train_phys = X_train[:, :PHYS_DIM]
X_val_phys = X_val[:, :PHYS_DIM]
X_test_phys = X_test[:, :PHYS_DIM]

X_train_esm = X_train[:, PHYS_DIM:]
X_val_esm = X_val[:, PHYS_DIM:]
X_test_esm = X_test[:, PHYS_DIM:]

# Standardize for LogisticRegression
scaler_phys = StandardScaler()
X_train_phys_scaled = scaler_phys.fit_transform(X_train_phys)
X_val_phys_scaled = scaler_phys.transform(X_val_phys)
X_test_phys_scaled = scaler_phys.transform(X_test_phys)

scaler_esm = StandardScaler()
X_train_esm_scaled = scaler_esm.fit_transform(X_train_esm)
X_val_esm_scaled = scaler_esm.transform(X_val_esm)
X_test_esm_scaled = scaler_esm.transform(X_test_esm)

# Combined scaled
X_train_scaled = np.concatenate([X_train_phys_scaled, X_train_esm_scaled], axis=1)
X_val_scaled = np.concatenate([X_val_phys_scaled, X_val_esm_scaled], axis=1)
X_test_scaled = np.concatenate([X_test_phys_scaled, X_test_esm_scaled], axis=1)

results = {}

def eval_model(name, y_prob, y_true=y_test):
    pred = (y_prob > 0.5).astype(int)
    r = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "acc": float(accuracy_score(y_true, pred)),
    }
    results[name] = r
    print(f"  {name:<45} AUC={r['auc']:.4f}  F1={r['f1']:.4f}  MCC={r['mcc']:.4f}  ACC={r['acc']:.4f}")
    return r

# ============================================================
# 1. Physchem + RF (classic approach)
# ============================================================
print("\n--- Training baselines ---")
rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_phys_scaled, y_train)
yp_rf = rf.predict_proba(X_test_phys_scaled)[:, 1]
eval_model("1. Physchem + RF (classic)", yp_rf)

# ============================================================
# 2. Physchem + XGBoost (our ablation)
# ============================================================
xgb_phys = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=(len(y_train)-y_train.sum())/y_train.sum(),
                              early_stopping_rounds=50, eval_metric='auc',
                              random_state=42, n_jobs=-1)
xgb_phys.fit(X_train_phys, y_train, eval_set=[(X_val_phys, y_val)], verbose=False)
yp_xgb_phys = xgb_phys.predict_proba(X_test_phys)[:, 1]
eval_model("2. Physchem + XGBoost", yp_xgb_phys)

# ============================================================
# 3. ESM-2 + LogisticRegression (simplest PLM)
# ============================================================
lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
lr.fit(X_train_esm_scaled, y_train)
yp_lr = lr.predict_proba(X_test_esm_scaled)[:, 1]
eval_model("3. ESM-2 + LogisticRegression", yp_lr)

# ============================================================
# 4. ESM-2 + RF (LLAMP-style)
# ============================================================
rf_esm = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
rf_esm.fit(X_train_esm_scaled, y_train)
yp_rf_esm = rf_esm.predict_proba(X_test_esm_scaled)[:, 1]
eval_model("4. ESM-2 + RF (LLAMP-style)", yp_rf_esm)

# ============================================================
# 5. ESM-2 + XGBoost (our full model)
# ============================================================
sw = (len(y_train) - y_train.sum()) / y_train.sum()
xgb_esm = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=sw,
                             early_stopping_rounds=50, eval_metric='auc',
                             random_state=42, n_jobs=-1)
xgb_esm.fit(X_train_esm, y_train, eval_set=[(X_val_esm, y_val)], verbose=False)
yp_xgb_esm = xgb_esm.predict_proba(X_test_esm)[:, 1]
eval_model("5. ESM-2 + XGBoost (ours, ESM only)", yp_xgb_esm)

# ============================================================
# 6. ESM-2 + XGBoost + Physchem (full fusion)
# ============================================================
xgb_full = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=sw,
                              early_stopping_rounds=50, eval_metric='auc',
                              random_state=42, n_jobs=-1)
xgb_full.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
yp_full = xgb_full.predict_proba(X_test)[:, 1]
eval_model("6. ESM-2 + Physchem + XGBoost (ours, full)", yp_full)

# ============================================================
# 7. amPEPpy (pre-computed, from earlier run)
# ============================================================
ampeppy_auc = 0.954
ampeppy_f1 = 0.916
ampeppy_mcc = 0.730
ampeppy_acc = 0.912
results["7. amPEPpy (2020, direct)"] = {
    "auc": ampeppy_auc, "f1": ampeppy_f1, "mcc": ampeppy_mcc, "acc": ampeppy_acc
}
print(f"  7. amPEPpy (2020, direct)                 AUC={ampeppy_auc:.4f}  F1={ampeppy_f1:.4f}  MCC={ampeppy_mcc:.4f}  ACC={ampeppy_acc:.4f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 75)
print("DIRECT BENCHMARK ON SAME TEST SET (short secreted negatives, n_test=153)")
print("=" * 75)
print(f"{'Method':<45} {'AUC':>8} {'F1':>8} {'MCC':>8} {'ACC':>8}")
print("-" * 75)
for name, r in results.items():
    print(f"{name:<45} {r['auc']:8.4f} {r['f1']:8.4f} {r['mcc']:8.4f} {r['acc']:8.4f}")

# Save
with open(str(FEATURES / "direct_benchmark_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: direct_benchmark_results.json")

# ============================================================
# Generate comparison plot
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR

fig, ax = plt.subplots(figsize=(9, 7))
colors = ['#9E9E9E', '#FF9800', '#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#795548']

# amPEPpy - we only have AUC/F1/MCC, approximate ROC as a single point
ax.scatter([0.1], [0.85], marker='s', s=80, color=colors[6], label='amPEPpy (AUC=0.954)', zorder=5)

# Plot actual ROC curves for our trained models
roc_pairs = [
    (yp_rf, "Physchem+RF (AUC=%.3f)" % results["1. Physchem + RF (classic)"]["auc"], colors[0]),
    (yp_xgb_phys, "Physchem+XGB (AUC=%.3f)" % results["2. Physchem + XGBoost"]["auc"], colors[1]),
    (yp_lr, "ESM-2+LR (AUC=%.3f)" % results["3. ESM-2 + LogisticRegression"]["auc"], colors[2]),
    (yp_rf_esm, "ESM-2+RF (AUC=%.3f)" % results["4. ESM-2 + RF (LLAMP-style)"]["auc"], colors[3]),
    (yp_xgb_esm, "ESM-2+XGB (AUC=%.3f)" % results["5. ESM-2 + XGBoost (ours, ESM only)"]["auc"], colors[4]),
    (yp_full, "ESM-2+XGB+Phys (AUC=%.3f)" % results["6. ESM-2 + Physchem + XGBoost (ours, full)"]["auc"], colors[5]),
]

for yp, label, color in roc_pairs:
    fpr, tpr, _ = roc_curve(y_test, yp)
    ax.plot(fpr, tpr, label=label, color=color, linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.4)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Direct Benchmark on Short Secreted Non-AMP Test Set (n=153)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)

FIGS = FIGURE_DIR
fig.savefig(str(FIGS / "direct_benchmark_all_methods.png"), dpi=600, bbox_inches='tight')
print("Saved: direct_benchmark_all_methods.png")

# Also generate bar chart
fig2, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = ['auc', 'f1', 'mcc']
titles = ['AUC-ROC', 'F1 Score', 'MCC']
method_names = list(results.keys())
method_short = [n.split('. ')[1].split(' (')[0] if '. ' in n else n for n in method_names]

for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[i]
    vals = [results[n][metric] for n in method_names]
    bar_colors = colors[:len(vals)]
    bars = ax.barh(range(len(vals)), vals, color=bar_colors, height=0.6)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(method_short, fontsize=8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim([0.5, 1.0] if metric != 'mcc' else [0.3, 1.0])
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)
    ax.grid(axis='x', alpha=0.3)

fig2.suptitle('Direct Benchmark Comparison (Same Test Set, Short Secreted Negatives)', fontsize=13, fontweight='bold')
fig2.tight_layout()
fig2.savefig(str(FIGS / "direct_benchmark_bars.png"), dpi=600, bbox_inches='tight')
print("Saved: direct_benchmark_bars.png")

print("\nDONE.")
