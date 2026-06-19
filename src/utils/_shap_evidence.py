import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score

PROCESSED = Path("D:/Research_AI_Bio/03_Datasets/Processed")
FEATURES = PROCESSED / "features"

# Load data
X = np.nan_to_num(np.load(FEATURES / "amp_test_X_150m_new_neg.npy").astype(np.float32), nan=0.0)
PHYS_NAMES = ["length", "mw", "charge", "pI", "hydrophob", "hmoment", "instability",
              "aliphatic", "boman", "entropy", "gravy", "pos_charge", "neg_charge",
              "charge_ratio", "pos_ratio", "neg_ratio"]

hmoment_idx = PHYS_NAMES.index("hmoment")
hydrophob_idx = PHYS_NAMES.index("hydrophob")
charge_idx = PHYS_NAMES.index("charge")

X_esm = X[:, 16:]  # 640-dim
X_phys = X[:, :16]

# Key analysis: can ESM-2 predict hydrophobic moment?
hmoment_vals = X[:, hmoment_idx]
hydrophob_vals = X[:, hydrophob_idx]

# Linear regression: ESM-2 -> HM
lr = LinearRegression()
lr.fit(X_esm, hmoment_vals)
r2_hm = lr.score(X_esm, hmoment_vals)
print(f"ESM-2 (640d) -> Hydrophobic Moment: R^2 = {r2_hm:.4f}")

# Also: ESM-2 -> Hydrophobicity
lr2 = LinearRegression()
lr2.fit(X_esm, hydrophob_vals)
r2_hydrophob = lr2.score(X_esm, hydrophob_vals)
print(f"ESM-2 (640d) -> Hydrophobicity: R^2 = {r2_hydrophob:.4f}")

# ESM-2 -> Charge
lr3 = LinearRegression()
lr3.fit(X_esm, X[:, charge_idx])
r2_charge = lr3.score(X_esm, X[:, charge_idx])
print(f"ESM-2 (640d) -> Net Charge: R^2 = {r2_charge:.4f}")

# Cross-validated R^2 (to check for overfitting)
from sklearn.model_selection import cross_val_score
cv_r2 = cross_val_score(LinearRegression(), X_esm, hmoment_vals, cv=5, scoring='r2')
print(f"\nESM-2 -> HM: cross-val R^2 = {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")

cv_r2_h = cross_val_score(LinearRegression(), X_esm, hydrophob_vals, cv=5, scoring='r2')
print(f"ESM-2 -> Hydrophobicity: cross-val R^2 = {cv_r2_h.mean():.4f} +/- {cv_r2_h.std():.4f}")

cv_r2_c = cross_val_score(LinearRegression(), X_esm, X[:, charge_idx], cv=5, scoring='r2')
print(f"ESM-2 -> Net Charge: cross-val R^2 = {cv_r2_c.mean():.4f} +/- {cv_r2_c.std():.4f}")

# Compare: physchem(charge+HM) -> SHAP of top ESM dims
# Actually, the key comparison: how well can charge+HM predict each ESM dim's SHAP?
# This was done earlier with R^2 < 0.08 for top 5 dims.
# The NEW evidence: how well can ESM-2 predict HM overall?

print(f"\n=== KEY EVIDENCE ===")
print(f"1. ESM-2 640d -> Hydrophobic Moment: R^2 = {r2_hm:.4f} (cross-val: {cv_r2.mean():.4f})")
print(f"2. ESM-2 640d -> Net Charge: R^2 = {r2_charge:.4f} (cross-val: {cv_r2_c.mean():.4f})")
print(f"3. Individual ESM dims -> HM: max |r| < 0.25 (weak individual, distributed)")
print(f"4. Charge+HM -> top-5 ESM dim SHAP: R^2 < 0.08 (SHAP signal irreducible to simple features)")

# Also compute: what fraction of SHAP variance is explained by physchem features?
import shap, xgboost as xgb, json

# Load trained model and compute SHAP
y_train = np.load(FEATURES / "amp_train_y_amp_new_neg.npy")
y_test = np.load(FEATURES / "amp_test_y_amp_new_neg.npy")
X_train = np.nan_to_num(np.load(FEATURES / "amp_train_X_150m_new_neg.npy").astype(np.float32), nan=0.0)

sw = (len(y_train) - y_train.sum()) / y_train.sum()
model = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=sw, early_stopping_rounds=50,
                          eval_metric="auc", random_state=42, n_jobs=-1)
model.fit(X_train, y_train, eval_set=[(X, y_test)], verbose=False)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# For each ESM dim, compute how much of its SHAP variance is explained by ALL 16 physchem features
total_phys_r2 = []
for dim in range(640):
    lr_dim = LinearRegression()
    lr_dim.fit(X_phys, shap_values[:, 16+dim])
    r2 = lr_dim.score(X_phys, shap_values[:, 16+dim])
    total_phys_r2.append(r2)

total_phys_r2 = np.array(total_phys_r2)
print(f"\n=== ESM SHAP explained by ALL 16 physchem features ===")
print(f"Mean R^2: {total_phys_r2.mean():.4f}")
print(f"Median R^2: {np.median(total_phys_r2):.4f}")
print(f"Max R^2: {total_phys_r2.max():.4f}")
print(f"Fraction with R^2 > 0.1: {(total_phys_r2 > 0.1).mean():.3f}")
print(f"Fraction with R^2 > 0.3: {(total_phys_r2 > 0.3).mean():.3f}")

# Top 20 ESM dims by SHAP — what fraction of their SHAP variance is explained by physchem?
top_esm = np.argsort(np.abs(shap_values[:, 16:]).mean(axis=0))[::-1][:20]
top_r2 = total_phys_r2[top_esm]
print(f"\nTop 20 ESM dims: mean physchem R^2 = {top_r2.mean():.4f}")

print("\nDONE.")
