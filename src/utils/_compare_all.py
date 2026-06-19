# Verify: compare amPEPpy (RF+physicochem) vs our XGBoost+physicochem (no ESM-2) vs XGBoost+ESM-2
# To ensure fair comparison
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics

# Load amPEPpy results
ampep_results = pd.read_csv(r"D:\Research_AI_Bio\03_Datasets\Final\ampeppy_test_results.csv")
y_true = ampep_results["y_true"].values
y_prob_ampep = ampep_results["y_prob"].values
y_pred_ampep = ampep_results["y_pred"].values

print("=== amPEPpy (RF + 理化特征) ===")
print(f"AUC: {metrics.roc_auc_score(y_true, y_prob_ampep):.4f}")
print(f"F1:  {metrics.f1_score(y_true, y_pred_ampep):.4f}")
print(f"MCC: {metrics.matthews_corrcoef(y_true, y_pred_ampep):.4f}")

# Now train XGBoost on same physicochemical features from test set
# Load train physchem features
X_train_phys = pd.read_csv(r"D:\Research_AI_Bio\03_Datasets\Processed\features\features_physicochem_train.csv", index_col=0)
X_val_phys = pd.read_csv(r"D:\Research_AI_Bio\03_Datasets\Processed\features\features_physicochem_val.csv", index_col=0)
X_test_phys = pd.read_csv(r"D:\Research_AI_Bio\03_Datasets\Processed\features\features_physicochem_test.csv", index_col=0)

y_train = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_train_y_amp.npy")
y_val = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_val_y_amp.npy")
y_test = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_test_y_amp.npy")

print(f"Physchem shapes: train {X_train_phys.shape}, val {X_val_phys.shape}, test {X_test_phys.shape}")

import xgboost as xgb
model_phys = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
    eval_metric='auc', early_stopping_rounds=30, random_state=42
)
model_phys.fit(X_train_phys, y_train, eval_set=[(X_val_phys, y_val)], verbose=False)

y_prob_phys_xgb = model_phys.predict_proba(X_test_phys)[:, 1]
y_pred_phys_xgb = model_phys.predict(X_test_phys)

print("\n=== XGBoost (仅理化特征, 不含ESM-2) ===")
print(f"AUC: {metrics.roc_auc_score(y_test, y_prob_phys_xgb):.4f}")
print(f"F1:  {metrics.f1_score(y_test, y_pred_phys_xgb):.4f}")
print(f"MCC: {metrics.matthews_corrcoef(y_test, y_pred_phys_xgb):.4f}")

# Also load ESM-2+XGBoost result from results.json
import json
with open(r"D:\Research_AI_Bio\03_Datasets\Processed\features\results.json") as f:
    r = json.load(f)
print(f"\n=== XGBoost (ESM-2 150M + 理化融合) ===")
print(f"AUC: {r.get('150m', {}).get('auc', 'N/A')}")
print(f"Content: {r}")
