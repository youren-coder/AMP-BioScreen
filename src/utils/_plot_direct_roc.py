import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load amPEPpy results
ampep = pd.read_csv(r"D:\Research_AI_Bio\03_Datasets\Final\ampeppy_test_results.csv")
y_true = ampep["y_true"].values

# Load our model probabilities (150M fused)
test_X = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_test_X_150m.npy")
test_y = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_test_y_amp.npy")

import xgboost as xgb
train_X = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_train_X_150m.npy")
train_y = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_train_y_amp.npy")
val_X = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_val_X_150m.npy")
val_y = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_val_y_amp.npy")

model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(train_y==0).sum()/(train_y==1).sum(),
    eval_metric='auc', early_stopping_rounds=30, random_state=42
)
model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
y_prob_ours = model.predict_proba(test_X)[:, 1]

# Compute ROC curves
fpr_a, tpr_a, _ = metrics.roc_curve(y_true, ampep["y_prob"].values)
auc_a = metrics.roc_auc_score(y_true, ampep["y_prob"].values)

fpr_o, tpr_o, _ = metrics.roc_curve(test_y, y_prob_ours)
auc_o = metrics.roc_auc_score(test_y, y_prob_ours)

# Also add CD-HIT 40% for reference (need to compute)
# For now just plot the two main curves + random baseline

plt.figure(figsize=(6, 5.5))
plt.plot(fpr_o, tpr_o, 'b-', linewidth=2, label=f'ESM-2 150M + XGBoost (AUC={auc_o:.3f})')
plt.plot(fpr_a, tpr_a, 'r--', linewidth=2, label=f'amPEPpy RF (AUC={auc_a:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3, label='Random (AUC=0.500)')

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Direct Comparison on Independent Test Set (n=137)', fontsize=13)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)

# Add inset text
plt.text(0.55, 0.25, f'Same test set\nSame train/val/test split\nReal non-AMP negatives',
         transform=plt.gca().transAxes, fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(r'D:\Research_AI_Bio\06_Figures\direct_comparison_roc.png', dpi=600, bbox_inches='tight')
print(f"ROC saved. Ours AUC={auc_o:.4f}, amPEPpy AUC={auc_a:.4f}")
