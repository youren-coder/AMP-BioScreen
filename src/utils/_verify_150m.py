import numpy as np
from sklearn import metrics
import xgboost as xgb

# Load 150M fused features
train_X = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_train_X_150m.npy")
val_X = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_val_X_150m.npy")
test_X = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_test_X_150m.npy")
train_y = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_train_y_amp.npy")
val_y = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_val_y_amp.npy")
test_y = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_test_y_amp.npy")

# Use same parameters as CD-HIT experiment
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(train_y==0).sum() / (train_y==1).sum(),
    eval_metric='auc', early_stopping_rounds=30, random_state=42
)
model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)

y_prob = model.predict_proba(test_X)[:, 1]
y_pred = model.predict(test_X)

print(f"AUC: {metrics.roc_auc_score(test_y, y_prob):.4f}")
print(f"F1:  {metrics.f1_score(test_y, y_pred):.4f}")
print(f"MCC: {metrics.matthews_corrcoef(test_y, y_pred):.4f}")
print(f"ACC: {metrics.accuracy_score(test_y, y_pred):.4f}")
print(f"Precision: {metrics.precision_score(test_y, y_pred):.4f}")
print(f"Recall: {metrics.recall_score(test_y, y_pred):.4f}")

# Also try with paper's original params (depth=4, lr=0.2)
model2 = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.2,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(train_y==0).sum() / (train_y==1).sum(),
    eval_metric='auc', early_stopping_rounds=20, random_state=42
)
model2.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
y_prob2 = model2.predict_proba(test_X)[:, 1]
y_pred2 = model2.predict(test_X)
print(f"\nPaper params (depth=4, lr=0.2):")
print(f"AUC: {metrics.roc_auc_score(test_y, y_prob2):.4f}")
print(f"F1:  {metrics.f1_score(test_y, y_pred2):.4f}")
