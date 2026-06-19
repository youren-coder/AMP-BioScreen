import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import json

PROCESSED = Path("D:/Research_AI_Bio/03_Datasets/Processed")
FEATURES = PROCESSED / "features"

# Load full Hemolytik2 with hemolysis labels
df = pd.read_csv("D:/Research_AI_Bio/02_Databases/hemolytik2_complete.csv")
df['seq'] = df['seq'].str.upper().str.strip()
df['hemo_label'] = (df['activity'].astype(str).str.upper().str.strip() == 'HEMOLYTIC').astype(int)

# Load our ESM-2 150M embeddings for training
# We need to map Hemolytik2 seqs to existing embeddings
# First check which Hemolytik2 seqs have existing embeddings
X_train_esm = np.load(FEATURES / "amp_train_esm2_150m.npy")
X_val_esm = np.load(FEATURES / "amp_val_esm2_150m.npy")
X_test_esm = np.load(FEATURES / "amp_test_esm2_150m.npy")

old_train = pd.read_csv(PROCESSED / "amp_train_real.csv")
old_val = pd.read_csv(PROCESSED / "amp_val_real.csv")
old_test = pd.read_csv(PROCESSED / "amp_test_real.csv")

seq_to_esm = {}
for i, row in old_train.iterrows():
    seq_to_esm[row["sequence"].upper().strip()] = X_train_esm[i]
for i, row in old_val.iterrows():
    seq_to_esm[row["sequence"].upper().strip()] = X_val_esm[i]
for i, row in old_test.iterrows():
    seq_to_esm[row["sequence"].upper().strip()] = X_test_esm[i]

# Since Hemolytik2 has only 48 hemolytic AND 6441 non-hemolytic (6489 unique),
# and we only have ESM-2 embeddings for our dataset (~1967 sequences),
# Let me train on what we can: Hemolytik2 labeled sequences with available embeddings
hemo_seqs_set = set(df['seq'])
our_seqs_set = set(seq_to_esm.keys())
usable = hemo_seqs_set & our_seqs_set
print(f"Usable Hemolytik2 sequences with embeddings: {len(usable)}")

# Build training data from usable sequences
X_list, y_list = [], []
for seq in usable:
    label_df = df[df['seq'] == seq]
    hemo_val = label_df['hemo_label'].iloc[0]
    esm_emb = seq_to_esm[seq]
    X_list.append(esm_emb)
    y_list.append(hemo_val)

X_hemo = np.array(X_list, dtype=np.float32)
y_hemo = np.array(y_list)

pos_count = y_hemo.sum()
neg_count = len(y_hemo) - pos_count
print(f"Training data: {len(X_hemo)} sequences, hemolytic={pos_count}, non-hemolytic={neg_count}")
print(f"Imbalance ratio: {neg_count/pos_count:.1f}:1")

# 5-fold CV for hemolysis regression (use proba as regression target)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_r2, cv_rmse = [], []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_hemo, y_hemo)):
    X_tr, X_vl = X_hemo[tr_idx], X_hemo[val_idx]
    y_tr, y_vl = y_hemo[tr_idx], y_hemo[val_idx]
    
    sw = (len(y_tr) - y_tr.sum()) / y_tr.sum() if y_tr.sum() > 0 else 1
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=sw,
        early_stopping_rounds=30, eval_metric='logloss',
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    y_prob = model.predict_proba(X_vl)[:, 1]
    
    r2 = r2_score(y_vl, y_prob)
    rmse = np.sqrt(mean_squared_error(y_vl, y_prob))
    cv_r2.append(r2)
    cv_rmse.append(rmse)
    print(f"Fold {fold+1}: R^2={r2:.4f}, RMSE={rmse:.4f}")

# Fit on all data for inference
sw = (len(y_hemo) - y_hemo.sum()) / y_hemo.sum()
final_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=sw,
    early_stopping_rounds=30, eval_metric='logloss',
    random_state=42, n_jobs=-1,
)
final_model.fit(X_hemo, y_hemo, eval_set=[(X_hemo, y_hemo)], verbose=False)

# Predict hemolysis for our AMP test set
test_amp = pd.read_csv(PROCESSED / "amp_test_new_neg.csv")
test_amp_pos = test_amp[test_amp['label_amp'] == 1]
test_hemo_preds = []
for _, row in test_amp_pos.iterrows():
    seq = row['sequence'].upper().strip()
    if seq in seq_to_esm:
        prob = final_model.predict_proba(seq_to_esm[seq].reshape(1, -1))[0, 1]
    else:
        prob = 0.5  # default if no embedding
    test_hemo_preds.append(prob)

test_hemo_preds = np.array(test_hemo_preds)
print(f"\nHemolysis predictions for {len(test_amp_pos)} AMPs in test set:")
print(f"  Mean P(hemolysis): {test_hemo_preds.mean():.4f}")
print(f"  P(hemolysis) < 0.2: {(test_hemo_preds < 0.2).sum()} ({(test_hemo_preds < 0.2).mean()*100:.1f}%)")
print(f"  P(hemolysis) < 0.3: {(test_hemo_preds < 0.3).sum()} ({(test_hemo_preds < 0.3).mean()*100:.1f}%)")
print(f"  P(hemolysis) > 0.5: {(test_hemo_preds > 0.5).sum()} ({(test_hemo_preds > 0.5).mean()*100:.1f}%)")

# Save results
results = {
    "cv_r2_mean": float(np.mean(cv_r2)),
    "cv_r2_std": float(np.std(cv_r2)),
    "cv_rmse_mean": float(np.mean(cv_rmse)),
    "cv_rmse_std": float(np.std(cv_rmse)),
    "training_samples": len(X_hemo),
    "n_hemolytic": int(pos_count),
}
with open(str(FEATURES / "hemolysis_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nCV R^2: {np.mean(cv_r2):.4f} +/- {np.std(cv_r2):.4f}")
print(f"CV RMSE: {np.mean(cv_rmse):.4f} +/- {np.std(cv_rmse):.4f}")
print("DONE.")
