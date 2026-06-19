"""Use original ESM-2 features but filter to CD-HIT representatives, then re-split and re-train."""
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb

# Step 1: Reconstruct the full feature matrix
train_X = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_train_X_150m.npy")
val_X = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_val_X_150m.npy")
test_X = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_test_X_150m.npy")
train_y = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_train_y_amp.npy")
val_y = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_val_y_amp.npy")
test_y = np.load(r"D:\Research_AI_Bio\03_Datasets\Processed\features\amp_test_y_amp.npy")

# Concatenate all
X_all = np.vstack([train_X, val_X, test_X])
y_all = np.concatenate([train_y, val_y, test_y])

# Load original CSV to map indices
train_df = pd.read_csv(r"D:\Research_AI_Bio\03_Datasets\Processed\amp_train_real.csv")
val_df = pd.read_csv(r"D:\Research_AI_Bio\03_Datasets\Processed\amp_val_real.csv")
test_df = pd.read_csv(r"D:\Research_AI_Bio\03_Datasets\Processed\amp_test_real.csv")

# Build global index->sequence mapping
all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
all_seqs = all_df["sequence"].tolist()
all_labels = all_df["label_amp"].tolist()

print(f"All data: {len(all_seqs)} sequences, X shape: {X_all.shape}")

# Step 2: For each CD-HIT threshold, find which global indices are representatives,
# then filter X_all and re-split.
# The CD-HIT reps CSV contains the full sequences - we match by sequence

results = {}

for thresh_label in ["90", "70", "40"]:
    reps_df = pd.read_csv(rf"D:\Research_AI_Bio\03_Datasets\Final\cdhit_{thresh_label}_reps.csv")
    rep_seqs = reps_df["sequence"].tolist()
    rep_labels = reps_df["label_amp"].tolist()
    
    # Match: find indices in all_seqs where sequence matches
    # Build a lookup: sequence -> list of global indices (sequences may repeat)
    seq_to_idx = {}
    for idx, seq in enumerate(all_seqs):
        seq_to_idx.setdefault(seq, []).append(idx)
    
    rep_indices = []
    for seq in rep_seqs:
        if seq in seq_to_idx:
            # Take first unused index (or just the first one)
            rep_indices.append(seq_to_idx[seq][0])
        else:
            print(f"  WARNING: seq not found in original data")
    
    # Filter X and y
    X_rep = X_all[rep_indices]
    y_rep = np.array(rep_labels)
    
    print(f"\n=== CD-HIT {thresh_label}% ===")
    print(f"  Sequences: {len(rep_indices)}, Pos: {y_rep.sum()}, Neg: {(y_rep==0).sum()}")
    
    # Re-split: stratified 80/10/10
    train_val, X_test, train_val_y, test_y_cd = train_test_split(
        np.arange(len(y_rep)), rep_labels, test_size=0.1, stratify=rep_labels, random_state=42
    )
    # train_val split: 8/9 train, 1/9 val (=80/10 of original)
    train_idx, val_idx, y_train_cd, y_val_cd = train_test_split(
        train_val, train_val_y, test_size=1/9, stratify=train_val_y, random_state=42
    )
    
    X_tr = X_rep[train_idx]
    X_v = X_rep[val_idx]
    X_te = X_rep[X_test]
    y_tr = np.array([rep_labels[i] for i in train_idx])
    y_v = np.array([rep_labels[i] for i in val_idx])
    y_te = np.array([rep_labels[i] for i in X_test])
    
    print(f"  Train: {len(y_tr)} (pos={y_tr.sum()}), Val: {len(y_v)} (pos={y_v.sum()}), Test: {len(y_te)} (pos={y_te.sum()})")
    
    # Train XGBoost
    scale_weight = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric='auc', early_stopping_rounds=30, random_state=42
    )
    model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = model.predict(X_te)
    
    auc = metrics.roc_auc_score(y_te, y_prob)
    f1 = metrics.f1_score(y_te, y_pred)
    mcc = metrics.matthews_corrcoef(y_te, y_pred)
    acc = metrics.accuracy_score(y_te, y_pred)
    
    print(f"  AUC: {auc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}, ACC: {acc:.4f}")
    
    results[thresh_label] = {
        "auc": auc, "f1": f1, "mcc": mcc, "acc": acc,
        "N_total": len(rep_indices), "N_pos": int(y_rep.sum()), "N_neg": int((y_rep==0).sum()),
        "test_N": len(y_te)
    }

# Also run original (no CD-HIT) for comparison: use existing splits
print("\n=== Original (no CD-HIT, existing split) ===")
model_orig = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(train_y==0).sum() / max((train_y==1).sum(), 1),
    eval_metric='auc', early_stopping_rounds=30, random_state=42
)
model_orig.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
y_prob_o = model_orig.predict_proba(test_X)[:, 1]
y_pred_o = model_orig.predict(test_X)

results["original"] = {
    "auc": metrics.roc_auc_score(test_y, y_prob_o),
    "f1": metrics.f1_score(test_y, y_pred_o),
    "mcc": metrics.matthews_corrcoef(test_y, y_pred_o),
    "acc": metrics.accuracy_score(test_y, y_pred_o),
    "N_total": len(y_all), "N_pos": int(y_all.sum()), "N_neg": int((y_all==0).sum()),
    "test_N": len(test_y)
}
print(f"  AUC: {results['original']['auc']:.4f}, F1: {results['original']['f1']:.4f}")

# Summary
print("\n=== CD-HIT Robustness: ESM-2 150M Features ===")
print(f"{'Threshold':<15} {'N':>6} {'TestN':>6} {'AUC':>8} {'F1':>8} {'MCC':>8} {'ACC':>8}")
print("-"*65)
for label in ["original", "90", "70", "40"]:
    r = results[label]
    print(f"{label:<15} {r['N_total']:>6} {r['test_N']:>6} {r['auc']:>8.4f} {r['f1']:>8.4f} {r['mcc']:>8.4f} {r['acc']:>8.4f}")

with open(r"D:\Research_AI_Bio\03_Datasets\Final\cdhit_xgb_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDone.")
