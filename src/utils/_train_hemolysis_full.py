import pandas as pd, numpy as np, os, time, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

from pathlib import Path
import torch, esm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import json

PROCESSED = Path("D:/Research_AI_Bio/03_Datasets/Processed")
FEATURES = PROCESSED / "features"

# Load labels - filter to only valid amino acid sequences
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
df = pd.read_csv("D:/Research_AI_Bio/02_Databases/hemolytik2_complete.csv")
df['seq'] = df['seq'].str.upper().str.strip()

# Filter: only sequences with standard amino acids, length 5-100
def is_valid(seq):
    return all(aa in VALID_AA for aa in seq) and 5 <= len(seq) <= 100

df = df[df['seq'].apply(is_valid)]
df['hemo_label'] = (df['activity'].astype(str).str.upper().str.strip() == 'HEMOLYTIC').astype(int)
uniq = df.groupby('seq')['hemo_label'].max().reset_index()
print(f"Valid Hemolytik2 unique: {len(uniq)}, hemolytic={uniq['hemo_label'].sum()}")

# Build existing embedding lookup
seq_to_esm = {}
for split_name, esm_file in [
    ("amp_train_real", "amp_train_esm2_150m.npy"),
    ("amp_val_real", "amp_val_esm2_150m.npy"),
    ("amp_test_real", "amp_test_esm2_150m.npy"),
]:
    df_split = pd.read_csv(PROCESSED / f"{split_name}.csv")
    X_esm = np.load(FEATURES / esm_file)
    for i, row in df_split.iterrows():
        seq_to_esm[row["sequence"].upper().strip()] = X_esm[i]

existing = set(seq_to_esm.keys())
new_to_extract = [s for s in uniq['seq'] if s not in existing]
print(f"Need to extract: {len(new_to_extract)}")
t0 = time.time()

if new_to_extract:
    print(f"Extracting ESM-2 150M for {len(new_to_extract)} sequences...")
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t30_150M_UR50D")
    device = torch.device("cpu")
    model = model.to(device).eval()
    bc = alphabet.get_batch_converter()
    
    for i in range(0, len(new_to_extract), 2):
        batch = new_to_extract[i:i+2]
        data = [(str(j), s) for j, s in enumerate(batch)]
        _, _, tokens = bc(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[30])
            emb = results["representations"][30]
            for j in range(len(batch)):
                mask = tokens[j] != alphabet.padding_idx
                seq_to_esm[batch[j]] = emb[j, mask].mean(dim=0).cpu().numpy().astype(np.float32)
        done = min(i+2, len(new_to_extract))
        if done % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {done}/{len(new_to_extract)} ({done/elapsed:.1f}/s, ETA {(len(new_to_extract)-done)*elapsed/done/60:.0f}min)")
    
    print(f"  Done in {(time.time()-t0)/60:.1f} min")

# Build training data
X_list, y_list = [], []
for _, row in uniq.iterrows():
    seq = row['seq']
    if seq in seq_to_esm:
        X_list.append(seq_to_esm[seq])
        y_list.append(row['hemo_label'])

X_hemo = np.array(X_list, dtype=np.float32)
y_hemo = np.array(y_list)
print(f"\nTraining data: {X_hemo.shape}, hemolytic={y_hemo.sum()}")

# 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
imbalance = (len(y_hemo) - y_hemo.sum()) / y_hemo.sum()
cv_aucs = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_hemo, y_hemo)):
    X_tr, X_vl = X_hemo[tr_idx], X_hemo[val_idx]
    y_tr, y_vl = y_hemo[tr_idx], y_hemo[val_idx]
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=imbalance,
        early_stopping_rounds=30, eval_metric='auc',
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    auc = roc_auc_score(y_vl, model.predict_proba(X_vl)[:, 1])
    cv_aucs.append(auc)

# Final model
final = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=imbalance,
    early_stopping_rounds=30, eval_metric='auc', random_state=42, n_jobs=-1)
final.fit(X_hemo, y_hemo, eval_set=[(X_hemo, y_hemo)], verbose=False)

# Predict on all AMPs
amp_all = pd.read_csv(PROCESSED / "amp_data_new_neg.csv")
amp_pos = amp_all[amp_all['label_amp'] == 1]
hemo_preds = []
for _, row in amp_pos.iterrows():
    seq = row['sequence'].upper().strip()
    hemo_preds.append(float(final.predict_proba(seq_to_esm.get(seq, np.zeros((1,640))).reshape(1,-1))[0,1]) if seq in seq_to_esm else 0.5)

hemo_preds = np.array(hemo_preds)
low_tox = (hemo_preds < 0.3).sum()
print(f"\nAMP hemolysis (n={len(amp_pos)}): non-hemolytic(p<0.3)={low_tox} ({100*low_tox/len(amp_pos):.1f}%)")
print(f"5-fold CV AUC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")

with open(str(FEATURES / "hemolysis_cv_results.json"), "w") as f:
    json.dump({"cv_auc_mean": float(np.mean(cv_aucs)), "cv_auc_std": float(np.std(cv_aucs)),
               "n_train": len(X_hemo), "n_hemolytic": int(y_hemo.sum()),
               "n_amp_non_hemolytic": int(low_tox), "n_amp_total": int(len(amp_pos))}, f, indent=2)
print("DONE.")
