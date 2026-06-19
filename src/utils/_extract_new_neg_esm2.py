"""
Extract ESM-2 150M embeddings for 600 new negatives,
compute physchem, build feature matrices, and train XGBoost comparison.

Usage: .venv\Scripts\python.exe src\_extract_new_neg_esm2.py
"""
import os, sys, time, warnings, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from pathlib import Path
from peptides import Peptide

PROCESSED = Path("D:/Research_AI_Bio/03_Datasets/Processed")
FEATURE_DIR = PROCESSED / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# STEP 1: Physchem for 600 new negatives
# ===================================================================
print("STEP 1: Physicochemical features...")
new_neg_df = pd.read_csv("D:/Research_AI_Bio/02_Databases/neg_short_secreted_matched.csv")
new_seqs = [s.strip().upper() for s in new_neg_df["sequence"].tolist()]

phys_rows = []
for seq in new_seqs:
    try:
        p = Peptide(seq)
        c = p.counts()
        pos = c.get("K", 0) + c.get("R", 0) + c.get("H", 0)
        neg = c.get("D", 0) + c.get("E", 0)
        total = sum(c.values()) or 1
        phys_rows.append(np.array([
            len(seq), p.molecular_weight(), p.charge(pH=7), p.isoelectric_point(),
            p.hydrophobicity(), p.hydrophobic_moment(), p.instability_index(),
            p.aliphatic_index(), p.boman(), p.entropy(), p.hydrophobicity(),
            pos, neg, (pos - neg) / total, pos / total, neg / total
        ], dtype=np.float32))
    except Exception as e:
        phys_rows.append(np.zeros(16, dtype=np.float32))

seq_to_physchem = {s: phys_rows[i] for i, s in enumerate(new_seqs)}
print(f"  Physchem: {len(seq_to_physchem)} entries, dim=16")

# ===================================================================
# STEP 2: ESM-2 150M extraction
# ===================================================================
print("\nSTEP 2: ESM-2 150M embedding...")
import torch, esm

model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t30_150M_UR50D")
device = torch.device("cpu")
model = model.to(device).eval()
esm_dim = model.embed_dim
print(f"  Model: esm2_t30_150M_UR50D, embed_dim={esm_dim}")

bc = alphabet.get_batch_converter()
batch_size = 2
seq_to_esm = {}
t0 = time.time()
n = len(new_seqs)

for i in range(0, n, batch_size):
    batch_seqs = new_seqs[i:i + batch_size]
    data = [(str(j), s) for j, s in enumerate(batch_seqs)]
    _, _, tokens = bc(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[30])
        emb = results["representations"][30]
        for j in range(len(batch_seqs)):
            mask = tokens[j] != alphabet.padding_idx
            vec = emb[j, mask].mean(dim=0).cpu().numpy().astype(np.float32)
            seq_to_esm[batch_seqs[j]] = vec

    done = min(i + batch_size, n)
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (n - done) / rate if rate > 0 else 0
    if done % 20 == 0 or done == n:
        print(f"  [{done}/{n}] {rate:.2f} seq/s, elapsed {elapsed/60:.0f}min, ETA {eta/60:.0f}min")

total = time.time() - t0
print(f"  ESM-2 done: {total/60:.1f}min, {len(seq_to_esm)} embeddings")

# Fuse
new_neg_fused = {}
for seq in new_seqs:
    phys = seq_to_physchem.get(seq, np.zeros(16, dtype=np.float32))
    esm = seq_to_esm.get(seq, np.zeros(esm_dim, dtype=np.float32))
    new_neg_fused[seq] = np.concatenate([phys, esm]).astype(np.float32)

fused_dim = 16 + esm_dim
print(f"  Fused dim: {fused_dim}")

np.savez(FEATURE_DIR / "new_neg_150m_fused.npz",
         sequences=np.array(new_seqs),
         embeddings=np.array([new_neg_fused[s] for s in new_seqs]))
print(f"  Saved: new_neg_150m_fused.npz")

# ===================================================================
# STEP 3: Build feature matrices
# ===================================================================
print("\nSTEP 3: Building feature matrices for new_neg and hybrid_neg...")

old_train = pd.read_csv(PROCESSED / "amp_train_real.csv")
old_val = pd.read_csv(PROCESSED / "amp_val_real.csv")
old_test = pd.read_csv(PROCESSED / "amp_test_real.csv")

old_train_X = np.load(FEATURE_DIR / "amp_train_X_150m.npy")
old_val_X = np.load(FEATURE_DIR / "amp_val_X_150m.npy")
old_test_X = np.load(FEATURE_DIR / "amp_test_X_150m.npy")

seq_to_old_fused = {}
for i, row in old_train.iterrows():
    seq_to_old_fused[row["sequence"].upper().strip()] = old_train_X[i]
for i, row in old_val.iterrows():
    seq_to_old_fused[row["sequence"].upper().strip()] = old_val_X[i]
for i, row in old_test.iterrows():
    seq_to_old_fused[row["sequence"].upper().strip()] = old_test_X[i]

for ds_name in ["new_neg", "hybrid_neg"]:
    print(f"\n  Dataset: {ds_name}")
    for split in ["train", "val", "test"]:
        df = pd.read_csv(PROCESSED / f"amp_{split}_{ds_name}.csv")
        X_list, y_list = [], []
        missing = 0
        for _, row in df.iterrows():
            seq = row["sequence"].upper().strip()
            if seq in seq_to_old_fused:
                X_list.append(seq_to_old_fused[seq])
            elif seq in new_neg_fused:
                X_list.append(new_neg_fused[seq])
            else:
                missing += 1
                X_list.append(np.zeros(fused_dim, dtype=np.float32))
            y_list.append(row["label_amp"])
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        np.save(FEATURE_DIR / f"amp_{split}_X_150m_{ds_name}.npy", X)
        np.save(FEATURE_DIR / f"amp_{split}_y_amp_{ds_name}.npy", y)
        print(f"    {split}: X={X.shape}, pos={y.sum()}, missing={missing}")

# ===================================================================
# STEP 4: Train XGBoost
# ===================================================================
print("\n" + "=" * 60)
print("STEP 4: XGBoost training comparison")
print("=" * 60)

import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score

def train_eval(ds_name):
    print(f"\n--- {ds_name} ---")
    X_train = np.load(FEATURE_DIR / f"amp_train_X_150m_{ds_name}.npy")
    y_train = np.load(FEATURE_DIR / f"amp_train_y_amp_{ds_name}.npy")
    X_val = np.load(FEATURE_DIR / f"amp_val_X_150m_{ds_name}.npy")
    y_val = np.load(FEATURE_DIR / f"amp_val_y_amp_{ds_name}.npy")
    X_test = np.load(FEATURE_DIR / f"amp_test_X_150m_{ds_name}.npy")
    y_test = np.load(FEATURE_DIR / f"amp_test_y_amp_{ds_name}.npy")

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_weight = n_neg / n_pos if n_pos > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        early_stopping_rounds=50, eval_metric='auc',
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(f"  Best iter: {model.best_iteration}")
    print(f"  AUC:  {auc:.4f}")
    print(f"  F1:   {f1:.4f}")
    print(f"  MCC:  {mcc:.4f}")
    print(f"  ACC:  {acc:.4f}")
    return {"auc": auc, "f1": f1, "mcc": mcc, "acc": acc, "best_iter": model.best_iteration}

results = {}
with open(FEATURE_DIR / "comprehensive_results.json") as f:
    old_res = json.load(f)
results["old_neg_baseline"] = {
    "auc": old_res.get("auc", 0.984),
    "f1": old_res.get("f1", 0.962),
    "mcc": old_res.get("mcc", 0.884),
    "acc": old_res.get("acc", 0.927),
}
print(f"\nBaseline AUC: {results['old_neg_baseline']['auc']:.4f}")

results["new_neg_short_secreted"] = train_eval("new_neg")
results["hybrid_neg"] = train_eval("hybrid_neg")

# ===================================================================
# Summary
# ===================================================================
print("\n" + "=" * 60)
print("RESULTS: Negative Sample Comparison")
print("=" * 60)
print(f"{'Dataset':<30} {'AUC':>8} {'F1':>8} {'MCC':>8} {'ACC':>8}")
print("-" * 60)
for name, res in results.items():
    print(f"{name:<30} {res['auc']:8.4f} {res['f1']:8.4f} {res['mcc']:8.4f} {res['acc']:8.4f}")

with open(FEATURE_DIR / "neg_sample_comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: neg_sample_comparison_results.json")
print("DONE.")
