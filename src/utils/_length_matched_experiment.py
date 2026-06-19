"""
Length-matched experiment v2: strict per-bin 1:1 matching.
Fixed: separate ESM-only vs fused embeddings.
"""
import os, sys, time, warnings, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score
import xgboost as xgb
from peptides import Peptide

PROCESSED = Path("D:/Research_AI_Bio/03_Datasets/Processed")
FEATURES = PROCESSED / "features"
FEATURES.mkdir(parents=True, exist_ok=True)

ESM_DIM = 640
PHYS_DIM = 16
FUSED_DIM = ESM_DIM + PHYS_DIM

# ============================================================
# STEP 1: Build ESM-only lookup from all existing sources
# ============================================================
print("STEP 1: Building ESM-only embedding lookup...")

esm_lookup = {}  # seq -> 640-dim ESM-2 embedding

# From old datasets (fused = physchem 16 + esm 640), extract esm part
for split in ["train", "val", "test"]:
    df = pd.read_csv(PROCESSED / f"amp_{split}_real.csv")
    X_fused = np.load(FEATURES / f"amp_{split}_X_150m.npy")  # (n, 656)
    X_esm = X_fused[:, PHYS_DIM:]  # (n, 640)
    for i, row in df.iterrows():
        esm_lookup[row["sequence"].upper().strip()] = X_esm[i].astype(np.float32)

# From new_neg fused
new_neg_npz = np.load(str(FEATURES / "new_neg_150m_fused.npz"), allow_pickle=True)
new_neg_fused = new_neg_npz["embeddings"]  # (600, 656)
new_neg_esm = new_neg_fused[:, PHYS_DIM:]  # (600, 640)
for seq, emb in zip(new_neg_npz["sequences"], new_neg_esm):
    esm_lookup[str(seq)] = emb.astype(np.float32)

print(f"  ESM lookup: {len(esm_lookup)} entries, dim={ESM_DIM}")

# ============================================================
# STEP 2: Identify missing sequences
# ============================================================
lm = pd.read_csv(PROCESSED / "amp_data_length_matched.csv")
lm_seqs = lm["sequence"].str.upper().str.strip().tolist()
existing = set(esm_lookup.keys())

missing_seqs = []
for s in lm_seqs:
    if s not in existing:
        missing_seqs.append(s)
missing_seqs = list(dict.fromkeys(missing_seqs))

print(f"  Total: {len(lm_seqs)}, Have: {len(lm_seqs)-len(missing_seqs)}, Need: {len(missing_seqs)}")

# ============================================================
# STEP 3: ESM-2 extraction for missing
# ============================================================
if missing_seqs:
    print(f"\nSTEP 2: ESM-2 150M for {len(missing_seqs)} sequences...")
    import torch, esm
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t30_150M_UR50D")
    device = torch.device("cpu")
    model = model.to(device).eval()

    bc = alphabet.get_batch_converter()
    batch_size = 2
    t0 = time.time()
    n = len(missing_seqs)

    for i in range(0, n, batch_size):
        batch_seqs = missing_seqs[i:i + batch_size]
        data = [(str(j), s) for j, s in enumerate(batch_seqs)]
        _, _, tokens = bc(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[30])
            emb = results["representations"][30]
            for j in range(len(batch_seqs)):
                mask = tokens[j] != alphabet.padding_idx
                vec = emb[j, mask].mean(dim=0).cpu().numpy().astype(np.float32)
                esm_lookup[batch_seqs[j]] = vec
        done = min(i + batch_size, n)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        if done % 100 == 0 or done == n:
            print(f"  [{done}/{n}] {rate:.2f} seq/s, {elapsed/60:.0f}min elapsed")

    print(f"  Done: {time.time()-t0:.0f}s, lookup now {len(esm_lookup)} entries")
else:
    print("\nSTEP 2: No extraction needed.")

# ============================================================
# STEP 4: Physchem for all sequences + fuse
# ============================================================
print("\nSTEP 3: Computing physchem + fusing features...")

# Build physchem cache for all sequences in lm
physchem_cache = {}
# Pre-compute for efficiency: batch all unique sequences
all_lm_seqs_unique = list(set(lm_seqs))
print(f"  Computing physchem for {len(all_lm_seqs_unique)} unique sequences...")

for seq in all_lm_seqs_unique:
    try:
        p = Peptide(seq)
        c = p.counts()
        pos = c.get("K", 0) + c.get("R", 0) + c.get("H", 0)
        neg = c.get("D", 0) + c.get("E", 0)
        total = sum(c.values()) or 1
        physchem_cache[seq] = np.array([
            len(seq), p.molecular_weight(), p.charge(pH=7), p.isoelectric_point(),
            p.hydrophobicity(), p.hydrophobic_moment(), p.instability_index(),
            p.aliphatic_index(), p.boman(), p.entropy(), p.hydrophobicity(),
            pos, neg, (pos - neg) / total, pos / total, neg / total
        ], dtype=np.float32)
    except Exception:
        physchem_cache[seq] = np.zeros(PHYS_DIM, dtype=np.float32)

# Build fused
X_list = []
y_list = []
missing_count = 0
for _, row in lm.iterrows():
    seq = row["sequence"].upper().strip()
    esm_emb = esm_lookup.get(seq)
    phys = physchem_cache.get(seq, np.zeros(PHYS_DIM, dtype=np.float32))
    
    if esm_emb is None:
        esm_emb = np.zeros(ESM_DIM, dtype=np.float32)
        missing_count += 1
    
    fused = np.concatenate([phys, esm_emb]).astype(np.float32)
    X_list.append(fused)
    y_list.append(row["label_amp"])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)
X = np.nan_to_num(X, nan=0.0)

print(f"  X: {X.shape}, y: {y.shape}, pos={y.sum()}, neg={len(y)-y.sum()}, missing_embeddings={missing_count}")

# Verify all rows same dimension
assert X.shape[1] == FUSED_DIM, f"Expected {FUSED_DIM}, got {X.shape[1]}"

# ============================================================
# STEP 5: Train/val/test split + XGBoost
# ============================================================
print("\nSTEP 4: XGBoost training...")

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

print(f"  Train: {X_tr.shape[0]}, pos={y_tr.sum()}")
print(f"  Val:   {X_val.shape[0]}, pos={y_val.sum()}")
print(f"  Test:  {X_test.shape[0]}, pos={y_test.sum()}")

model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=1.0,
    early_stopping_rounds=50, eval_metric='auc',
    random_state=42, n_jobs=-1,
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc_val = roc_auc_score(y_test, y_prob)
f1_val = f1_score(y_test, y_pred)
mcc_val = matthews_corrcoef(y_test, y_pred)
acc_val = accuracy_score(y_test, y_pred)

print(f"\n  === STRICT Length-Matched 1:1 Results ===")
print(f"  Best iter: {model.best_iteration}")
print(f"  AUC:  {auc_val:.4f}")
print(f"  F1:   {f1_val:.4f}")
print(f"  MCC:  {mcc_val:.4f}")
print(f"  ACC:  {acc_val:.4f}")

# 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = []
for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr_f, X_val_f = X[tr_idx], X[val_idx]
    y_tr_f, y_val_f = y[tr_idx], y[val_idx]
    m = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=50, eval_metric='auc',
        random_state=42, n_jobs=-1,
    )
    m.fit(X_tr_f, y_tr_f, eval_set=[(X_val_f, y_val_f)], verbose=False)
    yp = m.predict_proba(X_val_f)[:, 1]
    auc_f = roc_auc_score(y_val_f, yp)
    cv_aucs.append(auc_f)

print(f"\n  5-fold CV AUC: {np.mean(cv_aucs):.4f} +- {np.std(cv_aucs):.4f}")

# ============================================================
# STEP 6: Summary + save
# ============================================================
results = {
    "experiment": "length_matched_strict_1to1",
    "description": "Per-bin exact 1:1 matching, 915 pos + 915 neg, 1830 total",
    "test_auc": float(auc_val), "test_f1": float(f1_val),
    "test_mcc": float(mcc_val), "test_acc": float(acc_val),
    "cv_auc_mean": float(np.mean(cv_aucs)), "cv_auc_std": float(np.std(cv_aucs)),
    "best_iter": int(model.best_iteration),
}

with open(str(FEATURES / "length_matched_strict_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("FULL COMPARISON")
print("=" * 70)
print(f"{'Experiment':<40} {'AUC':>8} {'F1':>8} {'MCC':>8}")
print("-" * 70)

rows = [
    ("Old neg (full-length non-AMP, n=443)", 0.984, 0.962, 0.884),
    ("New neg (short secreted, stratified, n=600)", 0.946, 0.897, 0.740),
    ("Hybrid neg (n=1043)", 0.928, 0.879, 0.789),
    ("STRICT length-matched 1:1 (n=1830)", auc_val, f1_val, mcc_val),
]
for label, a, f, m in rows:
    print(f"{label:<40} {a:8.4f} {f:8.4f} {m:8.4f}")

print("\nDONE.")
