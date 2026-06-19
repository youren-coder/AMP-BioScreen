"""
08_hemolysis_prediction.py — 溶血毒性预测
基于 Hemolytik2 数据的溶血/非溶血二分类。
"""

import warnings, time, os, json
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

HEMO_CSV = Path("D:/Research_AI_Bio/02_Databases/hemolytik2_complete.csv")
FEATURE_DIR = Path("D:/Research_AI_Bio/03_Datasets/Processed/features")
OUTPUT = {}

print("=" * 55)
print("  溶血毒性预测 (Hemolytik2)")
print("=" * 55)

# 1. 加载数据
print("\n[1/6] 加载并过滤数据...")
df = pd.read_csv(HEMO_CSV)
print(f"  总记录: {len(df)}")

df_valid = df[df["non_hem"].isin(["Non-hemolytic", "Low hemolytic"])].copy()
n_non = (df_valid["non_hem"] == "Non-hemolytic").sum()
n_low = (df_valid["non_hem"] == "Low hemolytic").sum()
print(f"  有效: {len(df_valid)} (非溶血={n_non}, 低溶血={n_low})")

df_valid["label"] = (df_valid["non_hem"] == "Low hemolytic").astype(int)
sequences_raw = df_valid["seq"].tolist()

# Clean sequences for ESM-2 (remove non-standard amino acids)
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
def clean_seq(s):
    return "".join(c if c in VALID_AA else "A" for c in s.strip().upper())
sequences = [clean_seq(s) for s in sequences_raw]
y = df_valid["label"].values

# 2. 理化特征
print("\n[2/6] 提取理化特征 (peptides)...")
from peptides import Peptide

def extract_physio(seqs):
    rows = []
    for seq in seqs:
        try:
            p = Peptide(seq)
            c = p.counts()
            pos = c.get("K", 0) + c.get("R", 0) + c.get("H", 0)
            neg = c.get("D", 0) + c.get("E", 0)
            total = sum(c.values())
            rows.append({
                "length": len(seq), "mw": p.molecular_weight(),
                "charge": p.charge(pH=7), "pI": p.isoelectric_point(),
                "hydrophob": p.hydrophobicity(), "hmoment": p.hydrophobic_moment(),
                "instability": p.instability_index(), "aliphatic": p.aliphatic_index(),
                "boman": p.boman(), "entropy": p.entropy(),
                "pos_charge": pos, "neg_charge": neg,
                "charge_ratio": (pos - neg) / total if total > 0 else 0,
                "pos_ratio": pos / total if total > 0 else 0,
                "neg_ratio": neg / total if total > 0 else 0,
            })
        except:
            rows.append({k: np.nan for k in [
                "length","mw","charge","pI","hydrophob","hmoment",
                "instability","aliphatic","boman","entropy",
                "pos_charge","neg_charge","charge_ratio","pos_ratio","neg_ratio"]})
    return pd.DataFrame(rows)

X_physio = extract_physio(sequences).values.astype(np.float32)
print(f"  理化特征: {X_physio.shape}")

# 3. ESM-2 embedding (model cached, load is fast)
print("\n[3/6] 提取 ESM-2 嵌入 (CPU)...")
import torch, esm

t0 = time.time()
model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t12_35M_UR50D")
model.eval()
model.to(torch.device("cpu"))
bc = alphabet.get_batch_converter()
print(f"  模型加载: {time.time()-t0:.0f}s")

X_esm = []
batch_size = 4
n = len(sequences)
t0 = time.time()
for i in range(0, n, batch_size):
    batch = sequences[i:i+batch_size]
    _, _, tokens = bc([(str(j), s) for j, s in enumerate(batch)])
    tokens = tokens.to("cpu")
    with torch.no_grad():
        results = model(tokens, repr_layers=[12])
        emb = results["representations"][12]
        for j in range(len(batch)):
            m = tokens[j] != alphabet.padding_idx
            X_esm.append(emb[j, m].mean(dim=0).numpy())
    if (i // batch_size) % 200 == 0:
        elapsed = time.time() - t0
        rate = (i + batch_size) / elapsed if elapsed > 0 else 0
        print(f"  [{min(i+batch_size, n)}/{n}] {rate:.0f} seq/s", end="\r")

X_esm = np.array(X_esm, dtype=np.float32)
print(f"\n  ESM-2嵌入: {X_esm.shape}, 耗时 {time.time()-t0:.0f}s")
# 4. 合并特征
X_combined = np.concatenate([X_physio, X_esm], axis=1)
print(f"\n[4/6] 合并特征: {X_combined.shape}")

# 5. 交叉验证
print("\n[5/6] 5折交叉验证 (含特征消融)...")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
import xgboost as xgb

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_all = {"physio": [], "esm": [], "combined": []}

for fold, (train_idx, test_idx) in enumerate(skf.split(X_combined, y)):
    outputs = []
    for fname, X_f in [("physio", X_physio), ("esm", X_esm), ("combined", X_combined)]:
        X_tr, X_te = X_f[train_idx], X_f[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        neg, pos = (y_tr == 0).sum(), y_tr.sum()
        model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=neg/pos, random_state=42, n_jobs=-1, verbosity=0)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)
        cv_all[fname].append({
            "AUC": round(roc_auc_score(y_te, y_prob), 4),
            "AUPR": round(average_precision_score(y_te, y_prob), 4),
            "F1": round(f1_score(y_te, y_pred), 4),
            "MCC": round(matthews_corrcoef(y_te, y_pred), 4),
        })
    print(f"  折{fold+1}: AUC physio={cv_all['physio'][-1]['AUC']:.4f} esm={cv_all['esm'][-1]['AUC']:.4f} combined={cv_all['combined'][-1]['AUC']:.4f}")

# 6. 汇总
print("\n[6/6] 结果汇总 (5折CV均值±标准差)")
print("=" * 60)
print(f"{'指标':<8} {'仅理化':<18} {'仅ESM-2':<18} {'融合特征':<18}")
print("-" * 60)
for metric in ["AUC", "AUPR", "F1", "MCC"]:
    row = f"{metric:<8}"
    for name in ["physio", "esm", "combined"]:
        vals = [r[metric] for r in cv_all[name]]
        row += f" {np.mean(vals):.4f}±{np.std(vals):.4f}  "
    print(row)
print("=" * 60)

# 保存
OUTPUT = {name: {m: [r[m] for r in vals] for m in vals[0].keys()} for name, vals in cv_all.items()}
with open(FEATURE_DIR / "hemolysis_results.json", "w") as f:
    json.dump(OUTPUT, f, indent=2)
print("\n结果已保存: hemolysis_results.json")
print("DONE")
