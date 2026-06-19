"""
09_biomaterial_scoring.py — 生物材料适用性评分框架

基于 AMP 活性 + 溶血毒性综合预测，生成生物材料适用性评分。
"""

import warnings, json
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

FEATURE_DIR = Path("D:/Research_AI_Bio/03_Datasets/Processed/features")
HEMO_CSV = Path("D:/Research_AI_Bio/02_Databases/hemolytik2_complete.csv")
DATA_DIR = Path("D:/Research_AI_Bio/03_Datasets/Processed")
FIGURE_DIR = Path("D:/Research_AI_Bio/06_Figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 1. 加载 AMP 模型
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 55)
print("  生物材料适用性评分")
print("=" * 55)

model_amp = xgb.XGBClassifier()
model_amp.load_model(str(FEATURE_DIR / "xgb_model.json"))
print("\n[1/4] AMP model loaded")

# 2. 训练溶血模型
print("\n[2/4] Training hemolysis model on full data...")
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
            })
        except:
            rows.append({k: np.nan for k in [
                "length","mw","charge","pI","hydrophob","hmoment",
                "instability","aliphatic","boman","entropy",
                "pos_charge","neg_charge","charge_ratio"]})
    return pd.DataFrame(rows)

df_hemo = pd.read_csv(HEMO_CSV)
df_hemo = df_hemo[df_hemo["non_hem"].isin(["Non-hemolytic", "Low hemolytic"])]
y_hemo = (df_hemo["non_hem"] == "Low hemolytic").astype(int).values

# Clean sequences
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
seqs_hemo = ["".join(c if c in VALID_AA else "A" for c in s.strip().upper()) for s in df_hemo["seq"].tolist()]
X_hemo = extract_physio(seqs_hemo).fillna(0).values.astype(np.float32)
print(f"  Hemolysis data: {X_hemo.shape}")

neg, pos = (y_hemo == 0).sum(), y_hemo.sum()
model_hemo = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=neg/pos, random_state=42, n_jobs=-1, verbosity=0)
model_hemo.fit(X_hemo, y_hemo)
print(f"  Trained (scale_pos_weight={neg/pos:.1f}, AMP={pos}/{len(y_hemo)})")

# 3. 对测试集进行综合评估
print("\n[3/4] Scoring test set...")
df_test = pd.read_csv(DATA_DIR / "amp_test.csv")
seqs_test = df_test["sequence"].tolist()
X_test = np.load(FEATURE_DIR / "amp_test_X.npy")
X_test_physio = extract_physio(seqs_test).fillna(0).values.astype(np.float32)

p_amp = model_amp.predict_proba(X_test)[:, 1]
p_hemo = model_hemo.predict_proba(X_test_physio)[:, 1]
suitability = p_amp * (1 - p_hemo)

# 4. 评分分类
def classify_score(s):
    if s >= 0.8: return "Excellent"
    elif s >= 0.6: return "Good"
    elif s >= 0.4: return "Fair"
    else: return "Poor"

results = pd.DataFrame({
    "sequence": seqs_test, "label_amp": df_test["label_amp"].values,
    "p_amp": p_amp, "p_hemo": p_hemo,
    "suitability": suitability,
    "category": [classify_score(s) for s in suitability],
})

# 按适用性排序输出
results_sorted = results.sort_values("suitability", ascending=False)
print(f"\n  Top 5 AMP candidates for biomaterials:")
for i, row in results_sorted[results_sorted["label_amp"] == 1].head(5).iterrows():
    print(f"    Seq: {row['sequence'][:30]}... P(AMP)={row['p_amp']:.3f} P(hemo)={row['p_hemo']:.3f} Score={row['suitability']:.3f} ({row['category']})")

# 统计各类别分布
print(f"\n  Suitability distribution (all test set):")
for cat in ["Excellent", "Good", "Fair", "Poor"]:
    subset = results[results["category"] == cat]
    print(f"    {cat}: {len(subset)} ({len(subset[subset['label_amp']==1])} AMPs)")

print(f"\n  Suitability distribution (AMPs only):")
amp_results = results[results["label_amp"] == 1]
for cat in ["Excellent", "Good", "Fair", "Poor"]:
    cnt = len(amp_results[amp_results["category"] == cat])
    print(f"    {cat}: {cnt} ({100*cnt/len(amp_results):.0f}%)")

plt.figure(figsize=(8, 6))
colors = {"Excellent": "#2ecc71", "Good": "#3498db", "Fair": "#f39c12", "Poor": "#e74c3c"}
for cat in ["Excellent", "Good", "Fair", "Poor"]:
    sub = amp_results[amp_results["category"] == cat]
    plt.scatter(sub["p_amp"], sub["p_hemo"], c=colors[cat], label=f"{cat} ({len(sub)})", alpha=0.7, s=60)
plt.xlabel("P(AMP Activity)")
plt.ylabel("P(Hemolysis)")
plt.title("Biomaterial Suitability Assessment")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "biomaterial_suitability.png", dpi=600)
plt.close()
print(f"\n  Plot saved: biomaterial_suitability.png")

# 保存结果
results.to_csv(FEATURE_DIR / "biomaterial_scores.csv", index=False)
with open(FEATURE_DIR / "biomaterial_summary.json", "w") as f:
    json.dump({
        "mean_suitability_amp": float(amp_results["suitability"].mean()),
        "top_amp_score": float(amp_results["suitability"].max()),
        "excellent_count": int(len(amp_results[amp_results["category"] == "Excellent"])),
        "good_count": int(len(amp_results[amp_results["category"] == "Good"])),
    }, f, indent=2)

print("\n[4/4] Results saved")
print("=" * 55)
