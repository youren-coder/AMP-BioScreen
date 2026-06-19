"""Feature extraction with ONLY verified peptides v0.5.0 methods"""
import warnings, os
import pandas as pd
import numpy as np
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from peptides import Peptide
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR

PROCESSED = PROCESSED_DIR
FEATURE_DIR = PROCESSED / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATHS = {
    "train": PROCESSED / "amp_train.csv",
    "val":   PROCESSED / "amp_val.csv",
    "test":  PROCESSED / "amp_test.csv",
}

def extract_features(sequences):
    rows = []
    for seq in sequences:
        try:
            p = Peptide(seq)
            c = p.counts()
            pos = c.get("K", 0) + c.get("R", 0) + c.get("H", 0)
            neg = c.get("D", 0) + c.get("E", 0)
            total = sum(c.values())
            rows.append({
                "length": len(seq),
                "mw": p.molecular_weight(),
                "charge": p.charge(pH=7),
                "pI": p.isoelectric_point(),
                "hydrophob": p.hydrophobicity(),
                "hmoment": p.hydrophobic_moment(),
                "instability": p.instability_index(),
                "aliphatic": p.aliphatic_index(),
                "boman": p.boman(),
                "entropy": p.entropy(),
                "gravy": p.hydrophobicity(),
                "pos_charge": pos,
                "neg_charge": neg,
                "charge_ratio": (pos - neg) / total if total > 0 else 0,
                "pos_ratio": pos / total if total > 0 else 0,
                "neg_ratio": neg / total if total > 0 else 0,
            })
        except Exception:
            rows.append({k: np.nan for k in [
                "length", "mw", "charge", "pI", "hydrophob", "hmoment",
                "instability", "aliphatic", "boman", "entropy", "gravy",
                "pos_charge", "neg_charge", "charge_ratio",
                "pos_ratio", "neg_ratio",
            ]})
    return pd.DataFrame(rows)

print("=" * 60)
print("  Feature Extraction with peptides v0.5.0")
print("=" * 60)

for name, path in DATA_PATHS.items():
    df = pd.read_csv(path)
    print(f"[{name}] {len(df)} sequences...")
    feat = extract_features(df["sequence"].tolist())
    for col in ["label_amp", "label_hemolysis"]:
        if col in df.columns:
            feat[col] = df[col].values
    out = FEATURE_DIR / f"features_physicochem_{name}.csv"
    feat.to_csv(out, index=False, encoding="utf-8-sig")

    nan_pct = 100 * feat.isnull().sum().sum() / (feat.shape[0] * feat.shape[1])
    status = "OK" if nan_pct == 0 else f"{nan_pct:.1f}% NaN"
    print(f"  -> {out.name} ({feat.shape}, {status})")
    if nan_pct == 0:
        print(f"     mw={feat.iloc[0]['mw']:.0f}, pI={feat.iloc[0]['pI']:.2f}")

print("\nDONE")
