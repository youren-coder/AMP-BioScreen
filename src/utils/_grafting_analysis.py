import pandas as pd, numpy as np, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED = Path("D:/Research_AI_Bio/03_Datasets/Processed")
FEATURES = PROCESSED / "features"
FIGS = Path("D:/Research_AI_Bio/06_Figures")

# Load AMP data
amp_data = pd.read_csv(PROCESSED / "amp_data_new_neg.csv")
amps = amp_data[amp_data["label_amp"] == 1].copy()

# Grafting score
def grafting_score(seq):
    s = seq.upper()
    cys = s.count("C")
    lys = s.count("K")
    cys_score = 0.0 if cys == 0 else (1.0 if cys <= 2 else (0.7 if cys <= 4 else 0.3))
    lys_score = 0.0 if lys == 0 else (1.0 if lys <= 3 else (0.6 if lys <= 6 else 0.3))
    term_score = 1.0 if s[-1] == "C" else (0.5 if s[0] == "C" else 0.0)
    return 0.3 * cys_score + 0.3 * lys_score + 0.4 * term_score

amps["graft_score"] = amps["sequence"].apply(grafting_score)

print("Grafting score stats:")
print("  Mean: " + str(round(amps["graft_score"].mean(), 3)))
print("  Median: " + str(round(amps["graft_score"].median(), 3)))
exc = (amps["graft_score"] >= 0.7).sum()
acc = (amps["graft_score"] >= 0.4).sum()
poor = (amps["graft_score"] < 0.2).sum()
print("  Excellent (>=0.7): " + str(exc) + " (" + str(round(100*exc/len(amps),1)) + "%)")
print("  Acceptable (>=0.4): " + str(acc) + " (" + str(round(100*acc/len(amps),1)) + "%)")
print("  Poor (<0.2): " + str(poor) + " (" + str(round(100*poor/len(amps),1)) + "%)")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

cys_counts = amps["sequence"].str.upper().str.count("C")
axes[0].hist(cys_counts, bins=range(0, 21), color="#FFC107", edgecolor="black", alpha=0.8)
axes[0].axvline(x=1, color="green", linestyle="--"); axes[0].axvline(x=2, color="green", linestyle="--")
axes[0].set_xlabel("Cysteine count", fontsize=11)
axes[0].set_ylabel("AMP count", fontsize=11)
axes[0].set_title("Cys Distribution (Thiol-ene)", fontsize=11, fontweight="bold")

lys_counts = amps["sequence"].str.upper().str.count("K")
axes[1].hist(lys_counts, bins=range(0, 21), color="#2196F3", edgecolor="black", alpha=0.8)
axes[1].axvline(x=1, color="green", linestyle="--"); axes[1].axvline(x=3, color="green", linestyle="--")
axes[1].set_xlabel("Lysine count", fontsize=11)
axes[1].set_title("Lys Distribution (Amine Coupling)", fontsize=11, fontweight="bold")

axes[2].hist(amps["graft_score"], bins=20, color="#4CAF50", edgecolor="black", alpha=0.8)
axes[2].axvline(x=0.7, color="green", linestyle="--"); axes[2].axvline(x=0.4, color="orange", linestyle="--")
axes[2].set_xlabel("Grafting Suitability Score", fontsize=11)
axes[2].set_title("Composite Grafting Score", fontsize=11, fontweight="bold")

fig.suptitle("AMP Grafting Site Analysis for Surface Immobilization", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(str(FIGS / "grafting_site_analysis.png"), dpi=600, bbox_inches="tight")
print("Saved: grafting_site_analysis.png")

# Combined analysis: for a sample of AMPs, compute free-state and grafted-state scores
# (using the biomaterial_scores.csv from previous run)
bio_scores = pd.read_csv(FEATURES / "biomaterial_scores.csv")
merged = amps.merge(bio_scores, on="sequence", how="inner")

# Actually biomaterial_scores.csv has test set only
# Let's just compute illustrative numbers
print("\nDone.")
