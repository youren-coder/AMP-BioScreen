import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sys, os
_utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_utils_dir, '..'))
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR

# Load UniDL4BioPep training data stats
train_udl = pd.read_csv(TOOLS_DIR / "UniDL4BioPep_AMP/AMP_train.csv")
pos = train_udl[train_udl["label"] == 1]
neg = train_udl[train_udl["label"] == 0]

pos_lens = pos["sequence"].str.len()
neg_lens = neg["sequence"].str.len()

print("UniDL4BioPep training data:")
print(f"  Pos: n={len(pos)}, median len={pos_lens.median():.0f}, range=[{pos_lens.min()}-{pos_lens.max()}]")
print(f"  Neg: n={len(neg)}, median len={neg_lens.median():.0f}, range=[{neg_lens.min()}-{neg_lens.max()}]")
print(f"  Length ratio (neg/pos median): {neg_lens.median()/pos_lens.median():.1f}x")

# What AUC would you get using ONLY length as predictor?
all_lens = pd.concat([pos_lens, neg_lens]).values.reshape(-1, 1)
all_labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
lr = LogisticRegression()
lr.fit(all_lens, all_labels)
y_prob = lr.predict_proba(all_lens)[:, 1]
auc_length_only = roc_auc_score(all_labels, y_prob)
print(f"\nLength-only AUC on UniDL4BioPep training data: {auc_length_only:.4f}")

# Compare: our training data length-only AUC
our_pos = pd.read_csv(PROCESSED_DIR / "amp_data_new_neg.csv")
our_pos = our_pos[our_pos["label_amp"] == 1]
our_neg = pd.read_csv(PROCESSED_DIR / "amp_data_new_neg.csv")
our_neg = our_neg[our_neg["label_amp"] == 0]
our_pl = our_pos["sequence"].str.len()
our_nl = our_neg["sequence"].str.len()
our_all_lens = np.concatenate([our_pl, our_nl]).values.reshape(-1, 1)
our_all_labels = np.concatenate([np.ones(len(our_pl)), np.zeros(len(our_nl))])
lr2 = LogisticRegression()
lr2.fit(our_all_lens, our_all_labels)
our_auc_len = roc_auc_score(our_all_labels, lr2.predict_proba(our_all_lens)[:, 1])
print(f"Length-only AUC on OUR training data (short secreted neg): {our_auc_len:.4f}")

print(f"\nKey insight: UniDL4BioPep training data has {neg_lens.median()/pos_lens.median():.1f}x length gap")
print(f"  -> even pure length achieves AUC={auc_length_only:.4f}")
print(f"  -> model only needs to learn 'short vs long' to get high AUC on its own test set")
print(f"  -> but our test set has matched lengths -> model collapses")
