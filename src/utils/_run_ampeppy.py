import sys
sys.path.insert(0, r"D:\Research_AI_Bio\01_Projects\AMP-BioScreen\.venv\Lib\site-packages\amPEPpy")
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from Bio import SeqIO
import amPEPpy.amPEP as ampep

# Reload model (already trained)
with open(r"D:\Research_AI_Bio\03_Datasets\Final\ampeppy_model.pkl", "rb") as f:
    clf = pickle.load(f)

# Build test fasta with all sequences
test_csv = pd.read_csv(r"D:\Research_AI_Bio\03_Datasets\Processed\amp_test_real.csv")
with open(r"D:\Research_AI_Bio\03_Datasets\Final\test_all.fasta", "w") as f:
    for idx, row in test_csv.iterrows():
        label = "pos" if row["label_amp"]==1 else "neg"
        f.write(">test{}_".format(idx) + label + "\n" + str(row["sequence"]) + "\n")

# Now score
with open(r"D:\Research_AI_Bio\03_Datasets\Final\test_all.fasta", "r") as tf:
    test_df = ampep.score(tf)

y_true = test_csv["label_amp"].values
y_prob = clf.predict_proba(test_df)[:, 1]
y_pred = clf.predict(test_df)

print("=== amPEPpy Direct Comparison on Our Test Set ===")
print("Test samples:", len(y_true))
print("AUC-ROC:", round(metrics.roc_auc_score(y_true, y_prob), 4))
print("F1:", round(metrics.f1_score(y_true, y_pred), 4))
print("MCC:", round(metrics.matthews_corrcoef(y_true, y_pred), 4))
print("Accuracy:", round(metrics.accuracy_score(y_true, y_pred), 4))
print("Precision:", round(metrics.precision_score(y_true, y_pred), 4))
print("Recall:", round(metrics.recall_score(y_true, y_pred), 4))

results = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred})
results.to_csv(r"D:\Research_AI_Bio\03_Datasets\Final\ampeppy_test_results.csv", index=False)
print("Done.")
