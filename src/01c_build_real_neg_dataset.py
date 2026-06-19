"""Create new dataset with real non-AMP negative samples"""
import pandas as pd, os
from sklearn.model_selection import train_test_split

PROCESSED = "D:/Research_AI_Bio/03_Datasets/Processed"
RAW = "D:/Research_AI_Bio/02_Databases"

# 1. AMP positives
amp_df = pd.read_csv(os.path.join(PROCESSED, "amp_data_raw.csv"))
amp_pos = amp_df[amp_df["label_amp"] == 1].copy()
print(f"AMP positives: {len(amp_pos)}")

# 2. Real non-AMP negatives from UniProt
neg_files = [os.path.join(RAW, "uniprot_nonamp_p1.tsv")]
amp_seqs = set(s.upper().strip() for s in amp_pos["sequence"] if isinstance(s, str))
neg_records = []
seen = set()
for f in neg_files:
    if os.path.exists(f):
        df = pd.read_csv(f, sep="\t")
        for _, row in df.iterrows():
            seq = str(row["Sequence"]).upper().strip()
            if seq and len(seq) >= 5 and seq not in amp_seqs and seq not in seen:
                seen.add(seq)
                neg_records.append({"sequence": seq, "source": "UniProt_nonAMP", "label_amp": 0, "label_hemolysis": -1})

neg_df = pd.DataFrame(neg_records)
print(f"Real non-AMP negatives (valid): {len(neg_df)}")

# 3. Combine and split
combined = pd.concat([amp_pos, neg_df], ignore_index=True)
X_train, X_temp = train_test_split(combined, test_size=0.2, random_state=42, stratify=combined["label_amp"])
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42, stratify=X_temp["label_amp"])

# 4. Save
X_train.to_csv(os.path.join(PROCESSED, "amp_train_real.csv"), index=False, encoding="utf-8-sig")
X_val.to_csv(os.path.join(PROCESSED, "amp_val_real.csv"), index=False, encoding="utf-8-sig")
X_test.to_csv(os.path.join(PROCESSED, "amp_test_real.csv"), index=False, encoding="utf-8-sig")
combined.to_csv(os.path.join(PROCESSED, "amp_data_real_neg.csv"), index=False, encoding="utf-8-sig")

print(f"\nTrain: {len(X_train)} (AMP+={X_train['label_amp'].sum()} / non={(X_train['label_amp']==0).sum()})")
print(f"Val:   {len(X_val)} (AMP+={X_val['label_amp'].sum()} / non={(X_val['label_amp']==0).sum()})")
print(f"Test:  {len(X_test)} (AMP+={X_test['label_amp'].sum()} / non={(X_test['label_amp']==0).sum()})")
print(f"Total: {len(combined)} sequences")
print("Saved!")
