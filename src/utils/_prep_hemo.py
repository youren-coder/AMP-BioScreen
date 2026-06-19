import pandas as pd, numpy as np
from pathlib import Path
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR

# Hemolytik2 has activity column: "Hemolytic" vs "Not Hemolytic" vs empty
df = pd.read_csv(DATABASE_DIR / "hemolytik2_complete.csv")
df['seq'] = df['seq'].str.upper().str.strip()

# Activity distribution
act_counts = df['activity'].astype(str).str.strip().value_counts()
print("Activity distribution:")
for k, v in act_counts.items():
    print(f"  '{k}': {v}")

# Hemolytic = 1, others = 0
df['hemo_label'] = (df['activity'].astype(str).str.upper().str.strip() == 'HEMOLYTIC').astype(int)

# Unique sequences
uniq = df.groupby('seq')['hemo_label'].max().reset_index()
print(f"\nUnique sequences: {len(uniq)}, hemolytic={uniq['hemo_label'].sum()}")

# Check how many we need to extract ESM-2 for
PROCESSED = PROCESSED_DIR
FEATURES = PROCESSED / "features"

# Load existing ESM-2 embeddings
existing_seqs = set()
for split_name in ["amp_train_real", "amp_val_real", "amp_test_real"]:
    df_split = pd.read_csv(PROCESSED / f"{split_name}.csv")
    for s in df_split["sequence"]:
        existing_seqs.add(s.upper().strip())

# Also from new_neg data
new_neg = pd.read_csv(PROCESSED / "amp_data_new_neg.csv")
for s in new_neg["sequence"]:
    existing_seqs.add(s.upper().strip())

overlap = set(uniq['seq']) & existing_seqs
new_to_extract = set(uniq['seq']) - existing_seqs
print(f"\nHemolytik2 seqs: {len(uniq)}")
print(f"Already have embeddings: {len(overlap)}")
print(f"Need to extract: {len(new_to_extract)}")

# Save Hemolytik2 training data
uniq.to_csv(PROCESSED / "hemolytik2_labels.csv", index=False)
print("Saved: hemolytik2_labels.csv")
