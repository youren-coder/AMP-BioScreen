import pandas as pd, numpy as np
from pathlib import Path
import sys, os
_utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_utils_dir, '..'))
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR

# Map Hemolytik2 labels: hemolytic = 1, non-hemolytic = 0
df = pd.read_csv(DATABASE_DIR / "hemolytik2_complete.csv")
df['seq'] = df['seq'].str.upper().str.strip()

# Label: activity=="Hemolytic" -> 1, everything else -> 0
df['hemo_label'] = (df['activity'].astype(str).str.upper().str.strip() == 'HEMOLYTIC').astype(int)
print(f"Hemolytic (1): {df['hemo_label'].sum()}")
print(f"Non-hemolytic (0): {(df['hemo_label']==0).sum()}")

# Cross-reference with our AMP test set
PROCESSED = PROCESSED_DIR
test_df = pd.read_csv(PROCESSED / "amp_test_new_neg.csv")
test_seqs = set(test_df['sequence'].str.upper().str.strip())

hemo_seqs = set(df['seq'])
overlap = test_seqs & hemo_seqs
print(f"\nTest set overlap with Hemolytik2: {len(overlap)}/{len(test_seqs)}")

# Build hemolysis training dataset from Hemolytik2
hemo_data = df[['seq', 'hemo_label']].drop_duplicates(subset=['seq'])
print(f"Hemolytik2 unique seqs: {len(hemo_data)}, hemolytic={hemo_data['hemo_label'].sum()}")

# Also check: do our AMP positives overlap with Hemolytik2?
amp_pos = pd.read_csv(PROCESSED / "amp_data_new_neg.csv")
amp_pos = amp_pos[amp_pos['label_amp']==1]
amp_seqs = set(amp_pos['sequence'].str.upper().str.strip())
amp_hemo_overlap = amp_seqs & hemo_seqs
print(f"AMP sequences with hemolysis labels: {len(amp_hemo_overlap)}/{len(amp_seqs)}")
