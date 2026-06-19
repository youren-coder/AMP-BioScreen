from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR
﻿import pandas as pd, numpy as np

# Load Hemolytik2 data and check activity column
df = pd.read_csv(DATABASE_DIR / "hemolytik2_complete.csv")

# Check activity column values
act = df['activity'].dropna()
print(f"Activity column: {len(act)} values")
print(f"Unique count: {act.nunique()}")

# Check for hemolytic / non-hemolytic indicators
for keyword in ['hemolytic', 'HEMOLYTIC', 'non-hemolytic', 'NON-HEMOLYTIC', 'Not hemolytic', 'active', 'inactive']:
    count = act.astype(str).str.contains(keyword, case=False, na=False).sum()
    if count > 0:
        print(f"  '{keyword}': {count}")

# Look for numeric activity values
numeric_act = pd.to_numeric(act, errors='coerce')
n_numeric = numeric_act.notna().sum()
print(f"\nNumeric values: {n_numeric}")
if n_numeric > 0:
    print(f"  Range: [{numeric_act.min()}, {numeric_act.max()}]")
    print(f"  Median: {numeric_act.median()}")
    print(f"  Sample: {numeric_act.dropna().head(10).tolist()}")

# Check the seq column (peptide sequence)
print(f"\nSequence column: {df['seq'].notna().sum()} non-null")
print(f"Sample sequences:")
for s in df['seq'].dropna().head(3):
    print(f"  {str(s)[:60]}")
