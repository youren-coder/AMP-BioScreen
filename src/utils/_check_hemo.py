import pandas as pd, numpy as np
from pathlib import Path

# Check hemolysis data
PROCESSED = Path("D:/Research_AI_Bio/03_Datasets/Processed")
FEATURES = PROCESSED / "features"

# Load hemolysis labels
y_hem_train = np.load(FEATURES / "amp_train_y_hemolysis.npy")
y_hem_val = np.load(FEATURES / "amp_val_y_hemolysis.npy")
y_hem_test = np.load(FEATURES / "amp_test_y_hemolysis.npy")

print("Hemolysis labels:")
print(f"  Train: {y_hem_train.shape}, values: {np.unique(y_hem_train, return_counts=True)}")
print(f"  Val: {y_hem_val.shape}, values: {np.unique(y_hem_val, return_counts=True)}")
print(f"  Test: {y_hem_test.shape}, values: {np.unique(y_hem_test, return_counts=True)}")

# Check Hemolytik2 data
hemo_df = pd.read_csv("D:/Research_AI_Bio/02_Databases/hemolytik2_complete.csv")
print(f"\nHemolytik2 raw: {hemo_df.shape}")
print(f"Columns: {list(hemo_df.columns)[:10]}")
print(f"First few rows:")
print(hemo_df.head(3))

# Check if there's a hemolysis column
for col in hemo_df.columns:
    if 'hemo' in col.lower() or 'lytic' in col.lower() or 'activity' in col.lower():
        vals = hemo_df[col].dropna()
        print(f"\n{col}: {len(vals)} non-null, unique={vals.nunique()}")
        print(f"  Sample values: {vals.head(10).tolist()}")
