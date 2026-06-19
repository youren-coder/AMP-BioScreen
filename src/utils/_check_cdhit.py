# Check if scikit-bio is installed yet
try:
    from skbio import DistanceMatrix
    from skbio.sequence import Protein
    print("skbio available")
except ImportError:
    print("skbio NOT available")

# Alternative: use scipy for CD-HIT-like clustering
# CD-HIT algorithm: greedy incremental clustering
# For CD-HIT at 40% identity = 60% similarity threshold
# For CD-HIT at 70% identity = 30% similarity threshold
import pandas as pd
import numpy as np

# Load all train+test sequences
train = pd.read_csv(r"D:\Research_AI_Bio\03_Datasets\Processed\amp_train_real.csv")
test = pd.read_csv(r"D:\Research_AI_Bio\03_Datasets\Processed\amp_test_real.csv")
val = pd.read_csv(r"D:\Research_AI_Bio\03_Datasets\Processed\amp_val_real.csv")

all_data = pd.concat([train, test, val], ignore_index=True)
print("Total sequences:", len(all_data))
print("Pos:", (all_data["label_amp"]==1).sum(), "Neg:", (all_data["label_amp"]==0).sum())
