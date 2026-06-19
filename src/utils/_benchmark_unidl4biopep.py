import sys
sys.path.insert(0, r"D:\Research_AI_Bio\01_Projects\AMP-BioScreen\.venv\Lib\site-packages")

import numpy as np, pandas as pd, os, json
import tensorflow as tf
import joblib, torch, esm
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score

MODEL_DIR = "D:/Research_AI_Bio/08_Tools/UniDL4BioPep_AMP"
OUT_DIR = "D:/Research_AI_Bio/03_Datasets/Processed/features"

# Load UniDL4BioPep model and scaler
print("Loading UniDL4BioPep model...")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "AMP_best_model.keras"), compile=False)
scaler = joblib.load(os.path.join(MODEL_DIR, "AMP_minmax_scaler.pkl"))

# Load ESM-2 t6_8M
print("Loading ESM-2 t6_8M...")
t6_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
device = torch.device("cpu")
t6_model = t6_model.to(device).eval()
bc = alphabet.get_batch_converter()
batch_size = 64

def extract_and_predict(seqs, y_true):
    embeddings = []
    for i in range(0, len(seqs), batch_size):
        batch_seqs = seqs[i:i + batch_size]
        data = [(str(j), s) for j, s in enumerate(batch_seqs)]
        _, _, tokens = bc(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = t6_model(tokens, repr_layers=[6])
            emb = results["representations"][6]
            for j in range(len(batch_seqs)):
                mask = tokens[j] != alphabet.padding_idx
                embeddings.append(emb[j, mask].mean(dim=0).cpu().numpy())
    
    X = np.array(embeddings, dtype=np.float32)
    X_scaled = scaler.transform(X).reshape(-1, 320, 1)
    y_prob = model.predict(X_scaled, batch_size=32, verbose=0)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "acc": float(accuracy_score(y_true, y_pred)),
    }

results = {}

# Test 1: Short secreted negatives
print("\n--- Test 1: Short secreted negatives ---")
df = pd.read_csv("D:/Research_AI_Bio/03_Datasets/Processed/amp_test_new_neg.csv")
r = extract_and_predict(df["sequence"].str.upper().str.strip().tolist(), df["label_amp"].values)
results["UniDL4BioPep_short_secreted_neg"] = r
print(f"AUC={r['auc']:.4f} F1={r['f1']:.4f} MCC={r['mcc']:.4f} ACC={r['acc']:.4f}")

# Test 2: Old full-length negatives  
print("\n--- Test 2: Old full-length negatives ---")
df = pd.read_csv("D:/Research_AI_Bio/03_Datasets/Processed/amp_test_real.csv")
r = extract_and_predict(df["sequence"].str.upper().str.strip().tolist(), df["label_amp"].values)
results["UniDL4BioPep_old_full_length_neg"] = r
print(f"AUC={r['auc']:.4f} F1={r['f1']:.4f} MCC={r['mcc']:.4f} ACC={r['acc']:.4f}")

# Test 3: Strict length-matched
print("\n--- Test 3: Strict length-matched ---")
df = pd.read_csv("D:/Research_AI_Bio/03_Datasets/Processed/amp_data_length_matched.csv")
r = extract_and_predict(df["sequence"].str.upper().str.strip().tolist(), df["label_amp"].values)
results["UniDL4BioPep_length_matched"] = r
print(f"AUC={r['auc']:.4f} F1={r['f1']:.4f} MCC={r['mcc']:.4f} ACC={r['acc']:.4f}")

# Summary
print("\n" + "=" * 70)
print("UniDL4BioPep DIRECT BENCHMARK")
print("=" * 70)
print(f"{'Test Set':<35} {'AUC':>8} {'F1':>8} {'MCC':>8} {'ACC':>8}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<35} {r['auc']:8.4f} {r['f1']:8.4f} {r['mcc']:8.4f} {r['acc']:8.4f}")

with open(os.path.join(OUT_DIR, "unidl4biopep_benchmark.json"), "w") as f:
    json.dump(results, f, indent=2)
print("DONE.")
