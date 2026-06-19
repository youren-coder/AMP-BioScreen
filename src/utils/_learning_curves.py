"""
学习曲线实验：35M / 150M / 650M 三个 ESM-2 模型规模
训练时记录每轮 train/val AUC，用于判断过拟合 vs 欠拟合
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import xgboost as xgb

FEATURE_DIR = r"D:\Research_AI_Bio\03_Datasets\Processed\features"
FIGURE_DIR = r"D:\Research_AI_Bio\06_Figures"

# Load all data
train_y = np.load(f"{FEATURE_DIR}/amp_train_y_amp.npy")
val_y = np.load(f"{FEATURE_DIR}/amp_val_y_amp.npy")
test_y = np.load(f"{FEATURE_DIR}/amp_test_y_amp.npy")

# Model configs
models = {
    "35M": {
        "X_train": np.load(f"{FEATURE_DIR}/amp_train_X.npy"),
        "X_val": np.load(f"{FEATURE_DIR}/amp_val_X.npy"),
        "X_test": np.load(f"{FEATURE_DIR}/amp_test_X.npy"),
        "dim": 496,
    },
    "150M": {
        "X_train": np.load(f"{FEATURE_DIR}/amp_train_X_150m.npy"),
        "X_val": np.load(f"{FEATURE_DIR}/amp_val_X_150m.npy"),
        "X_test": np.load(f"{FEATURE_DIR}/amp_test_X_150m.npy"),
        "dim": 656,
    },
    "650M": {
        "X_train": np.load(f"{FEATURE_DIR}/amp_train_X_650m.npy"),
        "X_val": np.load(f"{FEATURE_DIR}/amp_val_X_650m.npy"),
        "X_test": np.load(f"{FEATURE_DIR}/amp_test_X_650m.npy"),
        "dim": 1296,
    },
}

# Fixed XGBoost params (same as paper's best for 150M)
xgb_params = {
    "n_estimators": 500,  # More estimators to see full convergence
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
    "random_state": 42,
}

all_results = {}

for name, cfg in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name} ({cfg['dim']}-dim)")
    print(f"{'='*60}")
    
    scale_weight = (train_y == 0).sum() / max((train_y == 1).sum(), 1)
    print(f"  Scale pos weight: {scale_weight:.2f}")
    print(f"  Train samples: {len(train_y)}, Val: {len(val_y)}, Test: {len(test_y)}")
    print(f"  Pos/Neg in train: {(train_y==1).sum()}/{(train_y==0).sum()}")
    
    model = xgb.XGBClassifier(
        **xgb_params,
        scale_pos_weight=scale_weight,
    )
    
    # Use eval_set with both train and val to get learning curves
    eval_sets = [
        (cfg["X_train"], train_y),
        (cfg["X_val"], val_y),
    ]
    eval_set_names = ["train", "val"]
    
    model.fit(
        cfg["X_train"], train_y,
        eval_set=eval_sets,
        verbose=False
    )
    
    # Extract learning curves
    results = model.evals_result()
    
    # Get best iteration
    best_iteration = model.best_iteration if model.best_iteration else len(results["validation_0"]["auc"]) - 1
    print(f"  Best iteration: {best_iteration}")
    
    # Test set evaluation
    y_prob = model.predict_proba(cfg["X_test"])[:, 1]
    y_pred = model.predict(cfg["X_test"])
    
    test_auc = metrics.roc_auc_score(test_y, y_prob)
    test_f1 = metrics.f1_score(test_y, y_pred)
    test_mcc = metrics.matthews_corrcoef(test_y, y_pred)
    
    print(f"  Test AUC: {test_auc:.4f}, F1: {test_f1:.4f}, MCC: {test_mcc:.4f}")
    
    # Store results
    all_results[name] = {
        "train_auc_curve": results["validation_0"]["auc"],
        "val_auc_curve": results["validation_1"]["auc"],
        "best_iteration": int(best_iteration),
        "n_estimators_actual": len(results["validation_0"]["auc"]),
        "test_auc": float(test_auc),
        "test_f1": float(test_f1),
        "test_mcc": float(test_mcc),
        "dim": cfg["dim"],
        "train_auc_final": float(results["validation_0"]["auc"][-1]),
        "val_auc_final": float(results["validation_1"]["auc"][-1]),
        "val_auc_best": float(max(results["validation_1"]["auc"])),
    }
    
    # Also record at key checkpoints for paper table
    for n in [50, 100, 200, 300, 400, 500]:
        idx = min(n - 1, len(results["validation_1"]["auc"]) - 1)
        all_results[name][f"val_auc_at_{n}"] = float(results["validation_1"]["auc"][idx])
        all_results[name][f"train_auc_at_{n}"] = float(results["validation_0"]["auc"][idx])

# Save raw data
with open(f"{FEATURE_DIR}/learning_curves.json", "w") as f:
    json.dump(all_results, f, indent=2)

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
colors = {"35M": "#E74C3C", "150M": "#2E86AB", "650M": "#8E44AD"}

for i, (name, cfg) in enumerate(models.items()):
    ax = axes[i]
    res = all_results[name]
    epochs = range(1, len(res["train_auc_curve"]) + 1)
    
    ax.plot(epochs, res["train_auc_curve"], color=colors[name], linewidth=1.5, alpha=0.7, label="Train AUC")
    ax.plot(epochs, res["val_auc_curve"], color=colors[name], linewidth=2, label="Val AUC")
    
    # Mark best iteration
    best = res["best_iteration"]
    ax.axvline(x=best + 1, color=colors[name], linestyle="--", alpha=0.5, linewidth=1)
    ax.scatter([best + 1], [res["val_auc_best"]], color=colors[name], s=60, zorder=5)
    ax.annotate(f"Best={best+1}\nVal AUC={res['val_auc_best']:.3f}",
                (best + 1, res["val_auc_best"]),
                textcoords="offset points", xytext=(8, -20),
                fontsize=8, color=colors[name])
    
    ax.set_title(f"{name} ({cfg['dim']} dims)\nTest AUC={res['test_auc']:.4f}", fontsize=11)
    ax.set_xlabel("Boosting Rounds", fontsize=9)
    ax.set_ylabel("AUC", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0.75, 1.02)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
    ax.grid(alpha=0.3)

plt.suptitle("XGBoost Learning Curves: ESM-2 Model Scale Comparison", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURE_DIR}/learning_curves.png", dpi=600, bbox_inches='tight')
print(f"\nLearning curves saved to {FIGURE_DIR}/learning_curves.png")

# --- Summary table ---
print(f"\n{'='*70}")
print(f"{'Checkpoint':<15} {'35M (480d)':<15} {'150M (640d)':<15} {'650M (1280d)':<15}")
print(f"{'-'*70}")
for n in [50, 100, 200, 300, 500]:
    vals = []
    for name in ["35M", "150M", "650M"]:
        key = f"val_auc_at_{n}"
        if key in all_results[name]:
            vals.append(f"{all_results[name][key]:.4f}")
        else:
            vals.append("-")
    print(f"Round {n:<10} {vals[0]:<15} {vals[1]:<15} {vals[2]:<15}")

print(f"\nFinal:")
for name in ["35M", "150M", "650M"]:
    r = all_results[name]
    gap = r["train_auc_final"] - r["val_auc_final"]
    print(f"  {name}: Train AUC={r['train_auc_final']:.4f}, Val AUC={r['val_auc_final']:.4f}, "
          f"Gap={gap:.4f}, Best Val={r['val_auc_best']:.4f}, Test AUC={r['test_auc']:.4f}")

# Classify: overfitting if train→1.0 and val drops; underfitting if train also low
print(f"\n{'='*70}")
print("DIAGNOSIS:")
for name in ["35M", "150M", "650M"]:
    r = all_results[name]
    train_final = r["train_auc_final"]
    val_best = r["val_auc_best"]
    gap = train_final - val_best
    
    if train_final > 0.995:
        status = "MEMORIZATION (train → 1.0, possible overfit)"
    elif gap > 0.08:
        status = "OVERFITTING (large train-val gap)"
    elif train_final < 0.92:
        status = "UNDERFITTING (train AUC too low)"
    else:
        status = "WELL-FITTED"
    
    print(f"  {name}: {status} (Train={train_final:.4f}, Best Val={val_best:.4f}, Gap={gap:.4f})")

print("\nDone.")
