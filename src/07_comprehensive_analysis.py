"""
07_comprehensive_analysis.py — 综合实验分析

包括：
1. 特征消融：理化特征 vs ESM-2 嵌入 vs 融合特征
2. 5折交叉验证
3. 性能对比表 + 可视化
"""

import warnings, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

FEATURE_DIR = Path("D:/Research_AI_Bio/03_Datasets/Processed/features")
FIGURE_DIR = Path("D:/Research_AI_Bio/06_Figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """加载三种特征的数据"""
    data = {}
    for name in ["train", "val", "test"]:
        # 理化特征 (从 CSV 加载，跳过标签列)
        df = pd.read_csv(FEATURE_DIR / f"features_physicochem_{name}.csv")
        feat_cols = [c for c in df.columns if c not in ["label_amp", "label_hemolysis"]]
        X_physio = df[feat_cols].values.astype(np.float32)

        # ESM-2 特征
        X_esm = np.load(FEATURE_DIR / f"amp_{name}_esm2.npy").astype(np.float32)

        # 融合特征
        X_combined = np.load(FEATURE_DIR / f"amp_{name}_X.npy").astype(np.float32)

        # 标签
        y = np.load(FEATURE_DIR / f"amp_{name}_y_amp.npy")

        data[name] = {
            "physio": X_physio,
            "esm": X_esm,
            "combined": X_combined,
            "y": y,
        }
    return data


def train_model(X_train, y_train, X_val, y_val, name="模型"):
    """训练 XGBoost + 超参搜索"""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    neg, pos = (y_train == 0).sum(), y_train.sum()
    scale = neg / pos

    best_score = 0
    best_model = None
    best_params = {}

    for lr in [0.05, 0.1]:
        for depth in [4, 6, 8]:
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=depth,
                learning_rate=lr,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale,
                eval_metric="auc",
                early_stopping_rounds=30,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
            try:
                score = model.best_score
            except:
                score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            if score > best_score:
                best_score = score
                best_params = {"lr": lr, "depth": depth, "n_est": model.best_iteration + 1}
                best_model = model

    print(f"  [{name}] 最佳: {best_params}, 验证AUC={best_score:.4f}")
    return best_model, best_params


def evaluate(model, X, y, name="数据集"):
    """评估模型"""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 f1_score, matthews_corrcoef, accuracy_score,
                                 precision_score, recall_score, confusion_matrix)
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    metrics = {
        "AUC_ROC": round(roc_auc_score(y, y_proba), 4),
        "AUC_PR":  round(average_precision_score(y, y_proba), 4),
        "Accuracy": round(accuracy_score(y, y_pred), 4),
        "F1":      round(f1_score(y, y_pred), 4),
        "MCC":     round(matthews_corrcoef(y, y_pred), 4),
        "Sensitivity": round(tp / (tp + fn), 4),
        "Specificity": round(tn / (tn + fp), 4),
    }
    return metrics


def cross_validate(X, y, n_folds=5):
    """k 折交叉验证"""
    from sklearn.model_selection import StratifiedKFold
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = {"AUC": [], "F1": [], "MCC": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        neg, pos = (y_tr == 0).sum(), y_tr.sum()
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=neg / pos,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        scores["AUC"].append(round(roc_auc_score(y_val, y_prob), 4))
        scores["F1"].append(round(f1_score(y_val, y_pred), 4))
        scores["MCC"].append(round(matthews_corrcoef(y_val, y_pred), 4))
        print(f"    折{fold+1}: AUC={scores['AUC'][-1]:.4f}, F1={scores['F1'][-1]:.4f}")

    return {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in scores.items()}


def plot_comparison(results, output_path):
    """画对比柱状图"""
    models = list(results.keys())
    metrics = ["AUC_ROC", "AUC_PR", "F1", "MCC"]
    x = np.arange(len(metrics))
    width = 0.25

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        values = [results[model][m] for m in metrics]
        bars = plt.bar(x + i * width, values, width, label=model)

    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title("Feature Ablation Comparison")
    plt.xticks(x + width, metrics)
    plt.ylim(0.8, 1.0)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"  对比图: {output_path}")


def main():
    print("=" * 65)
    print("  综合实验分析")
    print("=" * 65)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    data = load_data()
    for name in ["train", "val", "test"]:
        d = data[name]
        print(f"  {name}: physio={d['physio'].shape}, esm={d['esm'].shape}, combined={d['combined'].shape}")

    # 2. 特征消融
    print("\n[2/4] 特征消融实验 (训练三个模型)...")
    models = {}

    models["physio"], _ = train_model(
        data["train"]["physio"], data["train"]["y"],
        data["val"]["physio"], data["val"]["y"], "仅理化特征")

    models["esm"], _ = train_model(
        data["train"]["esm"], data["train"]["y"],
        data["val"]["esm"], data["val"]["y"], "仅ESM-2嵌入")

    models["combined"], _ = train_model(
        data["train"]["combined"], data["train"]["y"],
        data["val"]["combined"], data["val"]["y"], "融合特征")

    # 3. 测试集对比
    print("\n[3/4] 测试集对比...")
    results = {}
    for name, model in models.items():
        m = evaluate(model, data["test"][name], data["test"]["y"], name)
        results[name] = m

    # 打印对比表
    print(f"\n{'='*65}")
    print(f"  测试集性能对比")
    print(f"{'='*65}")
    header = f"{'指标':<15} {'仅理化':<12} {'仅ESM-2':<12} {'融合特征':<12}"
    print(header)
    print("-" * 65)
    for metric in ["AUC_ROC", "AUC_PR", "F1", "MCC", "Accuracy", "Sensitivity", "Specificity"]:
        row = f"{metric:<15}"
        for m in ["physio", "esm", "combined"]:
            row += f" {results[m][metric]:<12}"
        print(row)
    print(f"{'='*65}")

    # 4. 交叉验证
    print("\n[4/4] 5折交叉验证 (融合特征)...")
    cv_scores = cross_validate(
        data["train"]["combined"], data["train"]["y"], n_folds=5)
    print(f"\n  交叉验证结果:")
    for k, v in cv_scores.items():
        print(f"    {k}: {v}")

    # 保存结果
    output = {
        "feature_ablation_test": results,
        "cross_validation": cv_scores,
    }
    with open(FEATURE_DIR / "comprehensive_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: comprehensive_results.json")

    # 画图
    plot_comparison(results, FIGURE_DIR / "feature_ablation_comparison.png")

    print(f"\n{'='*65}")
    print(f"  分析完成!")
    print(f"  图片: {FIGURE_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
