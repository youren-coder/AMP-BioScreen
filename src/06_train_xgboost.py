"""
06_train_xgboost.py — XGBoost 模型训练 + SHAP 可解释性分析

输入: features/amp_{train,val,test}_X.npy + _y_amp.npy
输出: 模型文件、评估指标、SHAP 图（保存到 features/ 和 figures/）
"""

import warnings, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
from pathlib import Path
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR

warnings.filterwarnings("ignore")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """加载合并后的特征和标签"""
    data = {}
    for name in ["train", "val", "test"]:
        X = np.load(FEATURE_DIR / f"amp_{name}_X.npy")
        y = np.load(FEATURE_DIR / f"amp_{name}_y_amp.npy")
        data[name] = (X, y)
        print(f"  {name}: X={X.shape}, AMP+={int(y.sum())}, AMP-={int((y==0).sum())}")
    return data


def train_model(X_train, y_train, X_val, y_val):
    """训练 XGBoost 分类器，含超参数搜索"""
    import xgboost as xgb

    # 正负样本比例
    neg, pos = (y_train == 0).sum(), y_train.sum()
    scale_pos_weight = neg / pos
    print(f"  正负比例: {pos}/{neg}, scale_pos_weight={scale_pos_weight:.2f}")

    # 超参数搜索
    print("\n  超参数搜索中...")
    best_score = 0
    best_params = {}
    best_model = None

    for lr in [0.01, 0.05, 0.1]:
        for depth in [4, 6, 8]:
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=depth,
                learning_rate=lr,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="auc",
                early_stopping_rounds=30,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            score = model.best_score
            if score > best_score:
                best_score = score
                best_params = {"lr": lr, "depth": depth, "estimators": model.best_iteration + 1}
                best_model = model

    print(f"  最佳参数: {best_params}, 验证集AUC: {best_score:.4f}")
    return best_model, best_params


def evaluate(model, X, y, label="数据集"):
    """全面评估模型性能"""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, matthews_corrcoef, accuracy_score,
        precision_score, recall_score, confusion_matrix,
    )

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    metrics = {
        "AUC-ROC": float(roc_auc_score(y, y_proba)),
        "AUC-PR":  float(average_precision_score(y, y_proba)),
        "Accuracy": float(accuracy_score(y, y_pred)),
        "Precision": float(precision_score(y, y_pred, zero_division=0)),
        "Recall":    float(recall_score(y, y_pred)),
        "F1":        float(f1_score(y, y_pred)),
        "MCC":       float(matthews_corrcoef(y, y_pred)),
        "Sensitivity (TPR)": float(tp / (tp + fn)),
        "Specificity (TNR)": float(tn / (tn + fp)),
    }

    print(f"  [{label}]", end="")
    for k, v in metrics.items():
        print(f" {k}={v:.4f}", end="")
    print()

    return metrics, y_proba


def shap_analysis(model, X_train, X_test, feature_names, output_prefix):
    """SHAP 可解释性分析"""
    import shap

    print("  计算 SHAP 值...")

    # 使用训练集的子集作为背景 (TreeExplainer 更快)
    background = X_train[:100]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test[:200])

    # SHAP 摘要图（全局特征重要性）
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:200],
                     feature_names=feature_names,
                     show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_shap_summary.png", dpi=600)
    plt.close()

    # 条形图（平均绝对 SHAP 值）
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test[:200],
                     feature_names=feature_names,
                     plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_shap_bar.png", dpi=600)
    plt.close()

    print(f"  SHAP 图已保存")


def plot_roc_curves(train_proba, train_y, val_proba, val_y, test_proba, test_y, output_path):
    """绘制三个数据集的 ROC 曲线"""
    from sklearn.metrics import roc_curve, roc_auc_score

    plt.figure(figsize=(7, 6))
    for label, proba, y_true in [
        ("Train", train_proba, train_y),
        ("Validation", val_proba, val_y),
        ("Test", test_proba, test_y),
    ]:
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - AMP Classification")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"  ROC曲线: {output_path}")


def main():
    print("=" * 60)
    print("  XGBoost 模型训练 — AMP 分类")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    data = load_data()
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    # 特征名称
    physio_names = [
        "length", "mw", "charge", "pI", "hydrophob", "hmoment",
        "instability", "aliphatic", "boman", "entropy", "gravy",
        "pos_charge", "neg_charge", "charge_ratio", "pos_ratio", "neg_ratio",
    ]
    esm_names = [f"ESM_{i}" for i in range(480)]
    feature_names = physio_names + esm_names

    # 2. 训练模型
    print("\n[2/5] 训练 XGBoost...")
    model, best_params = train_model(X_train, y_train, X_val, y_val)

    # 3. 评估
    print("\n[3/5] 模型评估...")
    train_metrics, train_proba = evaluate(model, X_train, y_train, "训练集")
    val_metrics,   val_proba   = evaluate(model, X_val, y_val, "验证集")
    test_metrics,  test_proba  = evaluate(model, X_test, y_test, "测试集")

    # 4. SHAP 分析
    print("\n[4/5] SHAP 可解释性分析...")
    shap_analysis(model, X_train, X_test, feature_names,
                  str(FIGURE_DIR / "amp"))

    # 5. ROC 曲线
    print("\n[5/5] 可视化...")
    plot_roc_curves(
        train_proba, y_train,
        val_proba, y_val,
        test_proba, y_test,
        FIGURE_DIR / "amp_roc_curves.png"
    )

    # 保存结果
    results = {
        "best_params": best_params,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(FEATURE_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存模型
    model.save_model(str(FEATURE_DIR / "xgb_model.json"))
    print(f"  模型已保存: {FEATURE_DIR / 'xgb_model.json'}")

    # 输出摘要
    print(f"\n{'='*60}")
    print(f"  训练完成!")
    print(f"  测试集 AUC: {test_metrics['AUC-ROC']:.4f}")
    print(f"  测试集 F1:  {test_metrics['F1']:.4f}")
    print(f"  图片: {FIGURE_DIR}")
    print(f"{'='*60}")
    print(f"\n参考文献: Lundberg & Lee, SHAP (NeurIPS 2017)")
    print(f"         Chen & Guestrin, XGBoost (KDD 2016)")


if __name__ == "__main__":
    main()
