"""
05_merge_features.py — 合并理化特征与 ESM-2 嵌入

读取 15 维理化特征 + 480 维 ESM-2 嵌入，合并为完整特征矩阵。
输出保存为 .npy 格式，供 XGBoost 训练使用。
"""

import pandas as pd
import numpy as np
from pathlib import Path

FEATURE_DIR = Path("D:/Research_AI_Bio/03_Datasets/Processed/features")

PHYSIO_FILES = {
    "train": FEATURE_DIR / "features_physicochem_train.csv",
    "val":   FEATURE_DIR / "features_physicochem_val.csv",
    "test":  FEATURE_DIR / "features_physicochem_test.csv",
}

ESM_FILES = {
    "train": FEATURE_DIR / "amp_train_esm2.npy",
    "val":   FEATURE_DIR / "amp_val_esm2.npy",
    "test":  FEATURE_DIR / "amp_test_esm2.npy",
}


def merge_dataset(name, physio_path, esm_path):
    """合并单个数据集的理化特征与 ESM-2 嵌入"""
    print(f"\n[{name}]")

    # 加载理化特征
    df_physio = pd.read_csv(physio_path)
    print(f"  理化特征: {df_physio.shape}")

    # 分离标签
    label_amp = df_physio["label_amp"].values
    label_hemolysis = df_physio["label_hemolysis"].values
    feature_cols = [c for c in df_physio.columns
                     if c not in ["label_amp", "label_hemolysis"]]
    X_physio = df_physio[feature_cols].values.astype(np.float32)

    # 加载 ESM-2 嵌入
    X_esm = np.load(esm_path).astype(np.float32)
    print(f"  ESM-2嵌入: {X_esm.shape}")

    # 验证行数一致
    assert X_physio.shape[0] == X_esm.shape[0], \
        f"行数不匹配: {X_physio.shape[0]} vs {X_esm.shape[0]}"

    # 合并特征
    X_merged = np.concatenate([X_physio, X_esm], axis=1)
    print(f"  合并特征: {X_merged.shape} ({X_physio.shape[1]}理化 + {X_esm.shape[1]}ESM)")

    # 保存
    out_prefix = FEATURE_DIR / f"amp_{name}"
    np.save(f"{out_prefix}_X.npy", X_merged)
    np.save(f"{out_prefix}_y_amp.npy", label_amp)
    np.save(f"{out_prefix}_y_hemolysis.npy", label_hemolysis)
    print(f"  已保存: {out_prefix}_X.npy 等")

    return X_merged, label_amp, label_hemolysis


def main():
    print("=" * 60)
    print("  特征合并: 理化特征 + ESM-2 嵌入")
    print("=" * 60)

    for name in ["train", "val", "test"]:
        merge_dataset(name, PHYSIO_FILES[name], ESM_FILES[name])

    # 验证总数
    print(f"\n{'='*60}")
    for name in ["train", "val", "test"]:
        X = np.load(FEATURE_DIR / f"amp_{name}_X.npy")
        y = np.load(FEATURE_DIR / f"amp_{name}_y_amp.npy")
        print(f"  {name}: X={X.shape}, AMP+={y.sum()}, AMP-={(y==0).sum()}")
    print(f"{'='*60}")
    print(f"\n特征合并完成！")
    print(f"运行下一步: python src/06_train_xgboost.py")


if __name__ == "__main__":
    main()
