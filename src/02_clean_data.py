"""
02_clean_data.py — 数据清洗与特征准备

读取 amp_data_raw.csv，执行：
- 序列标准化（去除非标准氨基酸）
- 去重
- CD-HIT 去冗余（可选）
- 统一标签格式
- 保存为训练/验证/测试集
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("D:/Research_AI_Bio/03_Datasets/Processed/amp_data_raw.csv")
OUTPUT_DIR = Path("D:/Research_AI_Bio/03_Datasets/Processed/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def standardize_sequence(seq):
    """标准化序列：大写、去除非标准残基、替换X为随机A"""
    if not isinstance(seq, str):
        return ""
    return "".join(c if c in VALID_AA else "A" for c in seq.strip().upper())


def filter_by_length(df, min_len=5, max_len=200):
    """过滤长度异常序列"""
    df["seq_len"] = df["sequence"].str.len()
    before = len(df)
    df = df[(df["seq_len"] >= min_len) & (df["seq_len"] <= max_len)]
    print(f"  长度过滤: {before} -> {len(df)} (min={min_len}, max={max_len})")
    return df


def deduplicate(df):
    """按序列去重"""
    before = len(df)
    # 同一序列可能同时有AMP和非AMP标注，保留AMP优先
    df = df.sort_values("label_amp", ascending=False).drop_duplicates(
        subset=["sequence"], keep="first")
    print(f"  去重: {before} -> {len(df)}")
    return df


def split_dataset(df, test_size=0.1, val_size=0.1, random_state=42):
    """划分训练/验证/测试集（按 label_amp 分层）"""
    X = df.reset_index(drop=True)
    y = df["label_amp"].values

    X_temp, X_test = train_test_split(
        X, test_size=test_size, random_state=random_state,
        stratify=y)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val = train_test_split(
        X_temp, test_size=val_ratio, random_state=random_state,
        stratify=X_temp["label_amp"].values)

    print(f"\n  数据集划分:")
    print(f"    训练集:   {len(X_train)} ({X_train['label_amp'].mean():.1%} AMP)")
    print(f"    验证集:   {len(X_val)} ({X_val['label_amp'].mean():.1%} AMP)")
    print(f"    测试集:   {len(X_test)} ({X_test['label_amp'].mean():.1%} AMP)")

    return X_train, X_val, X_test


def main():
    print("=" * 60)
    print("  数据清洗流程")
    print("=" * 60)

    # 1. 读取
    print("\n[1/5] 读取原始数据...")
    df = pd.read_csv(INPUT_PATH)
    print(f"    读取 {len(df)} 条记录")

    # 2. 序列标准化
    print("\n[2/5] 标准化序列...")
    df["sequence"] = df["sequence"].apply(standardize_sequence)
    df = df[df["sequence"].str.len() > 0]
    print(f"    标准化后: {len(df)} 条")

    # 3. 长度过滤
    print("\n[3/5] 长度过滤...")
    df = filter_by_length(df)

    # 4. 去重
    print("\n[4/5] 去重...")
    df = deduplicate(df)

    # 5. 划分并保存
    print("\n[5/5] 划分数据集...")
    train_df, val_df, test_df = split_dataset(df)

    # 保存
    train_df.to_csv(OUTPUT_DIR / "amp_train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(OUTPUT_DIR / "amp_val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT_DIR / "amp_test.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"  清洗完成！文件已保存到: {OUTPUT_DIR}")
    print(f"    amp_train.csv: {len(train_df)} 条")
    print(f"    amp_val.csv:   {len(val_df)} 条")
    print(f"    amp_test.csv:  {len(test_df)} 条")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
