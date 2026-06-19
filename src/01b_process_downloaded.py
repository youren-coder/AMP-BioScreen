"""
01b_process_downloaded.py — 处理已下载的 TSV 文件

当自动下载失败时，用 PowerShell 手动下载后，运行此脚本处理。
"""

import pandas as pd
import numpy as np
import random, csv, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR

RAW_DIR = DATABASE_DIR
OUTPUT_DIR = PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_uniprot_tsv(filepath, label_name):
    """解析 UniProt TSV"""
    records = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            seq = row.get("Sequence", "").strip().upper()
            if not seq or len(seq) < 5:
                continue
            records.append({
                "sequence": seq,
                "source": label_name,
                "label_amp": 1,
                "label_hemolysis": -1,
            })
    return pd.DataFrame(records)


def generate_negative_samples(positive_seqs, n=5000, min_len=5, max_len=60):
    """生成随机负样本"""
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    random.seed(42)
    lens = [len(s) for s in positive_seqs if isinstance(s, str)]
    min_l = max(min(lens or [15]), min_len)
    max_l = min(max(lens or [30]), max_len)
    pos_set = set(s.upper().strip() for s in positive_seqs if isinstance(s, str))
    neg = set()
    while len(neg) < n:
        l = random.randint(min_l, max_l)
        seq = "".join(random.choices(aa_list, k=l))
        if seq not in pos_set:
            neg.add(seq)
    df = pd.DataFrame(list(neg), columns=["sequence"])
    df["source"] = "Random_NonAMP"
    df["label_amp"] = 0
    df["label_hemolysis"] = -1
    return df


def main():
    print("=" * 60)
    print("  处理已下载的 TSV 文件")
    print("=" * 60)

    # 查找已下载的 TSV 文件
    tsv_files = sorted(RAW_DIR.glob("uniprot_*.tsv"))
    if not tsv_files:
        # 也检查是否有 manual filename
        tsv_files = sorted(RAW_DIR.glob("*.tsv"))
    if not tsv_files:
        print("未找到任何 TSV 文件。")
        print(f"请先放置文件到: {RAW_DIR}")
        print("文件名示例: uniprot_amps.tsv, uniprot_defensin.tsv")
        return

    print(f"找到 {len(tsv_files)} 个 TSV 文件:")
    for f in tsv_files:
        print(f"  {f.name} ({f.stat().st_size} bytes)")

    # 解析所有文件
    all_dfs = []
    for f in tsv_files:
        label = f.stem.replace("uniprot_", "")
        df = parse_uniprot_tsv(f, label)
        print(f"  {f.name}: {len(df)} 条序列")
        all_dfs.append(df)

    df_pos = pd.concat(all_dfs, ignore_index=True)
    before = len(df_pos)
    df_pos = df_pos.drop_duplicates(subset=["sequence"])
    print(f"\n正样本合并去重: {before} -> {len(df_pos)}")

    # 过滤无效序列
    def is_valid(s):
        return isinstance(s, str) and all(c in VALID_AA for c in s.upper().strip())

    df_pos = df_pos[df_pos["sequence"].apply(is_valid)]
    df_pos = df_pos[df_pos["sequence"].str.len().between(5, 200)]
    print(f"过滤后: {len(df_pos)} 条有效序列")

    # 生成负样本
    print(f"\n生成负样本 (x3 正样本数量)...")
    n_neg = min(len(df_pos) * 3, 15000)
    df_neg = generate_negative_samples(df_pos["sequence"].tolist(), n=n_neg)
    print(f"负样本: {len(df_neg)} 条")

    # 合并
    df_all = pd.concat([df_pos, df_neg], ignore_index=True)

    # 划分数据集
    X_train, X_temp = train_test_split(
        df_all, test_size=0.2, random_state=42, stratify=df_all["label_amp"]
    )
    X_val, X_test = train_test_split(
        X_temp, test_size=0.5, random_state=42, stratify=X_temp["label_amp"]
    )

    # 保存
    X_train.to_csv(OUTPUT_DIR / "amp_train.csv", index=False, encoding="utf-8-sig")
    X_val.to_csv(OUTPUT_DIR / "amp_val.csv", index=False, encoding="utf-8-sig")
    X_test.to_csv(OUTPUT_DIR / "amp_test.csv", index=False, encoding="utf-8-sig")
    df_all.to_csv(OUTPUT_DIR / "amp_data_raw.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"  数据集已保存到: {OUTPUT_DIR}")
    print(f"  总记录:    {len(df_all)}")
    print(f"  AMP+样本:  {df_all['label_amp'].sum()} ({df_all['label_amp'].mean()*100:.0f}%)")
    print(f"  非AMP样本: {(df_all['label_amp'] == 0).sum()}")
    print(f"")
    print(f"  训练集: {len(X_train)} 条")
    print(f"  验证集: {len(X_val)} 条")
    print(f"  测试集: {len(X_test)} 条")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
