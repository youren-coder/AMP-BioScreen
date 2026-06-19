"""
04_esm2_local.py — 本地 CPU 运行 ESM-2 embedding

用法: python src/04_esm2_local.py
模型: esm2_t12_35M_UR50D (480维)
CPU 预计: 3696条序列约 1-2 小时
输出: features/amp_{train,val,test}_esm2.npy

支持断点续跑: 已完成的 .npy 文件会自动跳过。
"""

import os, sys, time, warnings
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR

warnings.filterwarnings("ignore")

PROCESSED = PROCESSED_DIR
FEATURE_DIR = PROCESSED / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILES = {
    "train": PROCESSED / "amp_train.csv",
    "val":   PROCESSED / "amp_val.csv",
    "test":  PROCESSED / "amp_test.csv",
}


def extract_embeddings(seqs, model, alphabet, device, batch_size=4):
    """批量提取 ESM-2 嵌入 (480维)"""
    bc = alphabet.get_batch_converter()
    all_emb = []
    n = len(seqs)
    t0 = time.time()

    for i in range(0, n, batch_size):
        batch = seqs[i:i + batch_size]
        data = [(str(j), s) for j, s in enumerate(batch)]
        _, _, tokens = bc(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[12])
            emb = results["representations"][12]
            for j in range(len(batch)):
                m = tokens[j] != alphabet.padding_idx
                v = emb[j, m].mean(dim=0).cpu().numpy()
                all_emb.append(v)

        elapsed = time.time() - t0
        done = min(i + batch_size, n)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        if done % 50 == 0 or done == n:
            print(f"  [{done}/{n}] {rate:.1f} seq/s, ETA {eta/60:.0f}min", end="")
            if eta > 0:
                print(f" ({elapsed/60:.0f}min elapsed)")
            else:
                print()

    total = time.time() - t0
    print(f"  完成! {total/60:.1f}min")
    return np.array(all_emb)


def main():
    print("=" * 60)
    print("  ESM-2 Embedding (本地 CPU)")
    print(f"  Model: esm2_t12_35M_UR50D")
    print(f"  Output: {FEATURE_DIR}")
    print("=" * 60)

    import torch
    import esm

    print("\n加载模型中（首次需下载约 600MB 权重）...")
    t0 = time.time()
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t12_35M_UR50D")
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    print(f"  模型加载完成 ({time.time()-t0:.0f}s)")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    for name, csv_path in DATA_FILES.items():
        npy_path = FEATURE_DIR / f"amp_{name}_esm2.npy"

        if npy_path.exists():
            print(f"\n[{name}] 已存在, 跳过: {npy_path.name}")
            continue

        print(f"\n[{name}] 读取数据...")
        df = pd.read_csv(csv_path)
        seqs = df["sequence"].tolist()
        print(f"  {len(seqs)} 条序列")

        emb = extract_embeddings(seqs, model, alphabet, device)
        print(f"  嵌入维度: {emb.shape}")

        np.save(npy_path, emb)
        mb = npy_path.stat().st_size / 1024 / 1024
        print(f"  已保存: {npy_path.name} ({mb:.1f} MB)")

    print(f"\n{'='*60}")
    print(f"  全部完成!")
    print(f"  下一步: python src/05_merge_features.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
