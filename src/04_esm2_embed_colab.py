"""
04_esm2_embed_colab.py — 在 Google Colab 上运行 ESM-2 embedding

完整操作步骤：
1. 打开 https://colab.research.google.com
2. File → Upload Notebook → 选择本文件
3. Runtime → Run all
"""

# %% [markdown]
# # ESM-2 Embedding 提取 — AMP-BioScreen
# 从 AMP 序列中提取 ESM-2 嵌入特征（1280维向量），用于后续 XGBoost 模型训练。
# **预计耗时**：35M参数模型，3696条序列，约5-8分钟（T4 GPU）。

# %% 安装依赖
# @title 安装 fair-esm
!pip install fair-esm -q

# %% 导入库
# @title 导入依赖
import torch
import esm
import numpy as np
import pandas as pd
from pathlib import Path
import warnings, time, os, sys
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR

warnings.filterwarnings("ignore")

# Google Colab 文件上传
try:
    from google.colab import files
    IN_COLAB = True
except:
    IN_COLAB = False

# %% 加载模型
# @title 加载 ESM-2 模型 (12层, 35M参数)
print("加载 ESM-2 (esm2_t12_35M_UR50D)...")
model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t12_35M_UR50D")
batch_converter = alphabet.get_batch_converter()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"设备: {device}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

# %% 获取数据
# @title 上传数据文件
# 方式A: 通过 Google Drive 挂载（推荐，一次上传永久使用）
# 方式B: 直接上传 CSV 文件（每次运行都需要上传）

USE_DRIVE = False  # 改为 True 则使用 Google Drive

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_DIR = "/content/drive/MyDrive/AMP_BioScreen/"
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    print(f"数据目录: {DATA_DIR}")
    print("请将 amp_train.csv, amp_val.csv, amp_test.csv 放入 Google Drive 的 AMP_BioScreen 文件夹")
else:
    DATA_DIR = ""
    print("请上传数据文件:")
    if IN_COLAB:
        uploaded = files.upload()

# 查找 CSV 文件
csv_files = sorted([f for f in os.listdir(DATA_DIR or ".") 
                     if f.endswith(".csv") and f.startswith("amp_")])
if not csv_files:
    csv_files = sorted([f for f in os.listdir(".") 
                         if f.endswith(".csv") and f.startswith("amp_")])

print(f"\n找到 CSV 文件: {csv_files}")
if not csv_files:
    raise FileNotFoundError("未找到 amp_train.csv 等数据文件。请上传后重试。")

# %% 提取 ESM-2 嵌入
# @title 提取嵌入向量
def extract_embeddings(sequences, batch_size=16):
    """批量提取 ESM-2 嵌入，返回 (N, 1280) 数组"""
    all_emb = []
    n_batches = (len(sequences) + batch_size - 1) // batch_size
    print(f"共 {len(sequences)} 条序列, {n_batches} 个批次")
    start_time = time.time()

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        data = [(str(j), seq) for j, seq in enumerate(batch)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[12])
            emb = results["representations"][12]
            for j in range(len(batch)):
                mask = tokens[j] != alphabet.padding_idx
                vec = emb[j, mask].mean(dim=0).cpu().numpy()
                all_emb.append(vec)

        elapsed = time.time() - start_time
        done = min(i + batch_size, len(sequences))
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(sequences) - done) / rate if rate > 0 else 0
        print(f"  [{done}/{len(sequences)}] {rate:.0f} seq/s, ETA {eta:.0f}s")

    total_time = time.time() - start_time
    print(f"完成! 耗时 {total_time:.0f}s ({total_time/60:.1f}min)")
    return np.array(all_emb)


for csv_file in csv_files:
    path = os.path.join(DATA_DIR, csv_file) if DATA_DIR else csv_file
    print(f"\n{'='*50}")
    print(f"  处理: {csv_file}")
    print(f"{'='*50}")

    df = pd.read_csv(path)
    sequences = df["sequence"].tolist()
    print(f"  序列数: {len(sequences)}")

    embeddings = extract_embeddings(sequences, batch_size=16)
    print(f"  输出维度: {embeddings.shape}")

    out_name = csv_file.replace(".csv", "_esm2.npy")
    np.save(out_name, embeddings)
    mb = embeddings.nbytes / 1024 / 1024
    print(f"  已保存: {out_name} ({mb:.1f} MB)")

print(f"\n{'='*50}")
print(f"  所有嵌入提取完成！")
print(f"{'='*50}")

# %% 下载
# @title 下载结果
print("\n生成的 .npy 文件列表:")
for f in sorted(Path(".").glob("amp_*_esm2.npy")):
    mb = f.stat().st_size / 1024 / 1024
    print(f"  {f.name} ({mb:.1f} MB)")

if IN_COLAB:
    print("\n开始下载到本地...")
    for f in sorted(Path(".").glob("amp_*_esm2.npy")):
        files.download(str(f))
    print("下载完成!")

# %% [markdown]
# ## 下载完成后
# 
# 将下载的 .npy 文件放到:
# ```
# See paths.py: FEATURE_DIR. Upload your *.npy files to the features/ directory.
# ```
# 
# 然后运行:
# ```
# python src/05_merge_features.py
# ```

# Note: This script is designed for Google Colab GPU. For local CPU, use 04_esm2_local.py.
