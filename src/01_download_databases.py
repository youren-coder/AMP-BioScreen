"""
01_download_databases.py — AMP 数据收集

从多个可访问的来源获取 AMP 数据。
支持 UniProt API（最稳定）、以及 DRAMP/APD3/DBAASP 的自动或手动下载。
"""

import os, sys, argparse, requests, json, time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path("D:/Research_AI_Bio/02_Databases").resolve()
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("D:/Research_AI_Bio/03_Datasets/Processed").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def retry_request(url, max_retries=4, timeout=60):
    """带重试的 HTTP 请求"""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 400:
                print(f"API error: {resp.text[:80]}")
                return None
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt < max_retries - 1:
                wait = [5, 15, 30, 60][attempt]
                print(f"  重试 {attempt+1}/{max_retries} ({wait}s)... {type(e).__name__}")
                time.sleep(wait)
            else:
                print(f"  [失败] {type(e).__name__} 重试 {max_retries} 次仍失败")
                return None
    return None


def fetch_from_uniprot(query, max_results=500, format_type="tsv"):
    """从 UniProt API 获取数据"""
    fields = "accession,sequence,length,organism_name,protein_name"
    url = (
        f"https://rest.uniprot.org/uniprotkb/search?query={query}"
        f"&format={format_type}&fields={fields}&size={max_results}"
    )
    resp = retry_request(url, max_retries=3, timeout=60)
    return resp


def download_uniprot_amps():
    """从 UniProt 下载 AMP 序列（最可靠的来源）"""
    print(f"\n{'='*60}")
    print("  来源: UniProt — 抗菌肽序列 (最稳定)")
    print(f"{'='*60}")

    queries = [
        ("antimicrobial", "antimicrobial AND length:[5 TO 100]"),
        ("defensin", "defensin AND length:[5 TO 100]"),
        ("cathelicidin", "cathelicidin AND length:[5 TO 100]"),
    ]
    all_seqs = set()
    records = []

    for label, query in queries:
        print(f"  搜索: {label}...")
        resp = fetch_from_uniprot(query, max_results=500)
        if resp is None or resp.status_code != 200:
            print(f"  {label}: 跳过 (无法访问)")
            continue

        lines = resp.text.strip().splitlines()
        print(f"  {label}: 返回 {len(lines)-1} 条")

        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) >= 2:
                seq = parts[1].strip().upper()
                if seq and seq not in all_seqs:
                    all_seqs.add(seq)
                    records.append({
                        "sequence": seq,
                        "source": f"UniProt_{label}",
                        "label_amp": 1,
                        "label_hemolysis": -1,
                        "accession": parts[0] if len(parts) > 0 else "",
                        "organism": parts[3] if len(parts) > 3 else "",
                    })

    if records:
        df = pd.DataFrame(records)
        print(f"\n  UniProt 总计: {len(df)} 条去重后 AMP 序列")
        return df
    else:
        print("  UniProt: 无法获取数据")
        return pd.DataFrame()

def download_file(url, dest_path, timeout=60, max_retries=2):
    """下载文件并显示进度条"""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest_path, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True,
                          desc=dest_path.name) as pbar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return False
    return False


def try_download(name, urls):
    """尝试多个 URL 下载同一个数据库"""
    ext = ".fasta" if name in ["dramp", "apd3"] else ".csv"
    dest = RAW_DIR / f"{name}{ext}"

    if dest.exists():
        print(f"  文件已存在，跳过下载: {dest}")
        return dest

    for url in urls:
        print(f"  尝试: {url}")
        if download_file(url, dest):
            print(f"  成功: {dest}")
            return dest
    return None


def parse_fasta(filepath):
    """解析 FASTA"""
    records = []
    seq_id, seq = None, []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    records.append((seq_id, "".join(seq)))
                seq_id = line[1:].split()[0]
                seq = []
            else:
                seq.append(line.upper())
        if seq_id is not None:
            records.append((seq_id, "".join(seq)))
    return records


def generate_negative_samples(positive_seqs, n=5000, min_len=5, max_len=60):
    """生成随机负样本"""
    import random
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
    parser = argparse.ArgumentParser(description="收集 AMP 数据")
    parser.add_argument("--uniprot-only", action="store_true",
        help="仅从 UniProt 获取数据（最稳定）")
    parser.add_argument("--try-databases", nargs="+",
        choices=["dramp", "apd3", "dbaasp", "hemolytik"],
        default=[], help="额外尝试下载的数据库")
    parser.add_argument("--neg-samples", type=int, default=5000,
        help="负样本数量")
    args = parser.parse_args()

    all_dfs = []

    # 1. UniProt（最稳定，始终尝试）
    df_uniprot = download_uniprot_amps()
    if len(df_uniprot) > 0:
        all_dfs.append(df_uniprot)

    # 2. 尝试其他数据库（可选）
    for db_name in args.try_databases:
        print(f"\n{'='*60}")
        print(f"  尝试额外数据库: {db_name}")
        print(f"{'='*60}")

        urls = {
            "dramp": [
                "http://dramp.cpu-bioinfor.org/downloads/DRAMP_AMP_Sequence.fasta",
            ],
            "apd3": [
                "https://aps.unmc.edu/APD3/APD3_fasta.txt",
            ],
            "dbaasp": [
                "https://dbaasp.org/download?type=csv",
            ],
            "hemolytik": [
                "http://crdd.osdd.net/raghava/hemolytik/download/Hemolytik.csv",
            ],
        }

        result = try_download(db_name, urls.get(db_name, []))

        if result is None:
            print(f"  -> 自动下载失败")
            print(f"  -> 可选：手动下载后放到 {RAW_DIR}")
            continue

        print(f"  正在处理: {result.name}...")
        try:
            if result.suffix == ".fasta":
                records = parse_fasta(result)
                df = pd.DataFrame(records, columns=["sequence_id", "sequence"])
                df["source"] = db_name.upper()
                df["label_amp"] = 1
                df["label_hemolysis"] = -1
                all_dfs.append(df)
                print(f"  {db_name}: {len(df)} 条序列")
            elif result.suffix == ".csv":
                df_raw = pd.read_csv(result, encoding="utf-8", low_memory=False)
                seq_col = [c for c in ["Sequence", "Peptide sequence", "sequence"]
                           if c in df_raw.columns]
                if seq_col:
                    df = pd.DataFrame()
                    df["sequence"] = df_raw[seq_col[0]]
                    df["source"] = db_name.upper()
                    df["label_amp"] = 1
                    if "Hemolytic activity" in df_raw.columns:
                        df["label_hemolysis"] = df_raw["Hemolytic activity"].apply(
                            lambda x: 1 if str(x).strip() in ["1", "yes", "active", "Hemolytic"] else 0
                        )
                    else:
                        df["label_hemolysis"] = -1
                    all_dfs.append(df)
                    print(f"  {db_name}: {len(df)} 条记录")
        except Exception as e:
            print(f"  处理失败: {e}")

    # 汇总
    if not all_dfs:
        print("\n未获取到任何数据。")
        print("请确保网络通畅，或使用 --uniprot-only 重试。")
        return

    df_pos = pd.concat(all_dfs, ignore_index=True)
    df_pos["sequence"] = df_pos["sequence"].str.strip().str.upper()
    df_pos = df_pos[df_pos["sequence"].str.len() >= 5]
    df_pos = df_pos.drop_duplicates(subset=["sequence"])
    print(f"\n正样本总计: {len(df_pos)}")

    # 负样本
    print("\n生成负样本...")
    df_neg = generate_negative_samples(df_pos["sequence"].tolist(), n=args.neg_samples)
    print(f"负样本: {len(df_neg)}")

    df_all = pd.concat([df_pos, df_neg], ignore_index=True)
    out_path = OUTPUT_DIR / "amp_data_raw.csv"
    df_all.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"  数据已保存: {out_path}")
    print(f"  总记录:     {len(df_all)}")
    print(f"  AMP+样本:   {df_all['label_amp'].sum()}")
    print(f"  非AMP样本:  {(df_all['label_amp'] == 0).sum()}")
    print(f"  溶血标注:   {(df_all['label_hemolysis'] >= 0).sum()}")
    print(f"{'='*60}")

    # 如果自动下载失败，打印手动下载指南
    print("\n手动下载指南（如需要补充数据）:")
    print("  1. DRAMP: 访问 http://dramp.cpu-bioinfor.org/downloads/")
    print("  2. APD3:  访问 https://aps.unmc.edu/downloads")
    print("  3. DBAASP: 访问 https://dbaasp.org/download")
    print("  4. 下载后放到 D:/Research_AI_Bio/02_Databases/")
    print("  5. 重新运行本脚本即可自动处理")


if __name__ == "__main__":
    main()

def download_via_powershell(url, dest_path, max_retries=3):
    """用 PowerShell 下载（在 Python requests 连不上时作为后备）"""
    import subprocess
    for attempt in range(max_retries):
        cmd = (
            f'Invoke-WebRequest -Uri "{url}"'
            f' -TimeoutSec 60 -UseBasicParsing'
            f' | Select-Object -ExpandProperty Content'
            f' | Out-File "{dest_path}" -Encoding utf8'
        )
        try:
            result = subprocess.run(
                ["powershell", "-Command", cmd],
                capture_output=True, text=True, timeout=90
            )
            if result.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 100:
                return True
            print(f"  PS下载: 状态码 {result.returncode}, 大小 {dest_path.stat().st_size if dest_path.exists() else 0}")
        except subprocess.TimeoutExpired:
            print(f"  超时: {url[:60]}...")
        except Exception as e:
            print(f"  PS下载失败: {type(e).__name__}")
        if attempt < max_retries - 1:
            time.sleep(5)
    return False


def fetch_from_uniprot_powershell(query, file_label):
    """用 PowerShell 从 UniProt 获取数据"""
    fields = "accession,sequence,length,organism_name,protein_name"
    url = (
        "https://rest.uniprot.org/uniprotkb/search?query="
        f"{query}&format=tsv&fields={fields}&size=500"
    )
    dest = RAW_DIR / f"uniprot_{file_label}.tsv"

    if dest.exists():
        print(f"  文件已存在: {dest.name} ({dest.stat().st_size} bytes)")
        return dest

    print(f"  通过 PowerShell 下载: {file_label}...")
    if download_via_powershell(url, dest):
        return dest
    return None


def parse_uniprot_tsv(filepath):
    """解析 UniProt TSV 文件"""
    import csv
    records = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        seq_col = "Sequence"
        for row in reader:
            seq = row.get(seq_col, "").strip().upper()
            if not seq or len(seq) < 5:
                continue
            records.append({
                "sequence": seq,
                "source": f"UniProt_{filepath.stem.replace('uniprot_', '')}",
                "label_amp": 1,
                "label_hemolysis": -1,
            })
    df = pd.DataFrame(records)
    return df
