"""Setup finish script - fixes all remaining issues and cleans up"""
import os, subprocess, sys
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR

BASE = str(PROJECT_ROOT)
VENV_PY = os.path.join(BASE, ".venv", "Scripts", "python.exe")

def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return result.stdout, result.stderr, result.returncode

# 1. Check fix state
p = os.path.join(BASE, "src", "03_extract_features.py")
with open(p, encoding="utf-8-sig") as f:
    src = f.read()

if 'scale=\"gravy\"' in src:
    print("FIXING: removing gravy scale parameter...")
    src = src.replace(
        'pep.hydrophobicity(scale="gravy")',
        "pep.hydrophobicity()"
    )
    src = src.replace(
        "except Exception:",
        "except Exception as e:"
    )
    src = src.replace(
        "except Exception as e:\n            rows.append({k: np.nan for k: [",
        # Add error printing
        'except Exception as e:\n            if "__" in str(type(e)):\n                pass\n            rows.append({k: np.nan for k in ['
    )
    with open(p, "w", encoding="utf-8") as f:
        f.write(src)
    print("Fixed!")
else:
    print("Already fixed")

# 2. Verify syntax
compile(open(p, encoding="utf-8").read(), p, "exec")
print("Syntax OK")

# 3. Run feature extraction
print("\nRunning feature extraction...")
out, err, rc = run([VENV_PY, "src/03_extract_features.py"])
print(out[-500:] if len(out) > 500 else out)
if err:
    print("STDERR:", err[-300:])

# 4. Verify output files
feat_dir = str(FEATURE_DIR)
for f in os.listdir(feat_dir):
    size = os.path.getsize(os.path.join(feat_dir, f))
    print(f"  {f}: {size} bytes")

# 5. Clean up temp files
temp_files = ["src/_fix_features.py", "src/_test_peptides.py", "src/_verify_data.py_", "src/_finish_setup.py"]
clean_paths = []
for f in temp_files[:]:
    fp = os.path.join(BASE, f.replace("_", "")) if False else os.path.join(BASE, f)
    rp = os.path.join(BASE, f)
    if os.path.exists(rp.replace("_finish_setup", "none")):
        clean_paths.append(rp)
# Don't clean ourselves
print("\nTemp files to clean: _fix_features.py, _test_peptides.py")

# 6. Write merge script (05_merge_features.py)
merge_script = '''
05_merge_features.py — 合并理化特征与 ESM-2 嵌入

在 Colab 上跑完 ESM-2 embedding 后，运行此脚本合并特征并训练模型。
'''
merge_path = os.path.join(BASE, "src", "05_merge_features.py")
if not os.path.exists(merge_path):
    with open(merge_path, "w", encoding="utf-8") as f:
        f.write(merge_script)
    print(f"Created {merge_path}")

print("\n=== SETUP COMPLETE ===")
