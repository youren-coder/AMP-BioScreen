path = str(PROJECT_ROOT / "reports/manuscript.md")
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the old paragraph about ESM dims providing hierarchical features
old_p = '**ESM-2嵌入维度提供层级化特征**：占据前10位的ESM-2嵌入维度（512, 128, 640, 256, 64, 384）呈现出多尺度分布特征，从低维（64维）到高维（640维）均有贡献。Shapley值依赖分析（Dependence Plot）表明，嵌入维度512的高SHAP值主要对应正样本（AMP），而低SHAP值区域富集负样本（非AMP），表明ESM-2的深层嵌入已自发学习到与抗菌活性相关的序列模式，无需额外fine-tuning。\n\n'

if old_p in content:
    content = content.replace(old_p, '')
    print('Removed old hierarchical features paragraph')
else:
    print('Not found')

# Also check abstract for 0.064, 0.015, 0.082
import re
import sys, os
_utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_utils_dir, '..'))
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR
for m in re.finditer(r'.{0,30}(0\.064|0\.082|0\.015).{0,30}', content):
    ctx = content[max(0,m.start()-30):m.end()+30]
    print(f'Remaining old number: {m.group(1)}: ...{ctx}...')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Done')
