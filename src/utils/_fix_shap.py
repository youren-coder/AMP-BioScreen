import sys, os
_utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_utils_dir, '..'))
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR
﻿import sys
sys.stdout.reconfigure(encoding='utf-8')

path = str(PROJECT_ROOT / "reports/manuscript.md")
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Reconcile hydrophob vs hmoment numbers
old_30 = '**电荷特征驱动AMP活性预测**：在16维理化特征中，净电荷（Net Charge）和等电点（pI）分别位居第2和第3位，显著领先于其他理化特征（如分子量mean|SHAP|=0.018、疏水性0.015）。高净电荷（正值）和高等电点一致推高AMP活性预测概率，这与AMPs经典分子机制完全一致[12]——带正电荷的AMP优先结合带负电荷的细菌膜磷脂头基（如磷脂酰甘油），而对电中性的真核细胞膜（富含磷脂酰胆碱和鞘磷脂）亲和力低，实现选择性杀伤。'

new_30 = '**电荷特征驱动AMP活性预测**：在16维理化特征中，净电荷（Net Charge, mean|SHAP|=0.011）和等电点（pI, 0.005）位于前5位。高净电荷一致推高AMP活性预测概率，与AMPs经典分子机制一致[12]——带正电荷的AMP优先结合带负电荷的细菌膜磷脂头基（如磷脂酰甘油）。需注意：此处讨论的是整体疏水性标量（hydrophob, mean|SHAP|=0.011），而非疏水力矩（Hydrophobic Moment, hmoment）——后者作为衡量两亲性的向量指标，其SHAP贡献（mean|SHAP|=0.026）位列所有理化特征第3位，将在3.0.1节专题分析。两亲性是AMP区别于普通阳离子肽的核心特征，不能与标量疏水性混同。'

if old_30 in content:
    content = content.replace(old_30, new_30)
    print('Fix 1: OK')
else:
    print('Fix 1: TEXT NOT FOUND - searching for partial match...')
    # Try to find what's actually there
    idx = content.find('电荷特征驱动AMP活性预测')
    if idx >= 0:
        print('Found at position', idx)
        print(content[idx:idx+300])

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('File saved')
