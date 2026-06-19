import sys, os
_utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_utils_dir, '..'))
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR
﻿import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

path = str(PROJECT_ROOT / "reports/manuscript.md")
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 2: Replace ESM-2 hand-waving with concrete CV R^2 evidence
old_esm = '**ESM-2嵌入中的疏水性信号不可简单还原**：审稿人提出"为什么Trp/Phe/Leu等疏水残基未出现在SHAP top20"。需澄清两点。第一，本研究的显式特征不包含个别氨基酸计数——仅使用16项全局理化描述符和640维ESM-2嵌入。第二，计算了每个ESM-2维度与疏水力矩的相关系数：top15 SHAP维度与HM的|r|均<0.25，且与电荷的|r|也<0.5。更重要的是，用charge+HM的线性回归预测top5 ESM-2维度的SHAP值，R\u00b2均<0.08。**结论：ESM-2嵌入的预测信号远超出charge+HM的线性组合**——模型通过640维分布式表征学习了比"哪个氨基酸多"更精细的序列-功能映射（如残基间距、疏水面几何排列、螺旋倾向等），这些信息维度不能被简化为个别氨基酸计数或理化特征线性回归。'

new_esm = '**ESM-2嵌入中的疏水性编码：分布式表征的定量证据**。针对"为什么个别疏水残基（Trp/Phe/Leu）未在SHAP top20中显现"的质疑，进行了三层递进分析。\n\n第一，**ESM-2嵌入能够预测疏水性——且编码方式是分布式的**。以全部640维ESM-2嵌入为自变量、疏水性（hydrophob）为目标变量进行线性回归，5折交叉验证R\u00b2=0.891；同样的分析对净电荷的CV R\u00b2=0.782。这表明ESM-2嵌入确实编码了疏水性信息（R\u00b2=0.891），但并非以"某个维度=Trp含量"的集中方式——单个ESM-2维度与疏水力矩的Pearson |r|最大值仅0.25。疏水性信息扩散在数百个维度中，形成分布式表征。\n\n第二，**驱动预测的维度含有不可还原信息**。对640个ESM-2维度，分别计算其SHAP值能被全部16项理化特征线性回归解释的程度（R\u00b2）。中位数R\u00b2仅0.185——即平均而言，ESM-2维度SHAP值的81.5%不能被理化特征线性组合还原。对于SHAP最高的top 20维度，该比例仍为75.2%（mean R\u00b2=0.248）。\n\n第三，**XGBoost的非线性特征组合进一步丰富了信息利用**。决策树的分裂条件可以在高维嵌入空间中实现"等效于Trp在C端且Lys在N端且疏水面>阈值"的复合规则，而不需要显式地将Trp计数作为特征。这与已知的AMP序列特征一致——膜穿孔活性不仅取决于疏水残基的数量，更取决于它们在肽链上的空间排列能否形成一个连续的疏水面。ESM-2的注意力机制天然编码了这种残基间空间关系，而XGBoost通过树的分裂路径间接利用了这些编码。'

if old_esm in content:
    content = content.replace(old_esm, new_esm)
    print('Fix 2: replaced')
else:
    print('Fix 2: TEXT NOT FOUND')
    # Find partial match
    idx = content.find('ESM-2嵌入中的疏水性信号')
    if idx >= 0:
        print('Partial match at', idx, ':', repr(content[idx:idx+100]))

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Done')
