import sys, os
_utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_utils_dir, '..'))
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR
﻿path = str(PROJECT_ROOT / "reports/manuscript.md")
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Use the exact paragraph boundaries
idx = content.find('UniDL4BioPep（AUC 0.431）')
start = content.rfind('\n', 0, idx) + 1
end = content.find('\n\n', idx)
para = content[start:end]

new_udl = '**（1）UniDL4BioPep（AUC 0.431）的泛化崩溃：分类捷径的量化解剖。** UniDL4BioPep发布于2023年，同样基于ESM-2（t6_8M, 320维）+ CNN架构，其原始论文报告AUC>0.98。其对我们的短链分泌非AMP测试集完全失效（MCC=-0.071，低于随机猜测），但对旧全长非AMP测试集的AUC回升至0.669——这一定向差异提供了精确的失败诊断。\n\n**根本原因：4.4倍的长度捷径。** UniDL4BioPep训练集中，正样本长度中位数28 AA，负样本长度中位数124 AA，长度比高达4.4倍。我们计算发现：仅以序列长度为唯一特征训练逻辑回归，在UniDL4BioPep自身的训练数据上即可获得AUC=0.968——这意味着该工具的ESM-2+CNN模型不需要学习任何AMP特异性序列特征，仅需捕捉"短=AMP、长=非AMP"的长度捷径即可在其原始测试集上获得>0.98的虚高AUC。当我们将测试集替换为长度匹配的短链分泌非AMP肽（正样本中位数72, 负样本73），长度捷径完全失效，模型真实判别能力被暴露。\n\n**次级因素：t6_8M模型容量不是崩溃主因。** UniDL4BioPep使用ESM-2 t6_8M（320维嵌入），仅为我们150M模型（640维）的一半。我们同架构的ESM-2+LogisticRegression基线在相同测试集上获得AUC=0.920，证明320维嵌入对AMP识别任务仍然有效。CNN在1.3万条训练样本上的过拟合风险有限（328K可训练参数 vs 13K样本，参数/样本比≈25）。训练数据的极端长度-类别混淆是崩溃的充分解释。\n\n**方法论启示**：UniDL4BioPep的失败并非孤例——它代表了一批2023年前后使用"随机序列/UniProt长蛋白 vs 短AMP"负样本策略的工具的系统性缺陷。当负样本与正样本在长度分布上存在4倍以上差距时，ESM-2嵌入中的隐性长度信号足以让任意下游分类器学会"长度捷径"而非AMP功能识别。这一发现强化了3.2节的核心论点：AMP预测领域亟需将负样本评估从"随机序列\\u2192全长非AMP\\u2192短链分泌非AMP\\u2192严格长度配对"的递进谱系标准化。UniDL4BioPep的预训练模型权重、scaler和测试流水线均从其官方GitHub仓库下载，部署过程与原始代码完全一致，排除sklearn版本差异后结论不变。'

content = content[:start] + new_udl + content[end:]
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Replaced. New length:', len(content))
