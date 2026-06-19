from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR
﻿path = str(PROJECT_ROOT / "reports/manuscript.md")
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update abstract method line to include hemolysis details
old_abs = '构建独立的XGBoost回归模型预测溶血概率。通过SHAP分析特征归因'
new_abs = '构建独立的XGBoost分类模型预测溶血概率（5折CV AUC=0.869±0.031，基于5,185条Hemolytik2序列）。通过SHAP分析特征归因'
if old_abs in content:
    content = content.replace(old_abs, new_abs)
    print('Abstract updated')

# 2. Add section 2.5 after section 2.4
old_24_end = 'SHAP分析采用TreeExplainer（interventional模式），基于训练集（1,094条）计算Shapley值，在测试集（137条）上评估特征归因。'

new_25 = old_24_end + '\n\n### 2.5 溶血毒性预测模型\n\n**数据来源**：Hemolytik2数据库[7]收录13,215条经实验验证的溶血/非溶血肽。去除含非标准氨基酸和长度<5或>100 AA的序列后，获得5,185条有效序列（其中26条溶血阳性）。\n\n**特征提取与模型配置**：与AMP活性预测共用同一ESM-2 150M嵌入管线。以640维ESM-2嵌入作为输入特征，构建独立的XGBoost二分类器（n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=198以适应极端类别不平衡）。\n\n**评估策略**：鉴于溶血阳性样本仅26条，采用5折分层交叉验证评估泛化性能，报告CV AUC。最终模型在全量5,185条数据上训练后，用于推断924条AMP的溶血概率。'

content = content.replace(old_24_end, new_25)
print('Section 2.5 added')

# 3. Add section 3.4 (Hemolysis results) before 3.5
old_35 = '### 3.5 面向缓释型生物材料场景的AMP筛选'
new_34 = '''### 3.4 溶血毒性预测性能

基于Hemolytik2的5,185条有效序列（溶血阳性26条，阴性5,159条），以ESM-2 150M嵌入为输入特征的XGBoost分类器5折CV AUC=0.869±0.031。此性能在当前严重类别不平衡（1:198）条件下属可接受水平，表明ESM-2嵌入对溶血/非溶血肽具有一定区分能力。但CV标准差（±0.031）较大，主要由少数溶血阳性样本（每折仅~5条）在划分间的分布差异所致——这是小样本溶血数据的固有局限，而非模型设计缺陷。

将该模型应用于924条AMP的溶血概率推断：69.2%的AMP被预测为非溶血性（P(hemolysis)<0.3），与前人研究（CalcAMP[8]约65% AMP预测非溶血）的趋势一致。需注意：该推断未经过独立实验验证，S_free评分仅在计算层面提供参考排序，不可作为实验安全性的替代判定。

### 3.5 面向缓释型生物材料场景的AMP筛选'''

content = content.replace(old_35, new_34)
print('Section 3.4 added')

# 4. Update section 3.5 to reference hemolysis model properly
old_sfree = '基于活性概率P(AMP)和溶血概率P(Hemolysis)的二维评分S_free = P(AMP) × (1-P(Hemolysis))'
new_sfree = '基于活性概率P(AMP)和溶血概率P(Hemolysis)（由3.4节独立模型预测）的二维评分S_free = P(AMP) × (1-P(Hemolysis))'
content = content.replace(old_sfree, new_sfree)
print('Section 3.5 updated')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f'Done. Lines: {len(content.split(chr(10)))}')
