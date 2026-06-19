from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR
﻿path = str(PROJECT_ROOT / "reports/manuscript.md")
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

fixes = 0

# Fix 1: AUC 0.987 in learning curve summary -> 0.984
old = '150M测试AUC 0.987与650M的0.979在95%置信区间内无显著差异'
new = '150M测试AUC 0.984与650M的0.979在95%置信区间内无显著差异'
if old in content:
    content = content.replace(old, new); fixes += 1; print('Fixed: AUC 0.987->0.984')

# Fix 2: Duplicate section 3.4 - rename hemolysis 3.4 -> 3.5, biomaterial 3.5 -> 3.6
old = '### 3.4 溶血毒性预测性能'
new = '### 3.5 溶血毒性预测性能'
if old in content:
    content = content.replace(old, new); fixes += 1; print('Fixed: hemolysis 3.4->3.5')

old = '### 3.5 面向缓释型生物材料场景的AMP筛选'
new = '### 3.6 面向缓释型生物材料场景的AMP筛选'
if old in content:
    content = content.replace(old, new); fixes += 1; print('Fixed: biomaterial 3.5->3.6')

old = '#### 3.5.1 接枝位点分析'
new = '#### 3.6.1 接枝位点分析'
if old in content:
    content = content.replace(old, new); fixes += 1; print('Fixed: grafting 3.5.1->3.6.1')

# Fix 3: The text reference "3.4节独立模型预测" -> "3.5节独立模型预测"
old = '（由3.4节独立模型预测）'
new = '（由3.5节独立模型预测）'
if old in content:
    content = content.replace(old, new); fixes += 1; print('Fixed: 3.4->3.5 reference in text')

# Fix 4: Check for reference to "3.5" that might now mean biomaterial
# The hemolysis section says things about hemolysis model, those are fine at 3.5
# The biomaterial section says things about S_free, those need to refer to 3.6 now

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'\nTotal fixes: {fixes}')
print(f'Lines: {len(content.split(chr(10)))}')
