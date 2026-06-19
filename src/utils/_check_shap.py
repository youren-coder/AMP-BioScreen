from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR
﻿import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

path = str(PROJECT_ROOT / "reports/manuscript.md")
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Check for remaining '0.015' references
count = content.count('0.015')
print(f'0.015 appears {count} times')

# Check for '0.064' (old SHAP number that was wrong)
count2 = content.count('0.064')
print(f'0.064 appears {count2} times')

# Check for '0.082' (another old number)
count3 = content.count('0.082')
print(f'0.082 appears {count3} times')

# Check abstract - does it have old charge SHAP numbers?
idx = content.find('摘要')
if idx >= 0:
    abstract_end = content.find('---', idx)
    abstract = content[idx:abstract_end]
    if '0.064' in abstract or '0.015' in abstract:
        print('Abstract still has old SHAP numbers')
    else:
        print('Abstract SHAP numbers OK')

# Check section 3.0 for charge/pI numbers
idx_30 = content.find('Net Charge') 
if idx_30 >= 0:
    print('\nSection 3.0 charge refs:')
    for line in content[idx_30:idx_30+2000].split('\n'):
        if 'charge' in line.lower() or 'pI' in line or '0.01' in line or '0.02' in line:
            print('  ' + line.strip()[:120])

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('\nDone')
