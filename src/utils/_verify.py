path = str(PROJECT_ROOT / "reports/manuscript.md")
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Verify 0.064 and 0.082 are gone
for n in ['0.064', '0.082']:
    if n in content:
        print(f'WARNING: {n} still present')
    else:
        print(f'CLEAN: {n} removed')

# Verify the duplicate phrase is gone
if '模型规模对比揭示模型规模对比' in content:
    print('WARNING: duplicate phrase still present')
else:
    print('CLEAN: duplicate phrase removed')

# Verify no broken punctuation
import re
import sys, os
_utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_utils_dir, '..'))
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR
broken = re.findall(r'[。，；]{2}', content)
if broken:
    print(f'WARNING: {len(broken)} broken punctuation marks')
else:
    print('CLEAN: no broken punctuation')

# Verify the '0.015' occurrences are all correct
for m in re.finditer(r'.{0,40}0\.015.{0,40}', content):
    ctx = content[max(0,m.start()-20):m.end()+20]
    print(f'0.015 at: ...{ctx}...')

print(f'\nTotal lines: {len(content.split(chr(10)))}')
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
