path = 'D:/Research_AI_Bio/07_Reports/论文初稿.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
import re

# Check if "orphan" figures/tables are actually referenced without parentheses
# e.g. "如图7所示" or "表1列出了"
for prefix, marker, label in [('图', 'Fig', 'Figure'), ('表', 'Tbl', 'Table')]:
    # Search for references without parentheses
    refs_no_paren = set()
    for m in re.finditer(r'(?:见|如|在|的|和|与|，|。|\s)' + prefix + r'(\d+[a-e]?)', content):
        refs_no_paren.add(m.group(1))
    
    # Combined with parenthetical refs
    paren_refs = set()
    for m in re.finditer(r'[（(]' + prefix + r'(\d+[a-e]?)[）)]', content):
        paren_refs.add(m.group(1))
    
    all_refs = refs_no_paren | paren_refs
    
    # Get captions
    if prefix == '图':
        caps = set()
        for m in re.finditer(r'!\[图(\d+[a-e]?)', content):
            caps.add(m.group(1))
    else:
        caps = set()
        for m in re.finditer(r'\*\*表(\d+[a-z]?)', content):
            caps.add(m.group(1))
    
    unreferenced = caps - all_refs
    if unreferenced:
        print(f'{label}: Unreferenced: {unreferenced}')
    else:
        print(f'{label}: All {len(caps)} referenced (OK)')

# Final: re-run all checks
print('\n=== FINAL AUDIT ===')
lines = content.split(chr(10))

# Check for remaining issues
issues = []
for kw in ['校园网', '校园', '遗憾', 'TMP', '~~']:
    if kw in content:
        issues.append(f'{kw} still present')

# Check section numbers for duplicates
sec_nums = re.findall(r'^(#{2,4})\s+([\d.]+)', content, re.MULTILINE)
from collections import Counter
sec_counts = Counter(n for _, n in sec_nums)
for n, c in sec_counts.items():
    if c > 1:
        issues.append(f'Duplicate section: {n} ({c}x)')

if issues:
    for i in issues: print(f'ISSUE: {i}')
else:
    print('ALL CLEAN')
    print(f'Lines: {len(lines)}')
    print(f'Sections: {len(sec_nums)}')
