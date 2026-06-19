path = 'D:/Research_AI_Bio/07_Reports/论文初稿.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
import re

issues = []

# 1. Cross-reference audit
# Check every [N] citation exists and is in order
cited = set()
for m in re.finditer(r'\[(\d+(?:[,，\s-]+\d+)*)\]', content):
    for part in re.findall(r'\d+', m.group(1)):
        cited.add(int(part))
max_ref = 52  # we have 52 refs now
missing_cited = sorted(set(range(1, max_ref+1)) - cited)
uncited = sorted(cited - set(range(1, max_ref+1)))
if missing_cited:
    issues.append(f'UNUSED refs: {missing_cited}')
if uncited:
    issues.append(f'NON-EXISTENT refs cited: {uncited}')

# Check figure cross-references: (图N) in text vs actual captions
fig_in_text = set()
for m in re.finditer(r'[（(]图(\d+)[a-e]?[）)]', content):
    fig_in_text.add(m.group(1))
fig_captions = set()
for m in re.finditer(r'!\[图(\d+)[a-e]?', content):
    fig_captions.add(m.group(1))
if fig_in_text - fig_captions:
    issues.append(f'Text refs to non-existent figs: {fig_in_text - fig_captions}')
if fig_captions - fig_in_text:
    issues.append(f'Orphan figures not referenced: {fig_captions - fig_in_text}')

# Check table cross-references
tbl_in_text = set()
for m in re.finditer(r'[（(]表(\d+[a-z]?)[）)]', content):
    tbl_in_text.add(m.group(1))
tbl_captions = set()
for m in re.finditer(r'\*\*表(\d+[a-z]?)', content):
    tbl_captions.add(m.group(1))
if tbl_in_text - tbl_captions:
    issues.append(f'Text refs to non-existent tables: {tbl_in_text - tbl_captions}')
if tbl_captions - tbl_in_text:
    issues.append(f'Orphan tables not referenced: {tbl_captions - tbl_in_text}')

# 2. Broken punctuation
broken_count = len(re.findall(r'[。，；]{2,}', content))
if broken_count:
    issues.append(f'Broken punctuation marks: {broken_count}')

# 3. Duplicate phrases (common copy-paste errors)
for phrase in ['模型规模对比揭示模型规模对比', 'AUC 0.987', 'F1=0.916']:
    if phrase in content:
        issues.append(f'Duplicate/old phrase: {phrase}')

# 4. Figure sequential check
fig_nums = []
for m in re.finditer(r'!\[图(\d+)[a-e]?', content):
    fig_nums.append(int(m.group(1)))
prev = 0
for n in fig_nums:
    if n != prev + 1 and n != prev:  # allow multi-panel (2a,2b...)
        if n != prev + 1:
            issues.append(f'Figure jump: {prev} -> {n}')
    prev = n

# 5. Check section numbering
secs = re.findall(r'^#{2,4}\s+(\d[\d.]*)', content, re.MULTILINE)
# Check for duplicate or missing section numbers
from collections import Counter
sec_counts = Counter(secs)
for s, c in sec_counts.items():
    if c > 1:
        issues.append(f'Duplicate section number: {s} ({c}x)')

# 6. Check for 'TMP' markers
if 'TMP' in content or '~~' in content:
    issues.append('TMP markers still present')

# 7. Check abstract length (should be ~250-300 words in Chinese)
abs_start = content.find('## 摘要')
abs_end = content.find('---', abs_start) if abs_start >= 0 else -1
if abs_start >= 0 and abs_end >= 0:
    abstract = content[abs_start:abs_end]
    abs_chars = len(re.sub(r'\s', '', abstract))
    if abs_chars > 600:
        issues.append(f'Abstract too long: ~{abs_chars} chars')

# Report
print(f'Total checks: PASS={len(issues)==0}')
if issues:
    for i, issue in enumerate(issues, 1):
        print(f'  ISSUE {i}: {issue}')
else:
    print('  All cross-references, numbering, and formatting checks PASSED')
    print(f'  Lines: {len(content.split(chr(10)))}')
    print(f'  Refs: {max_ref}')
    print(f'  Figures: {len(fig_captions)}')
    print(f'  Tables: {len(tbl_captions)}')
