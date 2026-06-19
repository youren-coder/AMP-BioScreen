path = 'D:/Research_AI_Bio/07_Reports/论文初稿.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
import re

# ISSUE 1: Find remaining AUC 0.987
for m in re.finditer(r'.{0,40}AUC.{0,5}0\.987.{0,40}', content):
    ctx = content[max(0,m.start()-20):m.end()+20]
    print(f'AUC 0.987: ...{ctx}...')

# ISSUE 2: Find F1=0.916 - is it amPEPpy reference or copy-paste error?
for m in re.finditer(r'.{0,40}F1=0\.916.{0,40}', content):
    ctx = content[max(0,m.start()-20):m.end()+20]
    print(f'F1=0.916: ...{ctx}...')

# ISSUE 3: Find duplicate section 3.4
for m in re.finditer(r'^### 3\.4', content, re.MULTILINE):
    print(f'SECTION 3.4 at pos {m.start()}: {content[m.start():m.start()+80]}')

# ISSUE 4: Check section ordering
all_secs = [(m.group(), content[:m.start()].count('\n')+1) 
            for m in re.finditer(r'^#{2,4}\s+\d[\d.]*', content, re.MULTILINE)]
print('\nSection ordering:')
for sec, line in all_secs:
    print(f'  L{line}: {sec}')
