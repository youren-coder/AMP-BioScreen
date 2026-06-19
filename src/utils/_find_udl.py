path = 'D:/Research_AI_Bio/07_Reports/论文初稿.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the exact paragraph
idx = content.find('UniDL4BioPep（AUC 0.431）')
if idx >= 0:
    # Find the paragraph boundaries
    start = content.rfind('\n', 0, idx) + 1
    end = content.find('\n\n', idx)
    if end < 0:
        end = len(content)
    para = content[start:end]
    print('EXACT PARAGRAPH:')
    print(repr(para[:200]))
    print('...')
    print(repr(para[-200:]))
