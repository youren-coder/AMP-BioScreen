import sys, os
_utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_utils_dir, '..'))
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR
﻿path = str(PROJECT_ROOT / "reports/manuscript.md")
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
