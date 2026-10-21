import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('documents/references.bib', 'r', encoding='utf-8') as f:
    text = f.read()

import re
matches = list(re.finditer(r'@(\w+)\s*\{\s*([^,]+),', text))
for idx in [4, 79, 46, 80]:
    start = matches[idx].start()
    end = matches[idx+1].start() if idx + 1 < len(matches) else len(text)
    print(f"=== Entry {idx} ===")
    print(text[start:end].strip())
    print()
