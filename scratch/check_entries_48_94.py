import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open('documents/references.bib', 'r', encoding='utf-8') as f:
    text = f.read()

matches = list(re.finditer(r'@(\w+)\s*\{\s*([^,]+),', text))
for i in range(48, len(matches)):
    start = matches[i].start()
    end = matches[i+1].start() if i + 1 < len(matches) else len(text)
    etype = matches[i].group(1)
    key = matches[i].group(2).strip()
    entry_text = text[start:end].strip()
    title_m = re.search(r'\btitle\s*=\s*[\{"](.+?)[\}"](?:\s*,|\s*\n|\s*\})', entry_text, re.DOTALL | re.IGNORECASE)
    title = title_m.group(1).strip() if title_m else ""
    print(f"[{i}] {key:<28} | {title[:60]}")
