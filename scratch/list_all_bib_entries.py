import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

bib_path = 'documents/references.bib'
with open(bib_path, 'r', encoding='utf-8') as f:
    text = f.read()

entries = []
# Find each @entry
matches = list(re.finditer(r'@(\w+)\s*\{\s*([^,]+),', text))
print(f"Total matched entries: {len(matches)}")

for i in range(len(matches)):
    start = matches[i].start()
    end = matches[i+1].start() if i + 1 < len(matches) else len(text)
    entry_text = text[start:end].strip()
    etype = matches[i].group(1)
    key = matches[i].group(2).strip()
    
    title_m = re.search(r'\btitle\s*=\s*[\{"](.+?)[\}"](?:\s*,|\s*\n|\s*\})', entry_text, re.DOTALL | re.IGNORECASE)
    title = title_m.group(1).strip() if title_m else ""
    title_clean = re.sub(r'\s+', ' ', title)
    
    entries.append({
        'idx': i,
        'type': etype,
        'key': key,
        'title': title_clean,
        'start': start,
        'end': end,
        'raw': entry_text
    })

for e in entries:
    print(f"[{e['idx']:02d}] Key: {e['key']:<30} | Title: {e['title'][:60]}")
