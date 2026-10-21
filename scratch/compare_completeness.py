import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open('documents/references.bib', 'r', encoding='utf-8') as f:
    text = f.read()

matches = list(re.finditer(r'@(\w+)\s*\{\s*([^,]+),', text))
entries = []

for i in range(len(matches)):
    start = matches[i].start()
    end = matches[i+1].start() if i + 1 < len(matches) else len(text)
    entry_text = text[start:end].strip()
    etype = matches[i].group(1)
    key = matches[i].group(2).strip()
    title_m = re.search(r'\btitle\s*=\s*[\{"](.+?)[\}"](?:\s*,|\s*\n|\s*\})', entry_text, re.DOTALL | re.IGNORECASE)
    title = title_m.group(1).strip() if title_m else ""
    title_norm = re.sub(r'[^a-zA-Z0-9]+', '', title).lower()
    
    entries.append({
        'idx': i,
        'type': etype,
        'key': key,
        'title': title,
        'title_norm': title_norm,
        'raw': entry_text
    })

# Compare duplicates
from collections import defaultdict
t_map = defaultdict(list)
for e in entries:
    t_map[e['title_norm']].append(e)

for t, grp in t_map.items():
    if len(grp) > 1:
        print(f"=== Duplicate Title: {grp[0]['title'][:60]} ===")
        for e in grp:
            print(f"[{e['idx']}] Key: {e['key']} | Fields count: {len(e['raw'].splitlines())}")
            # check if doi or publisher or pages exists
            has_doi = 'doi' in e['raw'].lower()
            has_journal = 'journal' in e['raw'].lower() or 'booktitle' in e['raw'].lower()
            print(f"     DOI: {has_doi} | Venue: {has_journal}")
