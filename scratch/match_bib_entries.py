import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

bib_path = 'documents/references.bib'
with open(bib_path, 'r', encoding='utf-8') as f:
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

# Check title duplicates
title_map = defaultdict(list)
key_map = defaultdict(list)

for e in entries:
    title_map[e['title_norm']].append(e)
    key_map[e['key']].append(e)

print(f"Total entries: {len(entries)}")
print(f"Unique title_norms: {len(title_map)}")
print(f"Unique keys: {len(key_map)}")

# List entries that have multiple items
for t, group in title_map.items():
    if len(group) > 1:
        print(f"\nDuplicate group for title: {group[0]['title'][:60]}")
        for e in group:
            print(f"  [{e['idx']}] key: {e['key']}, type: {e['type']}")

# Also check if any entries with unique title_norm have duplicate keys
print("\n--- Entries with same key but different titles ---")
for k, group in key_map.items():
    if len(group) > 1:
        tnorms = set(e['title_norm'] for e in group)
        if len(tnorms) > 1:
            print(f"Key {k} has DIFFERENT titles:")
            for e in group:
                print(f"  [{e['idx']}] {e['title'][:60]}")
