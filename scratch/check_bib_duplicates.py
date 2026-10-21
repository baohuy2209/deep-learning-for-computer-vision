import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

bib_path = 'documents/references.bib'

with open(bib_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Parse bib entries
# Pattern to find @type{key, ...}
entries = re.findall(r'@(\w+)\s*\{\s*([^,]+),', content)
print(f"Total entries found: {len(entries)}")

# Count keys
key_counts = defaultdict(list)
for entry_type, key in entries:
    key_counts[key.strip()].append(entry_type)

print("\n--- Duplicate Keys ---")
dup_keys = {k: v for k, v in key_counts.items() if len(v) > 1}
if dup_keys:
    for k, v in dup_keys.items():
        print(f"Key '{k}': {len(v)} occurrences ({v})")
else:
    print("No identical citation keys found.")

# Now parse full entries and check for duplicate titles/DOIs
# Split by @ at start of line
raw_blocks = re.split(r'\n(?=@)', content)
print(f"\nTotal raw blocks: {len(raw_blocks)}")

titles = defaultdict(list)
dois = defaultdict(list)

def clean_str(s):
    if not s:
        return ""
    # remove braces, quotes, spaces, lowercase
    return re.sub(r'[\s\{\}\"\']+', '', s).lower()

parsed_entries = []

for block in raw_blocks:
    block = block.strip()
    if not block or not block.startswith('@'):
        continue
    m = re.match(r'@(\w+)\s*\{\s*([^,]+),', block)
    if not m:
        continue
    etype, key = m.group(1), m.group(2).strip()
    
    # extract title
    title_m = re.search(r'\btitle\s*=\s*[\{"](.+?)[\}"](?:\s*,|\s*\n|\s*\})', block, re.DOTALL | re.IGNORECASE)
    title = title_m.group(1).strip() if title_m else ""
    title_norm = clean_str(title)
    
    # extract doi
    doi_m = re.search(r'\bdoi\s*=\s*[\{"](.+?)[\}"](?:\s*,|\s*\n|\s*\})', block, re.IGNORECASE)
    doi = doi_m.group(1).strip() if doi_m else ""
    doi_norm = clean_str(doi)
    
    parsed_entries.append({
        'type': etype,
        'key': key,
        'title': title,
        'title_norm': title_norm,
        'doi': doi,
        'doi_norm': doi_norm,
        'raw': block
    })
    
    if title_norm:
        titles[title_norm].append(key)
    if doi_norm:
        dois[doi_norm].append(key)

print("\n--- Duplicate Titles ---")
dup_titles = {t: keys for t, keys in titles.items() if len(keys) > 1}
for t, keys in dup_titles.items():
    print(f"Title: '{parsed_entries[[e['key'] for e in parsed_entries].index(keys[0])]['title']}'")
    print(f"  Keys: {keys}")

print("\n--- Duplicate DOIs ---")
dup_dois = {d: keys for d, keys in dois.items() if len(keys) > 1}
for d, keys in dup_dois.items():
    print(f"DOI: {d} -> Keys: {keys}")
