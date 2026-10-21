import re
import sys
from collections import OrderedDict
from difflib import SequenceMatcher

sys.stdout.reconfigure(encoding='utf-8')

with open('documents/references.bib', 'r', encoding='utf-8') as f:
    text = f.read()

def parse_bib_entry(entry_str):
    m = re.match(r'@(\w+)\s*\{\s*([^,]+),', entry_str)
    if not m:
        return None
    etype = m.group(1).lower()
    key = m.group(2).strip()
    body = entry_str[m.end():].rstrip()
    if body.endswith('}'):
        body = body[:-1]
    fields = OrderedDict()
    field_pattern = re.compile(r'([a-zA-Z_]+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|"([^"]*)"|([0-9a-zA-Z_]+))', re.DOTALL)
    for fm in field_pattern.finditer(body):
        fname = fm.group(1).lower()
        val = fm.group(2) if fm.group(2) is not None else (fm.group(3) if fm.group(3) is not None else fm.group(4))
        val = re.sub(r'\s+', ' ', val.strip())
        fields[fname] = val
    return {'type': etype, 'key': key, 'fields': fields, 'raw': entry_str}

raw_blocks = list(re.finditer(r'@(\w+)\s*\{\s*([^,]+),', text))
parsed_entries = []
for i in range(len(raw_blocks)):
    start = raw_blocks[i].start()
    end = raw_blocks[i+1].start() if i + 1 < len(raw_blocks) else len(text)
    block = text[start:end].strip()
    p = parse_bib_entry(block)
    if p:
        p['idx'] = i
        parsed_entries.append(p)

# Compare similarity
for i in range(len(parsed_entries)):
    for j in range(i+1, len(parsed_entries)):
        t1 = parsed_entries[i]['fields'].get('title', '')
        t2 = parsed_entries[j]['fields'].get('title', '')
        ratio = SequenceMatcher(None, t1.lower(), t2.lower()).ratio()
        if ratio > 0.85:
            k1 = parsed_entries[i]['key']
            k2 = parsed_entries[j]['key']
            print(f"Similarity {ratio:.2f}: [{parsed_entries[i]['idx']}] ({k1}) vs [{parsed_entries[j]['idx']}] ({k2})")
            print(f"  T1: {t1}")
            print(f"  T2: {t2}\n")
