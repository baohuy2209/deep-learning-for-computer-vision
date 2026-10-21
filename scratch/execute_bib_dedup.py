import re
import sys
from collections import OrderedDict

sys.stdout.reconfigure(encoding='utf-8')

# 1. Read existing references.bib
with open('documents/references.bib', 'r', encoding='utf-8') as f:
    text = f.read()

CITED_KEYS = {
    'agresti2010analysis', 'alantary2021multi', 'albadawy2018deep', 'aptos2019kaggle',
    'araujo2020drgraduate', 'cao2020rank', 'chilukoti2024reliable', 'cohen1968weighted',
    'cuadros2009eyepacs', 'cun2026global', 'decenciere2014feedback', 'deepmil2018iet',
    'delarosa2020large', 'dharrao2025ai', 'etdrs1991report10', 'folorunsho2025explainable',
    'galappaththige2024generalizing', 'gatedstatespace2026', 'gulrajani2021search',
    'gulshan2016development', 'hashmi2026dualresolution', 'he2016deep', 'ilse2018attention',
    'li2019diagnostic', 'liu2022regularized', 'lu2026mramnet', 'mccullagh1980regression',
    'mil2023acm', 'mohanty2023using', 'mutawa2023transfer', 'orvitdr2026', 'porwal2018indian',
    'quellec2021interpretable', 'retitransnet2026', 'shi2021deep', 'takahashi2020convolutional',
    'tan2019efficientnet', 'ting2017development', 'tyagi2026enhanced', 'vargas2020cumulative',
    'vitmae2024plos', 'wang2018non', 'wilkinson2003proposed', 'woo2018cbam', 'wu2025diabetic',
    'yang2022classification', 'zhang2018top'
}

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

# Explicit duplicate mapping (grouping entries that are the same paper)
# Key mapping to canonical key
alias_map = {
    'early1991grading': 'etdrs1991report10',
    'al2021multi': 'alantary2021multi',
    'atacs2026reti': 'retitransnet2026',
    'shaban2020convolutional': 'takahashi2020convolutional',
    'kundu2026orvit': 'orvitdr2026',
    'gupta2026reliable': 'gatedstatespace2026',
    'yang2024vision': 'vitmae2024plos',
    'zhou2018deep': 'deepmil2018iet',
    'avolio2023detection': 'mil2023acm',
    'papadopoulos2021interpretable': 'quellec2021interpretable',
    'araujo2020dr': 'araujo2020drgraduate',
    'shi2023deep': 'shi2021deep',
    'gulrajani2020search': 'gulrajani2021search',
}

# Distinguish brain tumor paper
for e in parsed_entries:
    if e['key'] == 'albadawy2018deep' and 'brain' in e['fields'].get('title', '').lower():
        e['key'] = 'albadawy2018brain'
    # map alias to canonical
    if e['key'] in alias_map:
        e['key'] = alias_map[e['key']]

# Group by canonical key
by_key = OrderedDict()
for e in parsed_entries:
    k = e['key']
    if k not in by_key:
        by_key[k] = []
    by_key[k].append(e)

print(f"Total canonical keys after alias mapping: {len(by_key)}")

# Merge duplicates under each key
deduped_entries = []

for k, grp in by_key.items():
    if len(grp) == 1:
        deduped_entries.append(grp[0])
        continue
    
    # Merge entries
    # Pick type
    chosen_type = grp[0]['type']
    for e in grp:
        if e['type'] in ['article', 'inproceedings', 'book']:
            chosen_type = e['type']
            break
            
    # Merge fields
    merged_fields = OrderedDict()
    # Preferred field order
    order = ['title', 'author', 'journal', 'booktitle', 'volume', 'number', 'pages', 'year', 'publisher', 'doi', 'note']
    
    all_fnames = []
    for fn in order:
        if any(fn in e['fields'] for e in grp):
            all_fnames.append(fn)
    for e in grp:
        for fn in e['fields']:
            if fn not in all_fnames:
                all_fnames.append(fn)
                
    for fn in all_fnames:
        vals = [e['fields'][fn] for e in grp if fn in e['fields'] and e['fields'][fn]]
        if not vals:
            continue
        # For title, pick clean title without extra outer braces if possible, or longer
        if fn == 'title':
            best_val = max(vals, key=len)
        elif fn == 'author':
            # pick the one with most details
            best_val = max(vals, key=len)
        elif fn == 'doi':
            best_val = vals[0]
        else:
            best_val = max(vals, key=len)
        merged_fields[fn] = best_val
        
    deduped_entries.append({
        'type': chosen_type,
        'key': k,
        'fields': merged_fields
    })

print(f"Total deduped entries: {len(deduped_entries)}")

# Verify all cited keys exist
present_keys = set(e['key'] for e in deduped_entries)
print(f"Are all 47 cited keys present? {CITED_KEYS.issubset(present_keys)}")
if not CITED_KEYS.issubset(present_keys):
    print("Missing:", CITED_KEYS - present_keys)

# Format to BibTeX string
lines = []
for e in deduped_entries:
    lines.append(f"@{e['type']}{{{e['key']},")
    # format fields with alignment
    max_flen = max(len(fn) for fn in e['fields']) if e['fields'] else 0
    for fn, fval in e['fields'].items():
        # clean double braces if any
        if fval.startswith('{') and fval.endswith('}'):
            inner = fval[1:-1]
            if not ('{' in inner or '}' in inner):
                fval = inner
        padding = " " * (max_flen - len(fn))
        lines.append(f"  {fn}{padding} = {{{fval}}},")
    # remove trailing comma from last field line
    if len(e['fields']) > 0:
        lines[-1] = lines[-1][:-1]
    lines.append("}\n")

formatted_bib = "\n".join(lines)

# Write to documents/references.bib
with open('documents/references.bib', 'w', encoding='utf-8') as f:
    f.write(formatted_bib)

print("Successfully wrote cleaned documents/references.bib!")
