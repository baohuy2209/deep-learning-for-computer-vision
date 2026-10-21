import re
import sys
from collections import OrderedDict, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# The 47 keys cited in documents/05_manuscript_draft_sections.md
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

with open('documents/references.bib', 'r', encoding='utf-8') as f:
    text = f.read()

# Helper to parse a bib entry into a dictionary of fields
def parse_bib_entry(entry_str):
    m = re.match(r'@(\w+)\s*\{\s*([^,]+),', entry_str)
    if not m:
        return None
    etype = m.group(1).lower()
    key = m.group(2).strip()
    
    # Body between first comma and last closing brace
    body = entry_str[m.end():].rstrip()
    if body.endswith('}'):
        body = body[:-1]
    
    fields = OrderedDict()
    
    # Regex to extract key = {value} or key = "value" or key = number
    # Handle multiline values
    field_pattern = re.compile(r'([a-zA-Z_]+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|"([^"]*)"|([0-9a-zA-Z_]+))', re.DOTALL)
    for fm in field_pattern.finditer(body):
        fname = fm.group(1).lower()
        val = fm.group(2) if fm.group(2) is not None else (fm.group(3) if fm.group(3) is not None else fm.group(4))
        # clean value
        val = re.sub(r'\s+', ' ', val.strip())
        fields[fname] = val
        
    return {
        'type': etype,
        'key': key,
        'fields': fields,
        'raw': entry_str
    }

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

print(f"Parsed {len(parsed_entries)} entries successfully.")

# Group entries by normalized title
def norm_title(t):
    return re.sub(r'[^a-zA-Z0-9]+', '', t).lower()

groups = defaultdict(list)
for e in parsed_entries:
    # Special handling for albadawy2018deep:
    # index 46 is review, index 80 is brain tumor segmentation
    if e['key'] == 'albadawy2018deep' and 'brain' in e['fields'].get('title', '').lower():
        e['key'] = 'albadawy2018brain'
    
    t = e['fields'].get('title', '')
    nt = norm_title(t)
    if not nt:
        # fallback to key
        nt = e['key']
    groups[nt].append(e)

print(f"Total unique title groups: {len(groups)}")

# Now merge each group into a single best entry
merged_entries = []

for nt, grp in groups.items():
    if len(grp) == 1:
        merged_entries.append(grp[0])
        continue
    
    # Multiple entries for same title!
    # 1. Pick canonical key: prefer key in CITED_KEYS, or earlier key
    chosen_key = None
    for e in grp:
        if e['key'] in CITED_KEYS:
            chosen_key = e['key']
            break
    if not chosen_key:
        # prefer earlier entry key
        chosen_key = grp[0]['key']
        
    # 2. Pick entry type: prefer article or inproceedings
    chosen_type = grp[0]['type']
    for e in grp:
        if e['type'] in ['article', 'inproceedings', 'book']:
            chosen_type = e['type']
            break
            
    # 3. Merge fields: union of fields, preferring the non-empty, longer or better formatted one
    merged_fields = OrderedDict()
    # Collect all field names in order
    all_fnames = []
    for e in grp:
        for fn in e['fields']:
            if fn not in all_fnames:
                all_fnames.append(fn)
                
    for fn in all_fnames:
        vals = [e['fields'][fn] for e in grp if fn in e['fields'] and e['fields'][fn]]
        if not vals:
            continue
        # Pick the best value: for title, remove extra braces; for author, pick longer; for doi, pick non-empty
        if fn == 'doi':
            best_val = vals[0]
        elif fn == 'author':
            best_val = max(vals, key=len)
        elif fn == 'title':
            best_val = max(vals, key=len)
        else:
            best_val = max(vals, key=len)
        merged_fields[fn] = best_val
        
    merged_entries.append({
        'type': chosen_type,
        'key': chosen_key,
        'fields': merged_fields
    })

print(f"Total merged entries: {len(merged_entries)}")

# Check keys in merged_entries
merged_keys = set(e['key'] for e in merged_entries)
print(f"Total unique keys: {len(merged_keys)}")
missing_cited = CITED_KEYS - merged_keys
print(f"Missing cited keys: {missing_cited}")

# Verify no duplicate keys
from collections import Counter
kc = Counter(e['key'] for e in merged_entries)
dup_k = {k: v for k, v in kc.items() if v > 1}
print(f"Duplicate keys in merged: {dup_k}")
