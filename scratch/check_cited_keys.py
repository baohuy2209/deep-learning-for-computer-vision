import re
import sys
import glob

sys.stdout.reconfigure(encoding='utf-8')

cited_keys = set()

# Check all md and tex files in documents and workspace
for fpath in glob.glob('documents/**/*.md', recursive=True) + glob.glob('**/*.tex', recursive=True):
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    cites = re.findall(r'\\cite\{([^}]+)\}', text)
    for c in cites:
        for k in c.split(','):
            cited_keys.add(k.strip())

print(f"Total unique keys cited in documents: {len(cited_keys)}")
print(sorted(list(cited_keys)))
