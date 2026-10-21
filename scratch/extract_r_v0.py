import pypdf
import sys

sys.stdout.reconfigure(encoding='utf-8')

reader = pypdf.PdfReader('documents/R_v0.pdf')
full_text = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    full_text.append(f"=== PAGE {i+1} ===\n{text}\n")

with open('documents/R_v0.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(full_text))

print(f"Extracted {len(reader.pages)} pages to documents/R_v0.txt")
