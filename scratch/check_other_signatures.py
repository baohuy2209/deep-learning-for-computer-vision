import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Check IDRiDXAIEvalDataset definition vs call
def_xai = [line.strip() for c in cells for line in "".join(c.get("source", [])).splitlines() if "def __init__" in line and "image_dirs" in "".join(c.get("source", []))]
print("IDRiDXAIEvalDataset init lines:")
for l in def_xai[:5]:
    print(" ", l)

call_xai = [line.strip() for c in cells for line in "".join(c.get("source", [])).splitlines() if "IDRiDXAIEvalDataset(" in line]
print("IDRiDXAIEvalDataset call lines:")
for l in call_xai:
    print(" ", l)

# Check build_model definition vs call
def_bm = [line.strip() for c in cells for line in "".join(c.get("source", [])).splitlines() if "def build_model" in line]
print("build_model def lines:")
for l in def_bm:
    print(" ", l)

# Check DRTrainer definition vs call
def_drt = [line.strip() for c in cells for line in "".join(c.get("source", [])).splitlines() if "class DRTrainer" in line]
print("DRTrainer def lines:")
for l in def_drt:
    print(" ", l)
