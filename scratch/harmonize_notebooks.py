import json
import shutil
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

DIR_PATH = "notebooks/diabetic_retinopathy/pipeline/model"
FILE_4 = os.path.join(DIR_PATH, "main_model (4).ipynb")
FILE_8 = os.path.join(DIR_PATH, "main_model (8).ipynb")
FILE_ROOT = os.path.join(DIR_PATH, "main_model.ipynb")

BACKUP_4 = os.path.join(DIR_PATH, "main_model_4_backup.ipynb")
BACKUP_8 = os.path.join(DIR_PATH, "main_model_8_backup.ipynb")
BACKUP_ROOT = os.path.join(DIR_PATH, "main_model_root_backup.ipynb")

print("[1] Creating safety backups of all 3 notebooks...")
shutil.copy2(FILE_4, BACKUP_4)
shutil.copy2(FILE_8, BACKUP_8)
shutil.copy2(FILE_ROOT, BACKUP_ROOT)
print(f"  -> Backed up (4) -> {BACKUP_4}")
print(f"  -> Backed up (8) -> {BACKUP_8}")
print(f"  -> Backed up root -> {BACKUP_ROOT}")

print("\n[2] Loading golden candidate: main_model (4).ipynb...")
with open(FILE_4, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Golden notebook loaded with {len(cells)} cells.")

# --- ENHANCEMENT 1: Cell 5 - Smart Google Drive Mount ---
print("\n[3] Upgrading Cell 5 (Smart Drive Mount)...")
cell_5_src = [
    "# ==============================================================================\n",
    "# 1. AUTOMATIC GOOGLE DRIVE MOUNTING (COLAB OR LOCAL)\n",
    "# ==============================================================================\n",
    "import os\n",
    "try:\n",
    "    from google.colab import drive\n",
    "    if not os.path.exists('/content/drive/MyDrive'):\n",
    "        print('Connecting to Google Drive...')\n",
    "        drive.mount('/content/drive')\n",
    "    else:\n",
    "        print('Google Drive already mounted at /content/drive.')\n",
    "except ImportError:\n",
    "    print('Running in local environment (google.colab not available). Drive mount skipped.')\n"
]
cells[5]["source"] = cell_5_src

# --- ENHANCEMENT 2: Cell 6 - Automated Dataset Extraction ---
print("[4] Upgrading Cell 6 (Automated Dataset Extraction)...")
cell_6_src = [
    "# ==============================================================================\n",
    "# 2. AUTOMATED DATASET UNPACKING TO COLAB LOCAL FAST SSD (/content/dataset_local)\n",
    "# ==============================================================================\n",
    "import os\n",
    "import time\n",
    "\n",
    "zip_source = '/content/drive/MyDrive/native_cropped.zip'   # Path on Google Drive\n",
    "extract_dest = '/content/dataset_local'                   # Local fast SSD directory on Colab\n",
    "\n",
    "if os.path.exists(zip_source):\n",
    "    if not os.path.exists(extract_dest) or len(os.listdir(extract_dest)) == 0:\n",
    "        os.makedirs(extract_dest, exist_ok=True)\n",
    "        print(f\"Extracting dataset archive '{zip_source}' to '{extract_dest}'...\")\n",
    "        t0 = time.time()\n",
    "        # Fast shell decompression\n",
    "        get_ipython().system('unzip -q \"$zip_source\" -d \"$extract_dest\"')\n",
    "        print(f\"-> Extraction completed in {time.time() - t0:.1f} seconds!\")\n",
    "        print(f\"-> Directory structure at '{extract_dest}':\")\n",
    "        get_ipython().system('ls -lh \"$extract_dest\"')\n",
    "    else:\n",
    "        print(f\"-> Dataset directory '{extract_dest}' already populated ({len(os.listdir(extract_dest))} items). Skipping extraction.\")\n",
    "else:\n",
    "    print(f\"[INFO] zip_source '{zip_source}' not found or running in local environment.\")\n",
    "    if os.path.exists(extract_dest):\n",
    "        print(f\"-> Found existing local dataset directory at '{extract_dest}'.\")\n",
    "    else:\n",
    "        print(f\"-> Note: Ensure dataset metadata and images are located at '{extract_dest}' or update DATA_ROOT.\")\n"
]
cells[6]["source"] = cell_6_src

# --- ENHANCEMENT 3: Cell 10 - Fixed BATCH_SIZE = 16 for Colab T4 GPU ---
print("[5] Upgrading Cell 10 (BATCH_SIZE = 16 for T4 GPU)...")
cell_10_src = "".join(cells[10]["source"])
# Replace BATCH_SIZE = 48 or 32 with BATCH_SIZE = 16
cell_10_src_updated = cell_10_src.replace(
    "BATCH_SIZE = 48  # Adjust to 16, 24, 32, or 48 based on available GPU VRAM",
    "BATCH_SIZE = 16  # Fixed at 16 for Google Colab T4 GPU (15GB VRAM) to prevent OOM"
).replace(
    "BATCH_SIZE = 32  # Adjust to 16 or 32 based on available GPU VRAM",
    "BATCH_SIZE = 16  # Fixed at 16 for Google Colab T4 GPU (15GB VRAM) to prevent OOM"
)
cells[10]["source"] = [line + "\n" for line in cell_10_src_updated.splitlines()]

# --- ENHANCEMENT 4: Final Cell - Disable runtime.unassign() ---
print("[6] Disabling runtime.unassign() in final cell...")
last_cell = cells[-1]
last_cell_src = [
    "# ==============================================================================\n",
    "# SESSION COMPLETION (AUTOMATIC DISCONNECT DISABLED TO PRESERVE RUNTIME)\n",
    "# ==============================================================================\n",
    "# To automatically unassign the Colab runtime after execution, uncomment the 2 lines below:\n",
    "# from google.colab import runtime\n",
    "# runtime.unassign()\n",
    "print('\\n[COMPLETE] All training, evaluation, threshold calibration, and publication figures finished successfully!')\n",
    "print('[INFO] Google Colab runtime kept active for manual inspection and downloading generated figures.')\n"
]
last_cell["source"] = last_cell_src

print("\n[7] Writing standardized, bug-free notebook to all 3 target paths...")
target_files = [FILE_4, FILE_8, FILE_ROOT]
for tf in target_files:
    with open(tf, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  -> Successfully synchronized: {tf} ({os.path.getsize(tf):,} bytes)")

print("\nAll 3 notebooks successfully updated and harmonized!")
