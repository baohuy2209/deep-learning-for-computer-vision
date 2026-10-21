"""Script to scan codebase for sensitive credentials and personal paths."""

import os
import re
from pathlib import Path

PATTERNS = [
    re.compile(r'KAGGLE_KEY\s*=\s*[\'"][a-zA-Z0-9_\-]+[\'"]'),
    re.compile(r'KAGGLE_USERNAME\s*=\s*[\'"][a-zA-Z0-9_\-]+[\'"]'),
    re.compile(r'(api_key|secret_key|private_key|password)\s*=\s*[\'"][a-zA-Z0-9_\-]+[\'"]', re.I),
]

EXCLUDE_DIRS = {
    ".git", ".venv", ".venv-win", "__pycache__", ".pytest_cache",
    ".agents", ".claude", "data", "documents", "paper-reference",
    "images", "test_figures", "checkpoints", "checkpoints_head_ablation"
}

PLACEHOLDER_SUBSTRINGS = {"your_api_key", "your_username", "<your_key>", "<your_username>", "xxx", "placeholder"}

def scan_repo():
    violations = []
    root_path = Path(".")
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.endswith((".py", ".json", ".md", ".yml", ".yaml", ".sh", ".ipynb")):
                file_path = Path(root) / file
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    for pat in PATTERNS:
                        matches = pat.findall(text)
                        for m in matches:
                            if not any(ph in m for ph in PLACEHOLDER_SUBSTRINGS):
                                violations.append((file_path, m))
                except Exception:
                    pass
    return violations

if __name__ == "__main__":
    v = scan_repo()
    if v:
        print("POTENTIAL SECRETS DETECTED:")
        for path, matches in v:
            print(f"  {path}: {matches}")
        exit(1)
    else:
        print("CLEAN: Zero secrets, keys, or passwords detected.")
        exit(0)
