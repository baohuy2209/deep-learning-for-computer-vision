import json
import re
from pathlib import Path

def parse_build_script():
    script_path = Path("scratch/build_lodo_notebook.py")
    text = script_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    cells = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Match add_md(...) or add_code(...)
        m_start = re.match(r'^\s*add_(md|code)\((?:r?\"\"\"|r?\'\'\')(.*)', line)
        if m_start:
            cell_type = "markdown" if m_start.group(1) == "md" else "code"
            rest = m_start.group(2)
            curr_lines = [rest + "\n"] if rest.strip() else []
            cell_id = None
            i += 1
            while i < len(lines):
                l = lines[i]
                # Look for end pattern: """ or """, cell_id="..." followed by )
                # Note: in code cells, internal """ might appear in docstrings, but they won't end with ", cell_id=..." or be standalone at the end of cell definition
                m_end = re.search(r'\"\"\"(?:,\s*cell_id=\"([^\"]+)\")?\s*\)\s*$', l)
                if not m_end:
                    # Also check for ''', cell_id="..."
                    m_end = re.search(r'\'\'\'(?:,\s*cell_id=\"([^\"]+)\")?\s*\)\s*$', l)

                if m_end:
                    before_quote = l[:m_end.start()]
                    if before_quote.strip():
                        curr_lines.append(before_quote + "\n")
                    cell_id = m_end.group(1)
                    cells.append({
                        "type": cell_type,
                        "id": cell_id,
                        "source": curr_lines
                    })
                    break
                else:
                    curr_lines.append(l)
                i += 1
        i += 1

    return cells

def assemble_notebook(cells, output_path: Path):
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "A100",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    for idx, c in enumerate(cells):
        raw_text = "".join(c["source"]).strip()
        lines = [line + "\n" for line in raw_text.splitlines()]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")

        cell = {
            "cell_type": c["type"],
            "metadata": {},
            "source": lines
        }
        if c["id"]:
            cell["id"] = c["id"]

        if c["type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

        nb["cells"].append(cell)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Successfully assembled {len(nb['cells'])} cells to {output_path}")

if __name__ == "__main__":
    cells = parse_build_script()
    print(f"Extracted {len(cells)} cells.")
    for idx, c in enumerate(cells):
        first_line = c["source"][0].strip()[:50] if c["source"] else "EMPTY"
        print(f"Cell {idx:02d}: {c['type']:8s} | id: {c['id']} | lines: {len(c['source'])} | {first_line}")
    
    out_nb = Path("notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb")
    assemble_notebook(cells, out_nb)

