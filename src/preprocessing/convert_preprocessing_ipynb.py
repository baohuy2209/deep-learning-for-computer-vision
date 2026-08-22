import json
import re

def convert_py_to_ipynb(py_path: str, ipynb_path: str):
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'^# %%\s*', content, flags=re.MULTILINE)
    cells = []

    for block in blocks:
        if not block.strip():
            continue
        
        lines = block.splitlines(keepends=True)
        first_line = lines[0].strip()
        
        if first_line == '[markdown]':
            cell_lines = lines[1:]
            cleaned_lines = []
            for l in cell_lines:
                if l.startswith('# '):
                    cleaned_lines.append(l[2:])
                elif l.startswith('#'):
                    cleaned_lines.append(l[1:])
                else:
                    cleaned_lines.append(l)
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": cleaned_lines
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines
            })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print(f"[SUCCESS] Converted {py_path} -> {ipynb_path} ({len(cells)} cells)")

if __name__ == "__main__":
    convert_py_to_ipynb(
        "notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py",
        "notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing.ipynb"
    )
