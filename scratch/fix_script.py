with open("scratch/update_lodo_clean.py", "r", encoding="utf-8") as f:
    content = f.read()

target = '        print(f"  [PENDING] Fold {f_idx} ({h_key}): {e}")\n\n\n\n# ------------------------------------------------------------------------------'
replacement = """        print(f"  [PENDING] Fold {f_idx} ({h_key}): {e}")

if audit_records:
    print()
    print("[SUCCESS] Pre-flight audit confirmed: Zero Patient Leakage across all verified folds!")
\"\"\"

# ------------------------------------------------------------------------------"""

assert target in content, "target not found"
content = content.replace(target, replacement, 1)

with open("scratch/update_lodo_clean.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed update_lodo_clean.py successfully!")
