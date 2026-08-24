"""
Chạy script này TRONG cùng venv/kernel đang bị lỗi (trước khi import albumentations)
để tìm phần tử bất thường trong sys.path hoặc file .pth gây lỗi
"TypeError: unhashable type: 'list'".
"""

import sys
import os
import site

print("=" * 60)
print("1) Kiểm tra sys.path — tìm phần tử không phải string")
print("=" * 60)
for i, p in enumerate(sys.path):
    if not isinstance(p, str):
        print(f"  [BẤT THƯỜNG] sys.path[{i}] = {p!r} (type={type(p)})")
    else:
        print(f"  [{i}] {p}")

print()
print("=" * 60)
print("2) Kiểm tra biến môi trường PYTHONPATH")
print("=" * 60)
pp = os.environ.get("PYTHONPATH")
print(f"  PYTHONPATH = {pp!r}")
if pp and ("[" in pp or "]" in pp):
    print(
        "  [BẤT THƯỜNG] PYTHONPATH chứa ký tự '[' hoặc ']' -> có thể bị set nhầm thành literal list"
    )

print()
print("=" * 60)
print("3) Quét các file .pth trong site-packages (thường là thủ phạm)")
print("=" * 60)
site_dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
try:
    site_dirs.append(site.getusersitepackages())
except Exception:
    pass

for sd in set(site_dirs):
    if not os.path.isdir(sd):
        continue
    for fname in os.listdir(sd):
        if fname.endswith(".pth"):
            fpath = os.path.join(sd, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if "[" in content or "import" in content:
                    print(f"  [KIỂM TRA] {fpath}")
                    for line in content.splitlines():
                        if line.strip() and not line.startswith("#"):
                            print(f"      {line}")
            except Exception as e:
                print(f"  Không đọc được {fpath}: {e}")

print()
print("=" * 60)
print("4) Thử gọi importlib.metadata.entry_points() trực tiếp để tái hiện lỗi")
print("=" * 60)
try:
    import importlib.metadata as im

    list(im.distributions())
    print("  OK - không lỗi khi quét distributions()")
except TypeError as e:
    print(f"  [TÁI HIỆN LỖI] {e}")
    import traceback

    traceback.print_exc()
