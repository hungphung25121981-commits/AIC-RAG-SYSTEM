"""
Chạy TRƯỚC khi build index (đặc biệt trên Kaggle) để xác nhận config.py đã tự
dò đúng path của cả 5 thư mục dataset, tránh mất thời gian chạy job dài rồi
mới phát hiện path sai / rỗng.

Chạy:
    python check_config.py
"""

import os
import config

REQUIRED_DIRS = {
    "VIDEOS_DIR": config.VIDEOS_DIR,
    "KEYFRAMES_DIR": config.KEYFRAMES_DIR,
    "OBJECTS_DIR": config.OBJECTS_DIR,
    "METADATA_DIR": config.METADATA_DIR,
    "MAPKEYFRAMES_DIR": config.MAPKEYFRAMES_DIR,
}
# CLIPFeatures do BTC cấp KHÔNG được dùng trong pipeline này (pipeline tự
# re-encode bằng SigLIP2), nên chỉ cảnh báo, không chặn job.
OPTIONAL_DIRS = {
    "CLIPFEAT_DIR (không dùng trong pipeline này)": config.CLIPFEAT_DIR,
}

print("=" * 60)
print("KIỂM TRA CONFIG -- CHẠY TRƯỚC KHI BUILD INDEX")
print("=" * 60)

all_ok = True
for name, path in REQUIRED_DIRS.items():
    exists = os.path.isdir(path)
    n_items = len(os.listdir(path)) if exists else 0
    status = "OK" if exists and n_items > 0 else "LỖI"
    if status == "LỖI":
        all_ok = False
    print(f"[{status}] {name:15s} = {path}  (số item bên trong: {n_items})")

for name, path in OPTIONAL_DIRS.items():
    exists = os.path.isdir(path)
    n_items = len(os.listdir(path)) if exists else 0
    status = "OK" if exists and n_items > 0 else "BỎ QUA (optional)"
    print(f"[{status}] {name:15s} = {path}  (số item bên trong: {n_items})")

print()
if all_ok:
    print(">>> Tất cả path OK, có thể chạy build_dense_index.py tiếp.")
else:
    print(">>> CÓ PATH SAI/RỖNG -- DỪNG LẠI, ĐỪNG CHẠY JOB DÀI.")
    print("Cách sửa:")
    print("  1. Kiểm tra đã 'Add Data' đủ 5 dataset (Videos/Keyframes/Objects/")
    print("     CLIPFeatures/Metadata) trong Kaggle notebook chưa.")
    print("  2. Nếu tên dataset không trùng tên thư mục chuẩn, set thủ công")
    print("     bằng biến môi trường trước khi import config, ví dụ:")
    print('       os.environ["AIC_KEYFRAMES_DIR"] = "/kaggle/input/ten-dataset-that/Keyframes"')
    raise SystemExit(1)
