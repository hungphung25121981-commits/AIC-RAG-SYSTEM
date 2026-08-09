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
    "CLIPFEAT_DIR": config.CLIPFEAT_DIR,
}

def get_dir_item_count(path: str) -> tuple[bool, int]:
    """Kiểm tra sự tồn tại và đếm số item an toàn."""
    if not isinstance(path, str) or not os.path.isdir(path):
        return False, 0
    try:
        return True, len(os.listdir(path))
    except Exception:
        return True, 0


print("=" * 70)
print("KIỂM TRA CONFIG -- CHẠY TRƯỚC KHI BUILD INDEX")
print("=" * 70)

all_ok = True

# 1. Kiểm tra các thư mục bắt buộc
for name, path in REQUIRED_DIRS.items():
    exists, n_items = get_dir_item_count(path)
    status = "OK" if exists and n_items > 0 else "LỖI"
    if status == "LỖI":
        all_ok = False
    print(f"[{status:^4s}] {name:<20s} = {str(path):<35s} (số item: {n_items})")

# 2. Kiểm tra các thư mục tùy chọn
for name, path in OPTIONAL_DIRS.items():
    exists, n_items = get_dir_item_count(path)
    status = "OK" if exists and n_items > 0 else "BỎ QUA"
    print(f"[{status:^4s}] {name:<20s} = {str(path):<35s} (số item: {n_items}) [Optional]")

print("-" * 70)

# 3. Quét thử keyframe thực tế nếu các path đều OK
if all_ok:
    print(">>> Tất cả path OK, đang thử quét thật keyframe để xác nhận cấu trúc thư mục...")
    try:
        import utils
        items = list(utils.iter_all_keyframes())
        if len(items) == 0:
            raise ValueError("Quét thành công nhưng tìm thấy 0 keyframe!")
            
        print(f">>> SUCCESS: Quét thành công {len(items)} keyframe. Mẫu dữ liệu đầu tiên:")
        print("   ", items[0])
        print(">>> Mọi thứ chuẩn bị hoàn tất. Bạn có thể bắt đầu chạy 'python build_dense_index.py'")
    except Exception as e:
        all_ok = False
        print(f">>> [LỖI] Quét keyframe thất bại: {e}")

if not all_ok:
    print("\n>>> CÓ PATH SAI/RỖNG -- DỪNG LẠI, ĐỪNG CHẠY JOB DÀI!")
    print("Cách sửa:")
    print("  1. Kiểm tra đã 'Add Data' đủ các bộ Dataset trên Kaggle chưa.")
    print("  2. Nếu tên thư mục không tự động khớp, hãy đặt biến môi trường thủ công:")
    print('     os.environ["AIC_KEYFRAMES_DIR"] = "/kaggle/input/ten-dataset/Keyframes"')
    raise SystemExit(1)
