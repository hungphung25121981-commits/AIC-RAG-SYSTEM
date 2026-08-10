import os
import zipfile
import urllib.request

# Đường dẫn đính kèm trên GitHub Release của bạn
RELEASE_URL = "https://github.com/USERNAME/AIC-RAG-SYSTEM/releases/download/v1.0/internvl2_5_local.zip"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(BASE_DIR, "aic2026_retrieval", "internvl2_5_local")
ZIP_PATH = os.path.join(BASE_DIR, "weights.zip")

if not os.path.exists(LOCAL_DIR):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"==> Đang tải weights từ GitHub ")
    urllib.request.urlretrieve(RELEASE_URL, ZIP_PATH)
    
    print("==> Giải nén weights...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(LOCAL_DIR)
        
    os.remove(ZIP_PATH) # Xóa file zip sau khi giải nén
    print("==> Tải và giải nén weights từ GitHub hoàn tất!")
