import os
import zipfile
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(BASE_DIR, "internvl2_5_local")

# Thay link Release của bạn vào đây
URL_PART1 = "https://github.com/USER/REPO/releases/download/v1.0/intern.zip.001"
URL_PART2 = "https://github.com/USER/REPO/releases/download/v1.0/intern.zip.002"

def download_and_extract_custom_weights():
    if os.path.exists(LOCAL_DIR) and os.listdir(LOCAL_DIR):
        print(f"[INFO] Thư mục weights đã tồn tại tại {LOCAL_DIR}, bỏ qua bước tải.")
        return

    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    p1_path = os.path.join(BASE_DIR, "part1.tmp")
    p2_path = os.path.join(BASE_DIR, "part2.tmp")
    full_zip_path = os.path.join(BASE_DIR, "intern_custom.zip")

    print("==> 1. Đang tải các phần weights custom từ GitHub Release...")
    urllib.request.urlretrieve(URL_PART1, p1_path)
    urllib.request.urlretrieve(URL_PART2, p2_path)

    print("==> 2. Đang nối các file...")
    with open(full_zip_path, "wb") as outfile:
        for p in [p1_path, p2_path]:
            with open(p, "rb") as infile:
                outfile.write(infile.read())
            os.remove(p)

    print("==> 3. Đang giải nén bộ weights custom...")
    with zipfile.ZipFile(full_zip_path, "r") as zip_ref:
        zip_ref.extractall(LOCAL_DIR)

    os.remove(full_zip_path)
    print(f"==> HOÀN TẤT! Weights custom đã được giải nén vào {LOCAL_DIR}")

if __name__ == "__main__":
    download_and_extract_custom_weights()
