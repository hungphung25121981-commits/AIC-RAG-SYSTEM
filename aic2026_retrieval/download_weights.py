iimport os
import subprocess
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(BASE_DIR, "internvl2_5_local")

# Dán 2 link copy từ GitHub Release vào đây
URL_PART1 = "https://github.com/hungphung25121981-commits/AIC-RAG-SYSTEM/releases/download/MODEL/internvl2_5_local.part1.rar"
URL_PART2 = "https://github.com/hungphung25121981-commits/AIC-RAG-SYSTEM/releases/download/MODEL/internvl2_5_local.part2.rar"

def download_and_extract_rar():
    if os.path.exists(LOCAL_DIR) and os.listdir(LOCAL_DIR):
        print(f"[INFO] Weights đã có sẵn tại {LOCAL_DIR}, bỏ qua bước tải.")
        return

    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    file_p1 = os.path.join(BASE_DIR, "internvl2_5_local.part1.rar")
    file_p2 = os.path.join(BASE_DIR, "internvl2_5_local.part2.rar")

    print("==> 1. Đang tải Part 1 (1.76 GB)...")
    urllib.request.urlretrieve(URL_PART1, file_p1)
    
    print("==> 2. Đang tải Part 2 (1.33 GB)...")
    urllib.request.urlretrieve(URL_PART2, file_p2)

    print("==> 3. Cài đặt unrar và giải nén RAR Multipart...")
    # Cài đặt công cụ unrar nếu chạy trên Kaggle/Colab/Ubuntu
    subprocess.run(["apt-get", "update", "-y"], stdout=subprocess.DEVNULL)
    subprocess.run(["apt-get", "install", "-y", "unrar"], stdout=subprocess.DEVNULL)

    # Giải nén file part1 (unrar sẽ tự động liên kết kéo part2 theo)
    cmd = f"unrar x -o+ {file_p1} {BASE_DIR}"
    subprocess.run(cmd, shell=True, check=True)

    # Dọn dẹp 2 file rar sau khi giải nén xong để đỡ tốn dung lượng ổ cứng
    if os.path.exists(file_p1): os.remove(file_p1)
    if os.path.exists(file_p2): os.remove(file_p2)

    print(f"==> HOÀN TẤT! Đã giải nén bộ weights custom vào {LOCAL_DIR}")

if __name__ == "__main__":
    download_and_extract_rar()
