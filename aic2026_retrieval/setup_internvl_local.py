import os
import re
from huggingface_hub import snapshot_download

# Thư mục lưu weight local
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "aic2026_retrieval", "internvl2_5_local")
MODEL_ID = "OpenGVLab/InternVL2_5-2B"

print(f"==> 1. Tải model {MODEL_ID} về {LOCAL_DIR}...")
snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR, local_dir_use_symlinks=False)

print("==> 2. Patch file modeling_internvl_chat.py...")
target_file = os.path.join(LOCAL_DIR, "modeling_internvl_chat.py")

if os.path.exists(target_file):
    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix FlashAttention trên Kaggle T4 GPU
    content = content.replace("from .flash_attention import FlashAttention", "# from .flash_attention import FlashAttention")
    
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(content)
    print("==> PATCH THÀNH CÔNG!")
