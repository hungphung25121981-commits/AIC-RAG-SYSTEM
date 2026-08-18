"""
Cấu hình chung -- đọc từ file YAML trong configs/, chọn theo biến môi trường AIC_CONFIG.

CÁCH DÙNG (đổi nền tảng = đổi 1 dòng, code các file khác không cần sửa gì):

    Local (mặc định, không set gì):
        python build_dense_index.py

    Kaggle:
        import os
        os.environ["AIC_CONFIG"] = "configs/kaggle.yaml"
        import config           # phải set env var TRƯỚC dòng import này

    Hoặc set qua shell trước khi chạy:
        AIC_CONFIG=configs/kaggle.yaml python build_dense_index.py

    Test nhanh với data mẫu nhỏ:
        AIC_CONFIG=configs/sample.yaml python build_dense_index.py

Mỗi giá trị dataset.*_dir trong YAML có thể là:
  - 1 đường dẫn cụ thể, hoặc
  - chuỗi "auto" -> tự quét /kaggle/input/*/<TênThưMục> (chỉ có tác dụng trên Kaggle)
"""

import glob
import os

import yaml

# ----------------------------------------------------------------------
# 0. Load file YAML tương ứng
# ----------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("AIC_CONFIG", os.path.join(_THIS_DIR, "configs", "base.yaml"))
INTERNVL_LOCAL_PATH = "./aic2026_retrieval/internvl2_5_local"
INTERNVL_HF_PATH = "OpenGVLab/InternVL2_5-2B"

# Gán mặc định nếu các file cũ truy cập INTERNVL_MODEL_PATH
INTERNVL_MODEL_PATH = INTERNVL_LOCAL_PATH

if not os.path.isfile(CONFIG_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy file config: {CONFIG_PATH}\n"
        f"Kiểm tra lại biến môi trường AIC_CONFIG, hoặc dùng 1 trong các file có sẵn: "
        f"configs/base.yaml, configs/kaggle.yaml, configs/colab.yaml, configs/sample.yaml"
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f) or {}

print(f"[config] Đang dùng: {CONFIG_PATH}")


def _auto_find_dir(folder_name: str) -> str:
    """Quét mọi dataset đã attach trong /kaggle/input/, tìm thư mục con tên khớp."""
    if os.path.isdir("/kaggle/input"):
        candidates = sorted(glob.glob(f"/kaggle/input/*/{folder_name}"))
        if candidates:
            return candidates[0]
        for path in sorted(glob.glob("/kaggle/input/*")):
            if os.path.basename(path).lower() == folder_name.lower():
                return path
        deep = sorted(glob.glob(f"/kaggle/input/*/**/{folder_name}", recursive=True))
        if deep:
            return deep[0]
        print(f"[config][WARN] Không tự tìm thấy thư mục '{folder_name}' trong /kaggle/input/. "
              f"Sửa lại giá trị dataset trong {CONFIG_PATH}.")
    return f"./{folder_name}"


def _resolve_dir(key: str, default_folder_name: str) -> str:
    # 1) biến môi trường luôn ưu tiên cao nhất (vd AIC_KEYFRAMES_DIR),
    #    kể cả khi YAML đã có giá trị cụ thể -- để có thể override nhanh
    #    lúc debug mà không cần sửa file YAML.
    env_var = f"AIC_{key.upper()}"
    env_override = os.environ.get(env_var)
    if env_override:
        return env_override

    val = (_cfg.get("dataset") or {}).get(key)
    if val == "auto":
        return _auto_find_dir(default_folder_name)
    if val:
        return val
    return _auto_find_dir(default_folder_name)


# ----------------------------------------------------------------------
# 1. Đường dẫn dataset
# ----------------------------------------------------------------------
VIDEOS_DIR         = _resolve_dir("videos_dir", "Videos")
KEYFRAMES_DIR       = _resolve_dir("keyframes_dir", "Keyframes")
OBJECTS_DIR         = _resolve_dir("objects_dir", "Objects")
CLIPFEAT_DIR         = _resolve_dir("clipfeat_dir", "CLIPFeatures")
METADATA_DIR         = _resolve_dir("metadata_dir", "Metadata")
MAPKEYFRAMES_DIR     = _resolve_dir("mapkeyframes_dir", "map-keyframes")

# ----------------------------------------------------------------------
# 2. Nơi lưu index đã build (dense + bm25 + caption)
# ----------------------------------------------------------------------
INDEX_DIR = _cfg.get("index_dir") or os.environ.get("AIC_INDEX_DIR", "./index_data")
os.makedirs(INDEX_DIR, exist_ok=True)

FAISS_INDEX_PATH   = os.path.join(INDEX_DIR, "siglip2.faiss")
ID_MAP_PATH        = os.path.join(INDEX_DIR, "id_map.jsonl")
CAPTION_JSONL_PATH = os.path.join(INDEX_DIR, "captions.jsonl")
BM25_CORPUS_PATH   = os.path.join(INDEX_DIR, "bm25_corpus.jsonl")
BM25_PICKLE_PATH   = os.path.join(INDEX_DIR, "bm25.pkl")

# ----------------------------------------------------------------------
# 3. Model
# ----------------------------------------------------------------------
_models = _cfg.get("models") or {}
SIGLIP2_MODEL_ID   = _models.get("siglip2", "google/siglip2-so400m-patch14-384")
_internvl_raw = _models.get("internvl", "OpenGVLab/InternVL2_5-8B")
if _internvl_raw.startswith("aic2026_retrieval/") or _internvl_raw.startswith("./"):
    # Resolve relative path theo vị trí file config.py, không phụ thuộc cwd lúc chạy lệnh
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INTERNVL_MODEL_ID = os.path.join(_repo_root, _internvl_raw)
else:
    INTERNVL_MODEL_ID = _internvl_raw
TRANSLATE_MODEL_ID   = _models.get("translate", "Helsinki-NLP/opus-mt-vi-en")

DEVICE = _cfg.get("device", "cuda")
DTYPE  = _cfg.get("dtype", "bfloat16")

# ----------------------------------------------------------------------
# 4. Tham số search / fusion
# ----------------------------------------------------------------------
_search = _cfg.get("search") or {}
DENSE_TOPK   = _search.get("dense_topk", 100)
BM25_TOPK    = _search.get("bm25_topk", 100)
RRF_K        = _search.get("rrf_k", 60)
FINAL_TOPK   = _search.get("final_topk", 100)
RERANK_TOPK  = _search.get("rerank_topk", 30)

OBJECT_SCORE_THRESHOLD = _cfg.get("object_score_threshold", 0.3)
