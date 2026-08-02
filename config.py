"""
Cấu hình chung cho pipeline retrieval AIC 2026.
Chỉnh lại các đường dẫn DATASET_ROOT / INDEX_DIR cho khớp với máy của bạn.
"""

import os

# ----------------------------------------------------------------------
# 1. Đường dẫn dataset (theo cấu trúc BTC cung cấp)
# ----------------------------------------------------------------------
DATASET_ROOT = os.environ.get("AIC_DATASET_ROOT", "/data/aic2026_batch1")

VIDEOS_DIR    = os.path.join(DATASET_ROOT, "Videos")
KEYFRAMES_DIR = os.path.join(DATASET_ROOT, "Keyframes")
OBJECTS_DIR   = os.path.join(DATASET_ROOT, "Objects")
CLIPFEAT_DIR  = os.path.join(DATASET_ROOT, "CLIPFeatures")   # .npy do BTC cấp (clip-ViT-B-32)
METADATA_DIR  = os.path.join(DATASET_ROOT, "Metadata")

# ----------------------------------------------------------------------
# 2. Nơi lưu index đã build (dense + bm25 + caption)
# ----------------------------------------------------------------------
INDEX_DIR = os.environ.get("AIC_INDEX_DIR", "./index_data")
os.makedirs(INDEX_DIR, exist_ok=True)

FAISS_INDEX_PATH   = os.path.join(INDEX_DIR, "siglip2.faiss")
ID_MAP_PATH        = os.path.join(INDEX_DIR, "id_map.jsonl")         # int_id -> (video_id, frame_id, path)
CAPTION_JSONL_PATH = os.path.join(INDEX_DIR, "captions.jsonl")       # int_id -> caption text (VI+EN)
BM25_CORPUS_PATH   = os.path.join(INDEX_DIR, "bm25_corpus.jsonl")    # int_id -> tokenized doc
BM25_PICKLE_PATH   = os.path.join(INDEX_DIR, "bm25.pkl")

# ----------------------------------------------------------------------
# 3. Model
# ----------------------------------------------------------------------
SIGLIP2_MODEL_ID = "google/siglip2-so400m-patch14-384"
# InternVL2.5 dùng để caption offline + rerank online (đổi sang bản 8B/26B nếu đủ VRAM)
INTERNVL_MODEL_ID = "OpenGVLab/InternVL2_5-8B"
# Model dịch/rerank nhẹ nếu không muốn tải InternVL full (fallback)
TRANSLATE_MODEL_ID = "Helsinki-NLP/opus-mt-vi-en"

DEVICE = "cuda"          # "cuda" | "cpu"
DTYPE  = "bfloat16"       # bfloat16 khuyến nghị cho GPU Ampere+

# ----------------------------------------------------------------------
# 4. Tham số search / fusion
# ----------------------------------------------------------------------
DENSE_TOPK   = 100     # số ứng viên lấy từ nhánh dense trước khi fusion
BM25_TOPK    = 100     # số ứng viên lấy từ nhánh BM25 trước khi fusion
RRF_K        = 60      # hằng số k trong công thức RRF (mặc định literature dùng 60)
FINAL_TOPK   = 100     # số kết quả cuối cùng trả về (khớp giới hạn 100 câu trả lời của BTC)
RERANK_TOPK  = 30      # chỉ đưa top N sau RRF vào VLM rerank (tránh tốn compute)

OBJECT_SCORE_THRESHOLD = 0.3   # ngưỡng confidence lọc object trong Objects/*.json
