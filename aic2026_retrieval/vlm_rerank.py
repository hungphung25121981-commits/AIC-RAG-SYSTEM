"""
Rerank tầng cuối: đưa top-K ảnh sau RRF fusion + query gốc (tiếng Việt) vào
InternVL2.5 để chấm điểm relevance 0-10, sort lại theo điểm này.

Bước này xử lý tốt các query composite (nhiều điều kiện cùng lúc: màu áo +
bối cảnh + hành động...) mà similarity vector đơn thuần dễ bỏ sót.
"""

import os
import re
import torch
from typing import List, Tuple
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import config
from build_captions import load_image_tensor, load_internvl  # tái dùng preprocessing + loader

# Tự động tải weights nếu thiếu
try:
    from download_weights import download_and_extract_custom_weights
except ImportError:
    download_and_extract_custom_weights = None


RERANK_PROMPT_TEMPLATE = (
    "<image>\n"
    "Câu mô tả cần tìm: \"{query}\"\n"
    "Hãy đánh giá mức độ khớp giữa bức ảnh này và câu mô tả trên theo thang điểm "
    "0-10 (0 = hoàn toàn không liên quan, 10 = khớp hoàn hảo mọi chi tiết). "
    "Chỉ trả lời đúng 1 số nguyên, không giải thích thêm."
)

import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import config


def answer_question(keyframe_path: str, question: str) -> str:
    """Trả lời câu hỏi VQA dựa trên 1 keyframe bằng model Hugging Face (OpenGVLab/InternVL2_5-2B)."""
    if not os.path.exists(keyframe_path):
        return f"Không tìm thấy file keyframe tại: {keyframe_path}"

    # Lấy repo ID từ config hoặc ưu tiên repo HF chính thức
    model_id = getattr(config, "INTERNVL_HF_PATH", "OpenGVLab/InternVL2_5-2B")

    # Kiểm tra an toàn: Nếu dính đường dẫn local thì tự động đổi về HF repo ID
    if model_id.startswith(".") or model_id.startswith("/") or "local" in model_id:
        model_id = "OpenGVLab/InternVL2_5-2B"

    print(f"[VQA] Loading model from Hugging Face: {model_id}...")

    dtype = getattr(config, "DTYPE", torch.bfloat16) if hasattr(config, "DTYPE") else torch.bfloat16
    device = getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = (
            AutoModel.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            .eval()
            .to(device)
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        image = Image.open(keyframe_path).convert("RGB")

        prompt = f"<image>\nQuestion: {question}\nAnswer in Vietnamese concise and accurate:"

        try:
            transform = model.build_transform(input_size=448)
            pixel_values = transform(image).unsqueeze(0).to(dtype).to(device)
        except AttributeError:
            pixel_values = model.extract_feature(image).to(dtype).to(device)

        generation_config = dict(max_new_tokens=128, do_sample=False)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config)

        # Giải phóng VRAM ngay sau khi trả lời xong
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()

    except Exception as e:
        return f"[ERROR VQA Runtime]: {e}"
class VLMReranker:
    def __init__(self):
        # 1. Kiểm tra nếu chưa có weights thì tự động tải
        model_path = os.path.abspath(config.INTERNVL_MODEL_PATH)
        if not os.path.exists(model_path) or not os.listdir(model_path):
            print(f"[WARN] Chưa tìm thấy Local Weights tại: {model_path}")
            if download_and_extract_custom_weights:
                print("[INFO] Kích hoạt tự động tải Weights...")
                download_and_extract_custom_weights()

        # 2. Tải model bằng hàm gốc của bạn
        self.model, self.tokenizer = load_internvl()
        self.generation_config = dict(max_new_tokens=8, do_sample=False)
        self.model_id = getattr(config, "INTERNVL_LOCAL_PATH", "./aic2026_retrieval/internvl2_5_local")
    @torch.no_grad()
    def score(self, image_path: str, query_vi: str) -> float:
        try:
            pixel_values = load_image_tensor(image_path).to(getattr(torch, config.DTYPE)).to(config.DEVICE)
            prompt = RERANK_PROMPT_TEMPLATE.format(query=query_vi)
            response = self.model.chat(self.tokenizer, pixel_values, prompt, self.generation_config)
            
            match = re.search(r"\d+", response)
            if match:
                val = float(match.group())
                return min(max(val, 0.0), 10.0)  # Giữ trong khoảng 0-10
            return 0.0
        except Exception as e:
            print(f"[WARN] Lỗi rerank {image_path}: {e}")
            return 0.0

    def rerank(self, candidates: List[Tuple[int, float]], query_vi: str,
               id_map_by_int_id: dict, top_k: int = None) -> List[Tuple[int, float]]:
        """
        candidates: list (int_id, rrf_score) đã lấy top-N từ hybrid_search.
        id_map_by_int_id: dict int_id -> row id_map (có keyframe_path).
        Trả về list (int_id, vlm_score) sort giảm dần.
        """
        top_k = top_k or config.RERANK_TOPK
        candidates = candidates[:top_k]

        scored = []
        for int_id, rrf_score in candidates:
            row = id_map_by_int_id[int_id]
            img_path = row.get("keyframe_path") or row.get("path")


        except Exception as e:
            return f"[ERROR VQA Runtime]: {e}"
