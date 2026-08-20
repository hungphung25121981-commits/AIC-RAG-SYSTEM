"""
Rerank tầng cuối: đưa top-K ảnh sau RRF fusion + query gốc (tiếng Việt) vào
InternVL2.5 để chấm điểm relevance 0-10, sort lại theo điểm này.

Bước này xử lý tốt các query composite (nhiều điều kiện cùng lúc: màu áo +
bối cảnh + hành động...) mà similarity vector đơn thuần dễ bỏ sót.
"""

import os
import re
from typing import List, Tuple
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoTokenizer
import config
from build_captions import load_image_tensor, load_internvl  # tái dùng preprocessing + loader
from transformers import GenerationMixin, GenerationConfig

# Tự động tải weights nếu thiếu
try:
    from download_weights import download_and_extract_custom_weights
except ImportError:
    download_and_extract_custom_weights = None


RERANK_PROMPT_TEMPLATE = (
    "<image>\n"
    'Câu mô tả cần tìm: "{query}"\n'
    "Hãy đánh giá mức độ khớp giữa bức ảnh này và câu mô tả trên theo thang điểm "
    "0-10 (0 = hoàn toàn không liên quan, 10 = khớp hoàn hảo mọi chi tiết). "
    "Chỉ trả lời đúng 1 số nguyên, không giải thích thêm."
)
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_path, input_size=448, max_num=12):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
def answer_question(keyframe_path: str, question: str) -> str:
    if not os.path.exists(keyframe_path):
        return f"Không tìm thấy file keyframe tại: {keyframe_path}"

    model_id = getattr(config, "INTERNVL_HF_PATH", "OpenGVLab/InternVL2_5-2B")
    if model_id.startswith(".") or model_id.startswith("/") or "local" in model_id:
        model_id = "OpenGVLab/InternVL2_5-2B"

    print(f"[VQA] Loading model from Hugging Face: {model_id}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    try:
        # Bắt buộc tắt low_cpu_mem_usage và không dùng device_map để tránh đẩy tensor vào 'meta'
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            device_map=None
        )

        # Chuyển model sang GPU thủ công
        model = model.to(device).eval()
        from transformers import GenerationMixin
        lm = model.language_model
        if not hasattr(lm, "generate"):
            lm.__class__ = type(
                lm.__class__.__name__,
                (lm.__class__, GenerationMixin),
                {}
            )
        pixel_values = load_image(keyframe_path, input_size=448, max_num=12)
        pixel_values = pixel_values.to(dtype=dtype, device=device)

        prompt = f"<image>\nQuestion: {question}\nAnswer in Vietnamese concise and accurate:"
        generation_config = dict(max_new_tokens=128, do_sample=False)
        with torch.no_grad():
            response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config)
        # XOÁ đoạn with torch.no_grad(): response, _ = model.chat(...) bị lặp lại ngay dưới
        # XOÁ đoạn with torch.no_grad(): response, _ = model.chat(...) bị lặp lại ngay dưới

        # Dọn dẹp VRAM
        del model
        del pixel_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()

    except Exception as e:
        import traceback
        traceback.print_exc()   # <-- in đầy đủ stack trace ra console
        return f"[ERROR VQA Runtime]: {e}"

def _patch_generation(model):
    """Fix: NoneType has no attribute '_from_model_config'.
    InternVL custom code monkey-patch GenerationMixin vào language_model
    nhưng không set generation_config -> generate() gọi None._from_model_config -> lỗi."""
    lm = getattr(model, "language_model", model)

    if not hasattr(lm, "generate"):
        lm.__class__ = type(lm.__class__.__name__, (lm.__class__, GenerationMixin), {})

    if getattr(lm, "generation_config", None) is None:
        lm.generation_config = GenerationConfig.from_model_config(lm.config)

    if getattr(model, "generation_config", None) is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)

    return model
class VLMReranker:
    def __init__(self):
        # 1. Kiểm tra nếu chưa có weights thì tự động tải
        model_path = getattr(
            config,
            "INTERNVL_MODEL_PATH",
            getattr(config, "INTERNVL_LOCAL_PATH", "./aic2026_retrieval/internvl2_5_local"),
        )
        model_path_abs = os.path.abspath(model_path)

        if not os.path.exists(model_path_abs) or not os.listdir(model_path_abs):
            print(f"[WARN] Chưa tìm thấy Local Weights tại: {model_path_abs}")
            if download_and_extract_custom_weights:
                print("[INFO] Kích hoạt tự động tải Weights...")
                download_and_extract_custom_weights()

        # 2. Tải model bằng hàm load_internvl
                # 2. Tải model bằng hàm load_internvl
        self.model, self.tokenizer = load_internvl()
        self.model = _patch_generation(self.model)   # <-- FIX, phòng khi load_internvl() cũng bị thiếu generation_config
        self.generation_config = dict(max_new_tokens=8, do_sample=False)
        self.model_id = model_path

    @torch.no_grad()
    def score(self, image_path: str, query_vi: str) -> float:
        try:
            if not os.path.exists(image_path):
                return 0.0

            dtype = getattr(config, "DTYPE", torch.bfloat16) if hasattr(config, "DTYPE") else torch.bfloat16
            device = getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

            pixel_values = load_image_tensor(image_path).to(dtype).to(device)
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

    def rerank(
        self,
        candidates: List[Tuple[int, float]],
        query_vi: str,
        id_map_by_int_id: dict,
        top_k: int = None,
    ) -> List[Tuple[int, float]]:
        """candidates: list (int_id, rrf_score) đã lấy top-N từ hybrid_search.

        id_map_by_int_id: dict int_id -> row id_map (có keyframe_path hoặc path). Trả về list (int_id, vlm_score) sort
        giảm dần.
        """
        top_k = top_k or getattr(config, "RERANK_TOPK", 10)
        candidates = candidates[:top_k]

        scored = []
        for int_id, rrf_score in candidates:
            row = id_map_by_int_id[int_id]
            img_path = row.get("keyframe_path") or row.get("path") or os.path.join(
                getattr(config, "KEYFRAMES_DIR", ""), row.get("video_id", ""), f"{row.get('frame_id', 0):06d}.jpg"
            )

            vlm_score = self.score(img_path, query_vi)
            scored.append((int_id, vlm_score))

        # Sắp xếp danh sách dựa trên VLM score giảm dần
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
