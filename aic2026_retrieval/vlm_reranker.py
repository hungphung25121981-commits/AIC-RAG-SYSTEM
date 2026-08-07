import re
import torch
from typing import List, Tuple

import config
from build_captions import load_image_tensor, load_internvl

RERANK_PROMPT_TEMPLATE = (
    "<image>\n"
    "Câu mô tả cần tìm: \"{query}\"\n"
    "Hãy đánh giá mức độ khớp giữa bức ảnh này và câu mô tả trên theo thang điểm "
    "0-10 (0 = hoàn toàn không liên quan, 10 = khớp hoàn hảo mọi chi tiết). "
    "Chỉ trả lời đúng 1 số nguyên, không giải thích thêm."
)

class VLMReranker:
    def __init__(self):
        self.model, self.tokenizer = load_internvl()
        self.generation_config = dict(max_new_tokens=8, do_sample=False)

    @torch.no_grad()
    def score(self, image_path: str, query_vi: str) -> float:
        try:
            pixel_values = load_image_tensor(image_path).to(getattr(torch, config.DTYPE)).to(config.DEVICE)
            prompt = RERANK_PROMPT_TEMPLATE.format(query=query_vi)
            response = self.model.chat(self.tokenizer, pixel_values, prompt, self.generation_config)
            match = re.search(r"\d+", response)
            return float(match.group()) if match else 0.0
        except Exception as e:
            print(f"[WARN] Lỗi rerank {image_path}: {e}")
            return 0.0

    def rerank(self, candidates: List[Tuple[int, float]], query_vi: str,
               id_map_by_int_id: dict, top_k: int = None) -> List[Tuple[int, float]]:
        top_k = top_k or getattr(config, "RERANK_TOPK", 10)
        candidates = candidates[:top_k]

        scored = []
        for int_id, _rrf_score in candidates:
            row = id_map_by_int_id[int_id]
            vlm_score = self.score(row["keyframe_path"], query_vi)
            scored.append((int_id, vlm_score))

        return sorted(scored, key=lambda x: -x[1])
