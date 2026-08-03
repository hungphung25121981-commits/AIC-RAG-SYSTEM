"""
Dịch query tiếng Việt -> tiếng Anh trước khi đưa vào SigLIP2 text encoder
(SigLIP2 base yếu tiếng Việt hơn nhiều so với tiếng Anh).

Có 2 lựa chọn:
  1) MarianMT nhẹ (Helsinki-NLP/opus-mt-vi-en) -- nhanh, đủ dùng cho câu mô tả cảnh.
  2) Dùng chính InternVL2.5 / Qwen2.5-VL (LLM backbone) để dịch -- chính xác hơn,
     đặc biệt với câu dài nhiều thành phần, nhưng chậm hơn. Bật bằng --use-llm.
"""

import config

_marian_model = None
_marian_tokenizer = None


def _load_marian():
    global _marian_model, _marian_tokenizer
    if _marian_model is None:
        from transformers import MarianMTModel, MarianTokenizer
        _marian_tokenizer = MarianTokenizer.from_pretrained(config.TRANSLATE_MODEL_ID)
        _marian_model = MarianMTModel.from_pretrained(config.TRANSLATE_MODEL_ID).to(config.DEVICE).eval()
    return _marian_model, _marian_tokenizer


def translate_vi2en(text_vi: str) -> str:
    """Dịch nhanh bằng MarianMT. Dùng cho pipeline online (latency thấp)."""
    model, tokenizer = _load_marian()
    batch = tokenizer([text_vi], return_tensors="pt", padding=True).to(config.DEVICE)
    gen = model.generate(**batch, max_new_tokens=128)
    return tokenizer.decode(gen[0], skip_special_tokens=True)


def translate_vi2en_llm(text_vi: str, vlm_model, vlm_tokenizer) -> str:
    """
    Dịch bằng chính InternVL2.5/Qwen2.5-VL đã load sẵn (dùng lại model rerank,
    khỏi tốn thêm VRAM cho model dịch riêng). Truyền model/tokenizer đã init từ
    vlm_rerank.py vào đây.
    """
    prompt = (
        f"Dịch câu sau sang tiếng Anh, chỉ trả về bản dịch, không giải thích:\n{text_vi}"
    )
    generation_config = dict(max_new_tokens=128, do_sample=False)
    response = vlm_model.chat(vlm_tokenizer, None, prompt, generation_config)
    return response.strip()
