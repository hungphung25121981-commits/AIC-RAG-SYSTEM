"""
Sinh caption cho toàn bộ keyframe bằng InternVL2.5 (chạy offline, tốn thời gian nhất
trong pipeline). Đã tối ưu tốc độ 3 lớp:
  1) BATCH inference (model.batch_chat) thay vì chat từng ảnh 1 -- đòn bẩy lớn nhất,
     tận dụng đúng GPU thay vì để GPU rảnh giữa các lần gọi.
  2) Quantize 4-bit (--quantize4bit) -- giảm VRAM, tăng tốc trên GPU nhỏ (T4).
  3) Model nhẹ hơn (mặc định 2B qua config, đổi trong configs/*.yaml).

Chạy:
    python build_captions.py --shard 0 --num-shards 1 --dedup-map \
        --batch-size 16 --quantize4bit --max-new-tokens 32

Output:
    config.CAPTION_JSONL_PATH (append theo shard, mỗi dòng: {int_id, video_id, frame_id, caption})
"""

import argparse
import os
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

import config
import utils

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image_tensor(image_path, input_size=448):
    """Preprocess đơn giản (single-tile, không dùng dynamic tiling để giữ tốc độ
    cho việc caption hàng loạt; nếu cần độ chi tiết cao hơn có thể bật lại
    dynamic-tiling như code gốc InternVL)."""
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    return transform(image).unsqueeze(0)


def load_internvl(quantize4bit: bool = False):
    from transformers import AutoModel, AutoTokenizer

    dtype = getattr(torch, config.DTYPE)
    load_kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if quantize4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("[build_captions] Bật quantize 4-bit (bitsandbytes, nf4).")

    model = AutoModel.from_pretrained(config.INTERNVL_MODEL_ID, **load_kwargs).eval()
    if not quantize4bit:
        # khi quantize, model đã tự được đặt lên device qua accelerate; không .to() thủ công nữa
        model = model.to(config.DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(
        config.INTERNVL_MODEL_ID, trust_remote_code=True, use_fast=False
    )
    return model, tokenizer


CAPTION_PROMPT = (
    "<image>\n"
    "Hãy mô tả chi tiết bức ảnh này bằng tiếng Việt: có những ai, đang làm gì, "
    "trang phục màu gì, bối cảnh/địa điểm ra sao, có vật thể đáng chú ý nào. "
    "Trả lời trong 1-2 câu ngắn gọn, súc tích, không lặp từ."
)


def caption_batch(model, tokenizer, items_batch, generation_config, dtype):
    """
    Caption 1 batch ảnh cùng lúc bằng model.batch_chat() (InternVL2/2.5 hỗ trợ sẵn),
    nhanh hơn nhiều so với gọi model.chat() từng ảnh vì tận dụng đúng GPU batching.
    Fallback về chat() từng ảnh nếu model không có batch_chat (một số bản cũ).
    """
    pixel_values_list = []
    num_patches_list = []
    for it in items_batch:
        pv = load_image_tensor(it["keyframe_path"]).to(dtype).to(config.DEVICE)
        pixel_values_list.append(pv)
        num_patches_list.append(pv.shape[0])  # =1 vì dùng single-tile

    pixel_values = torch.cat(pixel_values_list, dim=0)
    questions = [CAPTION_PROMPT] * len(items_batch)

    if hasattr(model, "batch_chat"):
        responses = model.batch_chat(
            tokenizer, pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config,
        )
    else:
        # fallback: model đời cũ không có batch_chat -- caption từng ảnh 1
        responses = []
        offset = 0
        for n in num_patches_list:
            pv = pixel_values[offset:offset + n]
            responses.append(model.chat(tokenizer, pv, CAPTION_PROMPT, generation_config))
            offset += n

    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dedup-map", action="store_true",
                         help="Chỉ caption các frame đại diện trong dedup_map.jsonl "
                              "(chạy dedup_keyframes.py trước). Giảm mạnh khối lượng caption.")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                         help="Giảm xuống (vd 24-32) để tăng tốc, caption chỉ cần 1-2 câu ngắn.")
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Số ảnh caption cùng lúc mỗi lần gọi model. Tăng nếu còn dư VRAM "
                              "(quan sát !nvidia-smi), giảm nếu bị OOM.")
    parser.add_argument("--quantize4bit", action="store_true",
                         help="Load model ở dạng 4-bit (bitsandbytes) -- giảm VRAM, tăng tốc "
                              "trên GPU nhỏ như T4. Cần: pip install bitsandbytes")
    args = parser.parse_args()

    items = list(utils.read_jsonl(config.ID_MAP_PATH))

    if args.dedup_map:
        dedup_path = os.path.join(config.INDEX_DIR, "dedup_map.jsonl")
        assert os.path.isfile(dedup_path), \
            "Chưa có dedup_map.jsonl -- chạy 'python dedup_keyframes.py' trước."
        dedup_rows = list(utils.read_jsonl(dedup_path))
        representative_ids = {r["int_id"] for r in dedup_rows if r["int_id"] == r["representative_int_id"]}
        items = [it for it in items if it["int_id"] in representative_ids]
        print(f"Đang dùng dedup_map: chỉ caption {len(items)} frame đại diện "
              f"(thay vì {len(dedup_rows)} frame gốc).")

    items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] Số keyframe cần caption: {len(items)}")

    out_path = config.CAPTION_JSONL_PATH + f".shard{args.shard}"
    done_ids = set()
    if os.path.isfile(out_path):
        done_ids = {row["int_id"] for row in utils.read_jsonl(out_path)}
        print(f"Đã caption sẵn {len(done_ids)} ảnh, sẽ resume tiếp...")

    items = [it for it in items if it["int_id"] not in done_ids]
    print(f"Còn lại cần caption: {len(items)} (batch_size={args.batch_size})")

    print(f"Đang load {config.INTERNVL_MODEL_ID}...")
    model, tokenizer = load_internvl(quantize4bit=args.quantize4bit)
    dtype = getattr(torch, config.DTYPE)
    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    for i in tqdm(range(0, len(items), args.batch_size), desc=f"Captioning (shard {args.shard})"):
        batch = items[i:i + args.batch_size]
        try:
            captions = caption_batch(model, tokenizer, batch, generation_config, dtype)
        except Exception as e:
            print(f"[WARN] Lỗi batch tại index {i}: {e} -- fallback caption rỗng cho batch này")
            captions = [""] * len(batch)

        for it, caption in zip(batch, captions):
            utils.append_jsonl(out_path, {
                "int_id": it["int_id"],
                "video_id": it["video_id"],
                "frame_id": it["frame_id"],
                "caption": caption,
            })

    print(f"Hoàn tất shard {args.shard} -> {out_path}")
    print("Sau khi tất cả shard chạy xong, merge bằng: "
          f"cat {config.CAPTION_JSONL_PATH}.shard* > {config.CAPTION_JSONL_PATH}")


if __name__ == "__main__":
    main()
