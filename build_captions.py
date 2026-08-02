"""
Sinh caption cho toàn bộ keyframe bằng InternVL2.5 (chạy offline, tốn thời gian nhất
trong pipeline -- nên chạy song song nhiều process/GPU nếu có, chia theo video_id).

Chạy:
    python build_captions.py --shard 0 --num-shards 4    # chạy 4 tiến trình song song

Output:
    config.CAPTION_JSONL_PATH (append theo shard, mỗi dòng: {int_id, video_id, frame_id, caption})

Lưu ý: script này cần config.ID_MAP_PATH đã được tạo bởi build_dense_index.py trước đó
(dùng chung danh sách keyframe + int_id để đồng bộ giữa các nhánh index).
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


def load_internvl():
    from transformers import AutoModel, AutoTokenizer

    dtype = getattr(torch, config.DTYPE)
    model = AutoModel.from_pretrained(
        config.INTERNVL_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().to(config.DEVICE)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    items = list(utils.read_jsonl(config.ID_MAP_PATH))
    items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] Số keyframe cần caption: {len(items)}")

    out_path = config.CAPTION_JSONL_PATH + f".shard{args.shard}"
    done_ids = set()
    if os.path.isfile(out_path):
        done_ids = {row["int_id"] for row in utils.read_jsonl(out_path)}
        print(f"Đã caption sẵn {len(done_ids)} ảnh, sẽ resume tiếp...")

    print("Đang load InternVL2.5...")
    model, tokenizer = load_internvl()
    generation_config = dict(max_new_tokens=128, do_sample=False)

    for it in tqdm(items, desc=f"Captioning (shard {args.shard})"):
        if it["int_id"] in done_ids:
            continue
        try:
            pixel_values = load_image_tensor(it["keyframe_path"]).to(getattr(torch, config.DTYPE)).to(config.DEVICE)
            caption = model.chat(tokenizer, pixel_values, CAPTION_PROMPT, generation_config)
        except Exception as e:
            print(f"[WARN] Lỗi caption {it['keyframe_path']}: {e}")
            caption = ""

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
