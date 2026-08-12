"""
Sinh caption cho toàn bộ keyframe bằng InternVL2.5 (đã tối ưu DataLoader & Batching).
vd: python build_captions.py --model-path "OpenGVLab/InternVL2_5-2B" --batch-size 64 --quantize4bit --max-new-tokens 32
"""

import argparse
import os
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config
import utils
try:
    from download_weights import download_and_extract_custom_weights
except ImportError:
    download_and_extract_custom_weights = None
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image_tensor(image_path, input_size=448, device=None):
    """Load một ảnh từ path và chuyển đổi thành Tensor để đưa vào model VLM/CLIP."""
    if device is None:
        device = getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {image_path}")

    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    tensor = transform(image).unsqueeze(0).to(device)
    
    dtype_str = getattr(config, "DTYPE", "float32")
    dtype = getattr(torch, dtype_str, torch.float32)
    return tensor.to(dtype)


class KeyframeDataset(Dataset):
    """Dataset load ảnh đa luồng bằng PyTorch DataLoader"""
    def __init__(self, items):
        self.items = items
        self.transform = build_transform(448)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            image = Image.open(item["keyframe_path"]).convert("RGB")
            tensor = self.transform(image)
        except Exception as e:
            tensor = torch.zeros((3, 448, 448))
        return item, tensor


def collate_fn(batch):
    items = [b[0] for b in batch]
    pixel_values = torch.stack([b[1] for b in batch], dim=0) # (B, 3, 448, 448)
    return items, pixel_values


# === FIX LỖI 1: Thêm model_path vào tham số đầu vào của hàm ===
def load_internvl(model_path: str = None, quantize4bit: bool = False):
    from transformers import AutoModel, AutoTokenizer
    
    # Ưu tiên model_path truyền từ CLI, nếu không dùng từ config
    model_id = model_path or getattr(config, "INTERNVL_MODEL_ID", "OpenGVLab/InternVL2_5-2B")
    dtype = getattr(torch, config.DTYPE)
    load_kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        _fast_init=False,
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

    # === FIX LỖI 2: Dùng model_id đã được tính toán thay vì config cố định ===
    model = AutoModel.from_pretrained(model_id, **load_kwargs).eval()
    if not quantize4bit:
        model = model.to(config.DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=False
    )
    return model, tokenizer


CAPTION_PROMPT = (
   "<image>\n"
    "Mô tả chi tiết khung hình video này bằng tiếng Việt:\n"
    "1. Liệt kê các văn bản, chữ xuất hiện trên màn hình (bao gồm tên kênh truyền hình, logo như HTV7, HTV, VTV, đồng hồ thời gian nếu có).\n"
    "2. Đối tượng chính: ngôi nhà, kiểu kiến trúc (nhà tạm, nhà tôn, nhà sàn, nhà lá, nhà cao tầng).\n"
    "3. Bối cảnh & môi trường: ven sông, con kênh, rác thải nhựa, ô nhiễm, cây cối, chuối, sông nước.\n"
    "4. Hành động & con người (nếu có).\n"
    "Trả lời ngắn gọn từ 2-3 câu chứa đầy đủ các từ khóa trực quan."
)


def caption_batch(model, tokenizer, pixel_values, batch_size, generation_config):
    """Caption 1 batch tensor đã được load sẵn từ DataLoader"""
    questions = [CAPTION_PROMPT] * batch_size
    num_patches_list = [1] * batch_size

    if hasattr(model, "batch_chat"):
        responses = model.batch_chat(
            tokenizer, pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config,
        )
    else:
        print("\n[WARN] Model không hỗ trợ batch_chat! Đang fallback về chat từng ảnh...")
        responses = []
        for i in range(batch_size):
            pv = pixel_values[i:i+1]
            responses.append(model.chat(tokenizer, pv, CAPTION_PROMPT, generation_config))

    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dedup-map", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--quantize4bit", action="store_true")
    parser.add_argument("--model-path", type=str, default=None, 
                        help="Path local hoặc HuggingFace ID cho InternVL")
    args = parser.parse_args()

    items = list(utils.read_jsonl(config.ID_MAP_PATH))

    if args.dedup_map:
        dedup_path = os.path.join(config.INDEX_DIR, "dedup_map.jsonl")
        assert os.path.isfile(dedup_path), "Chưa có dedup_map.jsonl"
        dedup_rows = list(utils.read_jsonl(dedup_path))
        representative_ids = {r["int_id"] for r in dedup_rows if r["int_id"] == r["representative_int_id"]}
        items = [it for it in items if it["int_id"] in representative_ids]

    items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]

    out_path = config.CAPTION_JSONL_PATH + f".shard{args.shard}"
    done_ids = set()
    if os.path.isfile(out_path):
        done_ids = {row["int_id"] for row in utils.read_jsonl(out_path)}

    items = [it for it in items if it["int_id"] not in done_ids]
    print(f"Còn lại cần caption: {len(items)} (batch_size={args.batch_size})")

    if len(items) == 0:
        print("Đã hoàn tất toàn bộ!")
        return

    # In log rõ ràng nguồn model load từ đâu
    target_model_path = args.model_path or getattr(config, "INTERNVL_MODEL_ID", "OpenGVLab/InternVL2_5-2B")
    print(f"Đang load model từ: {target_model_path}...")
    
    model, tokenizer = load_internvl(model_path=args.model_path, quantize4bit=args.quantize4bit)
    dtype = getattr(torch, config.DTYPE)
    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    dataset = KeyframeDataset(items)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    for batch_items, pixel_values in tqdm(dataloader, desc=f"Captioning (shard {args.shard})"):
        pixel_values = pixel_values.to(dtype).to(config.DEVICE)
        
        try:
            with torch.no_grad():
                captions = caption_batch(model, tokenizer, pixel_values, len(batch_items), generation_config)
        except Exception as e:
            print(f"\n[WARN] Lỗi batch: {e} -- Fallback rỗng")
            captions = [""] * len(batch_items)

        records = [
            {
                "int_id": it["int_id"],
                "video_id": it["video_id"],
                "frame_id": it["frame_id"],
                "caption": cap,
            }
            for it, cap in zip(batch_items, captions)
        ]
        
        with open(out_path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(utils.json_dumps(r) + "\n")

    print(f"Hoàn tất shard {args.shard} -> {out_path}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
