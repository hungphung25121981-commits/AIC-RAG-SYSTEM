"""
Build nhánh Dense: encode toàn bộ keyframe bằng SigLIP2 -> FAISS index.

Chạy:
    python build_dense_index.py --batch-size 64

Output:
    - config.FAISS_INDEX_PATH   : FAISS index (IndexFlatIP, cosine sim vì đã normalize)
    - config.ID_MAP_PATH        : jsonl map int_id -> (video_id, frame_id, path)
"""

import argparse
import os
import faiss
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config
import utils


class ImageDataset(Dataset):
    """Dataset load ảnh đa luồng bằng CPU trước khi chuyển sang GPU."""
    def __init__(self, items, processor):
        self.items = items
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        p = item.keyframe_path
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"\n[WARN] Không mở được ảnh {p}: {e}")
            img = Image.new("RGB", (384, 384))  # Ảnh placeholder tránh lệch index

        return img


def collate_fn_factory(processor):
    def collate_fn(batch_imgs):
        # Batching ảnh bằng processor của SigLIP2
        inputs = processor(images=batch_imgs, return_tensors="pt")
        return inputs
    return collate_fn


def load_siglip2():
    from transformers import AutoModel, AutoProcessor

    dtype = getattr(torch, config.DTYPE)
    model = AutoModel.from_pretrained(config.SIGLIP2_MODEL_ID, torch_dtype=dtype).to(config.DEVICE).eval()
    processor = AutoProcessor.from_pretrained(config.SIGLIP2_MODEL_ID)
    return model, processor


@torch.no_grad()
def encode_images(model, processor, items, batch_size, device):
    """Encode danh sách ảnh -> ma trận embedding đã L2-normalize (numpy float32)."""
    dataset = ImageDataset(items, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,        # Đọc 4 ảnh song song bằng CPU
        pin_memory=True,      # Tăng tốc độ nạp dữ liệu lên GPU
        collate_fn=collate_fn_factory(processor)
    )

    all_embs = []
    dtype = getattr(torch, config.DTYPE)

    for inputs in tqdm(dataloader, desc="Encoding keyframes (SigLIP2)"):
        # Đưa tensor inputs sang GPU với dtype phù hợp
        inputs = {k: v.to(device=device, dtype=dtype) if v.dtype.is_floating_point else v.to(device) 
                  for k, v in inputs.items()}

        outputs = model.get_image_features(**inputs)
        feats = utils.extract_feature_tensor(outputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        all_embs.append(feats.float().cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số keyframe để test nhanh")
    args = parser.parse_args()

    print("Đang quét dataset...")
    items = list(utils.iter_all_keyframes())
    if args.limit:
        items = items[: args.limit]
    print(f"Tổng số keyframe: {len(items)}")

    if len(items) == 0:
        raise RuntimeError(
            "Không có keyframe nào để encode (items rỗng). Chạy 'python check_config.py' "
            "để kiểm tra lại path dataset trước khi chạy tiếp."
        )

    print("Đang load model SigLIP2...")
    model, processor = load_siglip2()

    embeddings = encode_images(model, processor, items, args.batch_size, config.DEVICE)

    assert embeddings.shape[0] == len(items), "Số embedding không khớp số keyframe!"

    print("Đang build FAISS index (IndexFlatIP)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    print(f"Đã lưu FAISS index -> {config.FAISS_INDEX_PATH} (dim={dim}, ntotal={index.ntotal})")

    print("Đang ghi id map...")
    utils.write_jsonl(
        config.ID_MAP_PATH,
        (
            {
                "int_id": it.int_id,
                "video_id": it.video_id,
                "frame_id": it.frame_id,
                "pts_time": it.pts_time,
                "keyframe_path": it.keyframe_path,
                "object_json_path": it.object_json_path,
            }
            for it in items
        ),
    )
    print(f"Đã lưu id map -> {config.ID_MAP_PATH}")


if __name__ == "__main__":
    main()
