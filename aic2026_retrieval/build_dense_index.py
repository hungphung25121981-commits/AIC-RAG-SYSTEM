"""
Build nhánh Dense: encode toàn bộ keyframe bằng SigLIP2 -> FAISS index.

Chạy:
    python build_dense_index.py --batch-size 64

Output:
    - config.FAISS_INDEX_PATH   : FAISS index (IndexFlatIP, cosine sim vì đã normalize)
    - config.ID_MAP_PATH        : jsonl map int_id -> (video_id, frame_id, path)
"""

import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import config
import utils


def load_siglip2():
    from transformers import AutoModel, AutoProcessor

    dtype = getattr(torch, config.DTYPE)
    model = AutoModel.from_pretrained(config.SIGLIP2_MODEL_ID, torch_dtype=dtype).to(config.DEVICE).eval()
    processor = AutoProcessor.from_pretrained(config.SIGLIP2_MODEL_ID)
    return model, processor


@torch.no_grad()
def encode_images(model, processor, image_paths, batch_size, device):
    """Encode danh sách ảnh -> ma trận embedding đã L2-normalize (numpy float32)."""
    all_embs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding keyframes (SigLIP2)"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"[WARN] Không mở được ảnh {p}: {e}")
                imgs.append(Image.new("RGB", (384, 384)))  # ảnh trắng placeholder, tránh lệch index

        inputs = processor(images=imgs, return_tensors="pt").to(device)
        inputs = {k: v.to(getattr(torch, config.DTYPE)) if v.dtype == torch.float32 else v
                  for k, v in inputs.items()}

        outputs = model.get_image_features(**inputs)
        # SigLIP2 tuỳ version transformers trả về tensor thẳng hoặc object
        # ModelOutput (pooler_output/image_embeds/...) -- xử lý chung ở utils.py
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

    image_paths = [it.keyframe_path for it in items]
    embeddings = encode_images(model, processor, image_paths, args.batch_size, config.DEVICE)

    assert embeddings.shape[0] == len(items), "Số embedding không khớp số keyframe!"

    print("Đang build FAISS index (IndexFlatIP)...")
    import faiss
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
