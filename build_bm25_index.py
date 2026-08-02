"""
Build nhánh BM25: ghép [caption InternVL2.5] + [object labels] + [metadata video]
thành 1 document text / keyframe, tokenize tiếng Việt, build BM25Okapi.

Yêu cầu chạy sau:
    1. build_dense_index.py   (tạo id_map.jsonl)
    2. build_captions.py      (tạo captions.jsonl, có thể bỏ qua nếu chưa kịp caption
                                 -> BM25 vẫn chạy được chỉ với object + metadata, chỉ yếu hơn)

Chạy:
    python build_bm25_index.py

Output:
    config.BM25_CORPUS_PATH  : text corpus theo từng int_id (để debug/đọc lại)
    config.BM25_PICKLE_PATH  : BM25Okapi đã build sẵn (pickle)
"""

import pickle
from collections import defaultdict

from rank_bm25 import BM25Okapi
from tqdm import tqdm

import config
import utils


def main():
    id_map = list(utils.read_jsonl(config.ID_MAP_PATH))
    print(f"Tổng số keyframe: {len(id_map)}")

    # 1) load caption nếu có
    captions = {}
    import os
    if os.path.isfile(config.CAPTION_JSONL_PATH):
        for row in utils.read_jsonl(config.CAPTION_JSONL_PATH):
            captions[row["int_id"]] = row.get("caption", "")
    else:
        print("[WARN] Chưa thấy captions.jsonl -- BM25 sẽ chỉ dùng object + metadata.")

    # 2) cache metadata theo video_id (tránh đọc file JSON lặp lại)
    metadata_cache = {}

    # 3) build corpus
    corpus_rows = []
    for it in tqdm(id_map, desc="Building BM25 corpus"):
        video_id = it["video_id"]
        if video_id not in metadata_cache:
            meta = utils.load_metadata(video_id)
            title = meta.get("title", "") or meta.get("Title", "")
            desc = meta.get("description", "") or meta.get("Description", "")
            metadata_cache[video_id] = f"{title} {desc}".strip()

        caption_text = captions.get(it["int_id"], "")
        object_labels = utils.load_objects(it.get("object_json_path"))
        doc_text = " ".join([
            caption_text,
            " ".join(object_labels),
            metadata_cache[video_id],
        ]).strip()

        tokens = utils.tokenize_vi(doc_text)
        corpus_rows.append({
            "int_id": it["int_id"],
            "video_id": video_id,
            "frame_id": it["frame_id"],
            "text": doc_text,
            "tokens": tokens,
        })

    utils.write_jsonl(config.BM25_CORPUS_PATH, iter(corpus_rows))
    print(f"Đã lưu corpus text -> {config.BM25_CORPUS_PATH}")

    print("Đang build BM25Okapi...")
    tokenized_corpus = [row["tokens"] for row in corpus_rows]
    bm25 = BM25Okapi(tokenized_corpus)

    int_ids_order = [row["int_id"] for row in corpus_rows]  # thứ tự khớp tokenized_corpus

    with open(config.BM25_PICKLE_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "int_ids_order": int_ids_order}, f)
    print(f"Đã lưu BM25 index -> {config.BM25_PICKLE_PATH}")


if __name__ == "__main__":
    main()
