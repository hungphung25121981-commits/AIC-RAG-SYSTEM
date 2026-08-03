"""
Dedup keyframe: nhóm các frame liên tiếp (cùng video) có embedding SigLIP2 gần
giống nhau -> chỉ giữ 1 đại diện/nhóm để đưa vào build_captions.py.
Giảm mạnh số ảnh cần caption với video có nhiều cảnh tĩnh kéo dài.

Yêu cầu chạy sau build_dense_index.py (cần FAISS index + id_map.jsonl đã có).

Chạy:
    python dedup_keyframes.py --threshold 0.97

Output:
    config.INDEX_DIR/dedup_map.jsonl
    mỗi dòng: {"int_id": X, "representative_int_id": Y, "video_id": ..., "frame_id": ...}
    - nếu int_id == representative_int_id -> đây là đại diện, CẦN caption
    - nếu khác -> đây là frame trùng, KHÔNG cần caption riêng, dùng caption của
      representative_int_id khi build BM25 corpus.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import faiss

import config
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.97,
                         help="Cosine similarity >= threshold coi là trùng (0-1). "
                              "0.97-0.99 an toàn cho cảnh tĩnh; hạ xuống 0.93-0.95 nếu muốn dedup mạnh tay hơn.")
    args = parser.parse_args()

    print("Đang load FAISS index + id_map...")
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    id_map = list(utils.read_jsonl(config.ID_MAP_PATH))
    assert index.ntotal == len(id_map), \
        "FAISS và id_map lệch số lượng -- chạy debug_pipeline.py trước để kiểm tra alignment!"

    # nhóm theo video_id, giữ nguyên thứ tự int_id (vì iter_all_keyframes() duyệt
    # tuần tự theo từng thư mục video, các frame cùng video luôn liền kề nhau)
    by_video = defaultdict(list)
    for row in id_map:
        by_video[row["video_id"]].append(row)

    dedup_rows = []
    total_frames = 0
    total_representatives = 0

    for video_id, rows in by_video.items():
        rows = sorted(rows, key=lambda r: r["int_id"])
        rep_int_id = None
        rep_emb = None

        for row in rows:
            emb = index.reconstruct(row["int_id"]).reshape(1, -1)  # đã normalize sẵn lúc build
            emb = emb / (np.linalg.norm(emb) + 1e-8)

            if rep_emb is None:
                is_new_rep = True
            else:
                sim = float(np.dot(emb, rep_emb.T)[0, 0])
                is_new_rep = sim < args.threshold

            if is_new_rep:
                rep_int_id = row["int_id"]
                rep_emb = emb
                total_representatives += 1

            dedup_rows.append({
                "int_id": row["int_id"],
                "representative_int_id": rep_int_id,
                "video_id": video_id,
                "frame_id": row["frame_id"],
            })
            total_frames += 1

    out_path = os.path.join(config.INDEX_DIR, "dedup_map.jsonl")
    utils.write_jsonl(out_path, iter(dedup_rows))

    reduction = 100 * (1 - total_representatives / total_frames)
    print(f"\nTổng số keyframe        : {total_frames}")
    print(f"Số đại diện cần caption  : {total_representatives}")
    print(f"Giảm được               : {reduction:.1f}%")
    print(f"Đã lưu -> {out_path}")
    print("\nBước tiếp theo: chạy build_captions.py --dedup-map để chỉ caption phần đại diện.")


if __name__ == "__main__":
    main()
