"""
Script kiểm tra sức khoẻ pipeline khi search ra sai video/sai rank.
Chạy sau khi đã build index xong:

    python debug_pipeline.py --query "câu query đang bị sai" --expected-video L01_V001

Sẽ in ra:
  1. Kiểm tra alignment: số lượng FAISS ntotal / id_map / bm25 có khớp nhau không.
  2. Top-5 riêng của Dense, riêng của BM25, và sau khi RRF fuse -- để biết
     nhánh nào đang "kéo" kết quả sai.
  3. Nếu truyền --expected-video, kiểm tra video đó có nằm trong top-K của
     từng nhánh không, và ở rank bao nhiêu (giúp biết là "tìm không ra" hay
     "tìm ra nhưng rank thấp").
"""

import argparse
import pickle

import config
import search
import translate
import utils


def check_alignment():
    print("=" * 60)
    print("BƯỚC 1: KIỂM TRA ALIGNMENT GIỮA CÁC INDEX")
    print("=" * 60)

    import faiss
    faiss_index = faiss.read_index(config.FAISS_INDEX_PATH)
    id_map = list(utils.read_jsonl(config.ID_MAP_PATH))

    with open(config.BM25_PICKLE_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25_ids = bm25_data["int_ids_order"]

    print(f"FAISS ntotal        : {faiss_index.ntotal}")
    print(f"id_map.jsonl rows    : {len(id_map)}")
    print(f"BM25 int_ids_order   : {len(bm25_ids)}")

    problems = []
    if faiss_index.ntotal != len(id_map):
        problems.append(
            f"[LỖI] FAISS ({faiss_index.ntotal}) và id_map ({len(id_map)}) LỆCH SỐ LƯỢNG "
            "-> gần như chắc chắn đây là nguyên nhân 'sai video'. Phải build lại "
            "build_dense_index.py để 2 file này sinh ra CÙNG 1 lần chạy."
        )
    if len(bm25_ids) != len(id_map):
        problems.append(
            f"[LỖI] BM25 ({len(bm25_ids)}) và id_map ({len(id_map)}) LỆCH SỐ LƯỢNG "
            "-> build_bm25_index.py đang đọc id_map.jsonl khác với lúc build BM25. "
            "Build lại build_bm25_index.py."
        )

    # kiểm tra int_id có liên tục 0..N-1 và không trùng lặp không
    int_ids = sorted(row["int_id"] for row in id_map)
    expected = list(range(len(id_map)))
    if int_ids != expected:
        problems.append(
            "[LỖI] int_id trong id_map.jsonl KHÔNG liên tục/không khớp 0..N-1 "
            "-> có id bị trùng hoặc bị nhảy số, sẽ làm map ngược sai video."
        )

    # spot-check vài dòng đầu/cuối để xem frame_id có "nhìn hợp lý" không
    print("\n3 dòng đầu id_map (kiểm tra thủ công video_id/frame_id có hợp lý không):")
    for row in id_map[:3]:
        print(" ", row)
    print("3 dòng cuối id_map:")
    for row in id_map[-3:]:
        print(" ", row)

    if problems:
        print("\n>>> PHÁT HIỆN VẤN ĐỀ:")
        for p in problems:
            print(" -", p)
    else:
        print("\n>>> Alignment OK, số lượng khớp nhau ở cả 3 index.")

    return id_map


def isolate_branches(query_vi, id_map, expected_video=None):
    print("\n" + "=" * 60)
    print("BƯỚC 2: TÁCH RIÊNG TỪNG NHÁNH ĐỂ CÔ LẬP LỖI")
    print("=" * 60)

    id_map_by_int_id = {row["int_id"]: row for row in id_map}

    dense = search.DenseSearcher()
    bm25 = search.BM25Searcher()

    query_en = translate.translate_vi2en(query_vi)
    print(f"\nQuery VI : {query_vi}")
    print(f"Query EN (dùng cho Dense): {query_en}")
    print("  -> Nếu bản dịch này SAI NGHĨA so với câu gốc, nhánh Dense sẽ tìm nhầm hướng.")

    dense_results = dense.search(query_en, top_k=20)
    bm25_results = bm25.search(query_vi, top_k=20)

    print("\n--- TOP-10 NHÁNH DENSE (SigLIP2) ---")
    for rank, (int_id, score) in enumerate(dense_results[:10], start=1):
        row = id_map_by_int_id[int_id]
        flag = "  <== khớp expected_video" if expected_video and row["video_id"] == expected_video else ""
        print(f"{rank:2d}. {row['video_id']:15s} frame_id={row['frame_id']:<8d} score={score:.4f}{flag}")

    print("\n--- TOP-10 NHÁNH BM25 ---")
    for rank, (int_id, score) in enumerate(bm25_results[:10], start=1):
        row = id_map_by_int_id[int_id]
        flag = "  <== khớp expected_video" if expected_video and row["video_id"] == expected_video else ""
        print(f"{rank:2d}. {row['video_id']:15s} frame_id={row['frame_id']:<8d} score={score:.4f}{flag}")

    fused = search.rrf_fusion(
        [[i for i, _ in dense_results], [i for i, _ in bm25_results]]
    )
    print("\n--- TOP-10 SAU RRF FUSION ---")
    for rank, (int_id, score) in enumerate(fused[:10], start=1):
        row = id_map_by_int_id[int_id]
        flag = "  <== khớp expected_video" if expected_video and row["video_id"] == expected_video else ""
        print(f"{rank:2d}. {row['video_id']:15s} frame_id={row['frame_id']:<8d} rrf_score={score:.5f}{flag}")

    if expected_video:
        print(f"\n--- Kiểm tra riêng video kỳ vọng: {expected_video} ---")
        for name, results in [("Dense", dense_results), ("BM25", bm25_results)]:
            found_rank = next(
                (r for r, (i, _) in enumerate(results, start=1) if id_map_by_int_id[i]["video_id"] == expected_video),
                None,
            )
            if found_rank:
                print(f"  [{name}] tìm thấy ở rank {found_rank}/{len(results)}")
            else:
                print(f"  [{name}] KHÔNG xuất hiện trong top-{len(results)} "
                      f"-> nhánh này không nhận diện được đúng nội dung, không phải lỗi fusion.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--expected-video", type=str, default=None,
                         help="video_id đáp án đúng (nếu biết), để kiểm tra rank cụ thể")
    args = parser.parse_args()

    id_map = check_alignment()
    isolate_branches(args.query, id_map, args.expected_video)


if __name__ == "__main__":
    main()
