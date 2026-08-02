"""
CLI chạy 1 query end-to-end: Dense (SigLIP2) + BM25 -> RRF -> (tuỳ chọn) VLM rerank
-> in ra danh sách (rank, video_id, frame_id) theo đúng format nộp bài Textual KIS.

Chạy:
    python query_cli.py --query "Tìm cảnh một diễn giả mặc áo đỏ phát biểu ngoài trời" --rerank

    # không rerank (nhanh hơn, dùng lúc cần tốc độ / thử nghiệm):
    python query_cli.py --query "..." --top-k 100
"""

import argparse
import csv

import config
import search
import translate
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Câu truy vấn tiếng Việt")
    parser.add_argument("--top-k", type=int, default=config.FINAL_TOPK)
    parser.add_argument("--rerank", action="store_true", help="Bật VLM rerank tầng cuối")
    parser.add_argument("--out-csv", type=str, default=None, help="Ghi kết quả ra file CSV nộp bài")
    args = parser.parse_args()

    print("Đang load index (dense + bm25)...")
    dense_searcher = search.DenseSearcher()
    bm25_searcher = search.BM25Searcher()

    print("Đang dịch query VI->EN cho nhánh Dense...")
    query_en = translate.translate_vi2en(args.query)
    print(f"  VI: {args.query}")
    print(f"  EN: {query_en}")

    fused = search.hybrid_search(args.query, query_en, dense_searcher, bm25_searcher, top_k=args.top_k)

    id_map_rows = list(utils.read_jsonl(config.ID_MAP_PATH))
    id_map_by_int_id = {row["int_id"]: row for row in id_map_rows}

    if args.rerank:
        print("Đang chạy VLM rerank tầng cuối (InternVL2.5)...")
        import vlm_rerank
        reranker = vlm_rerank.VLMReranker()
        final_results = reranker.rerank(fused, args.query, id_map_by_int_id, top_k=config.RERANK_TOPK)
        # phần còn lại (sau rerank_topk) giữ nguyên thứ tự RRF, nối vào cuối
        reranked_ids = {i for i, _ in final_results}
        remaining = [(i, s) for i, s in fused if i not in reranked_ids]
        final_results = final_results + remaining
    else:
        final_results = fused

    print("\n=== KẾT QUẢ (rank | video_id | frame_id | score) ===")
    rows_out = []
    for rank, (int_id, score) in enumerate(final_results[: args.top_k], start=1):
        row = id_map_by_int_id[int_id]
        print(f"{rank:3d} | {row['video_id']:15s} | frame_id={row['frame_id']:<8d} | score={score:.4f}")
        rows_out.append({"rank": rank, "video_id": row["video_id"], "frame_id": row["frame_id"], "score": score})

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["rank", "video_id", "frame_id", "score"])
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nĐã ghi kết quả -> {args.out_csv}")


if __name__ == "__main__":
    main()
