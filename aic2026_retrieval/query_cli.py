"""
CLI chạy 1 query end-to-end: Dense (SigLIP2) + BM25 -> RRF -> (tuỳ chọn) VLM rerank
-> in ra danh sách (rank, video_id, frame_id) theo đúng format nộp bài Textual KIS.

Chạy:
    python query_cli.py --query "Tìm cảnh một diễn giả mặc áo đỏ phát biểu ngoài trời" --rerank

    # Không rerank (nhanh hơn, dùng lúc cần tốc độ / thử nghiệm):
    python query_cli.py --query "..." --top-k 100
    
"""

import argparse
import csv
import os
import torch

import config
import search
import utils


def main():
    parser = argparse.ArgumentParser(description="AIC Retrieval & Visual QA CLI")
    parser.add_argument("--query", type=str, required=True, help="Câu truy vấn tiếng Việt để tìm kiếm frame")
    parser.add_argument("--top-k", type=int, default=config.FINAL_TOPK, help="Số lượng kết quả xuất ra")
    parser.add_argument("--rerank", action="store_true", help="Bật VLM rerank tầng cuối cho danh sách kết quả")
    parser.add_argument("--out-csv", type=str, default=None, help="Ghi kết quả ra file CSV nộp bài")
    
    # Dành riêng cho câu hỏi VQA
    parser.add_argument("--question", type=str, default=None, help="Câu hỏi cần VLM trả lời dựa trên Top-1 frame")
    parser.add_argument("--qa", action="store_true", help="Bật chế độ QA")
    args = parser.parse_args()

    # =========================================================================
    # 1. RETRIEVAL (Giữ nguyên như cũ: Hybrid Search Dense + BM25)
    # =========================================================================
    print("Đang load index (dense + bm25)...")
    dense_searcher = search.DenseSearcher()
    bm25_searcher = search.BM25Searcher()

    print("Đang dịch query VI->EN cho nhánh Dense...")
    try:
        import translate
        query_en = translate.translate_vi2en(args.query)
    except Exception as e:
        print(f"[WARN] Lỗi dịch tự động ({e}), dùng câu gốc cho Dense search.")
        query_en = args.query

    print(f"  VI: {args.query}")
    print(f"  EN: {query_en}")

    fused = search.hybrid_search(args.query, query_en, dense_searcher, bm25_searcher, top_k=args.top_k)

    id_map_rows = list(utils.read_jsonl(config.ID_MAP_PATH))
    id_map_by_int_id = {row["int_id"]: row for row in id_map_rows}

    # Bật Rerank nếu truyền flag --rerank
    if args.rerank:
        print("Đang giải phóng bộ nhớ GPU trước khi chạy VLM Rerank...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Đang chạy VLM rerank tầng cuối (InternVL2.5)...")
        import vlm_rerank
        reranker = vlm_rerank.VLMReranker()
        final_results = reranker.rerank(fused, args.query, id_map_by_int_id, top_k=config.RERANK_TOPK)
        
        reranked_ids = {i for i, _ in final_results}
        remaining = [(i, s) for i, s in fused if i not in reranked_ids]
        final_results = final_results + remaining
    else:
        final_results = fused

    # Hiển thị và lưu CSV
    print("\n=== KẾT QUẢ (rank | video_id | frame_id | score) ===")
    rows_out = []
    for rank, (int_id, score) in enumerate(final_results[: args.top_k], start=1):
        row = id_map_by_int_id[int_id]
        print(f"{rank:3d} | {row['video_id']:15s} | frame_id={row['frame_id']:<8d} | score={score:.4f}")
        rows_out.append({"rank": rank, "video_id": row["video_id"], "frame_id": row["frame_id"], "score": score})

    if args.out_csv:
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["rank", "video_id", "frame_id", "score"])
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nĐã ghi kết quả -> {args.out_csv}")

    # =========================================================================
    # 2. VQA (Chỉ chạy khi có --question hoặc --qa)
    # =========================================================================
   # BƯỚC VQA
    # BƯỚC VQA
    if args.question or args.qa:
        q_text = args.question if args.question else args.query
        top1_int_id = final_results[0][0]
        top1_row = id_map_by_int_id[top1_int_id]

        print("\n=================== VLM ANSWER (TOP-1 FRAME) ===================")
        print(f"Target Video: {top1_row['video_id']} | Frame ID: {top1_row['frame_id']}")
        print(f"Question    : {q_text}")

        # 1. Thử lấy đường dẫn có sẵn từ id_map
        img_path = top1_row.get("keyframe_path") or top1_row.get("path")

        # 2. Nếu không có hoặc file không tồn tại, tự dò tìm linh hoạt trên đĩa
        if not img_path or not os.path.exists(img_path):
            video_dir = os.path.join(config.KEYFRAMES_DIR, top1_row["video_id"])
            frame_id_str = str(top1_row["frame_id"])

            if os.path.exists(video_dir):
                # Quét tất cả file trong thư mục video để tìm file chứa frame_id
                matched_files = [
                    os.path.join(video_dir, f) for f in os.listdir(video_dir)
                    if frame_id_str in f and f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                if matched_files:
                    img_path = matched_files[0]  # Lấy file khớp đầu tiên
            
            # Fallback nếu quét không ra
            if not img_path:
                img_path = os.path.join(video_dir, f"{top1_row['frame_id']:06d}.jpg")

        print(f"Keyframe Path resolved: {img_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            import vlm_rerank
            answer = vlm_rerank.answer_question(keyframe_path=img_path, question=q_text)
            print(f"\n👉 ANSWER: {answer}")
        except Exception as e:
            print(f"[ERROR] Không thể chạy VQA: {e}")
        print("================================================================\n")


if __name__ == "__main__":
    main()
