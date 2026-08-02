# AIC 2026 — Pipeline Retrieval (SigLIP2 + BM25 hybrid + RRF)

Pipeline cho truy vấn **Textual KIS** (và làm nền cho Q&A). Tách rõ 2 giai đoạn:
build index (offline, chạy 1 lần) và query (online, chạy lúc thi).

## 0. Cài đặt

```bash
pip install -r requirements.txt
```

Nếu chỉ có CPU (không khuyến nghị vì rất chậm khi encode/caption toàn bộ keyframe),
đổi `DEVICE = "cpu"` và `DTYPE = "float32"` trong `config.py`.

## 1. Chỉnh đường dẫn dataset

Sửa `config.py`, hoặc set biến môi trường:

```bash
export AIC_DATASET_ROOT=/path/to/aic2026_batch1
export AIC_INDEX_DIR=./index_data
```

**Quan trọng:** hàm `_read_frame_index_metadata()` trong `utils.py` giả định file
map tên-keyframe → frame_id thật nằm ở `Keyframes/<video_id>/map.json`. Nếu BTC
cấp format khác (ví dụ 1 file metadata chung, hoặc field trong Metadata/*.json),
**phải sửa lại hàm này trước khi build index** — đây là bước quan trọng nhất vì
toàn bộ answer nộp bài phụ thuộc frame_id đúng.

## 2. Build index (offline)

```bash
# Bước 1: encode dense bằng SigLIP2 -> FAISS + id_map.jsonl
python build_dense_index.py --batch-size 64

# Bước 2: caption toàn bộ keyframe bằng InternVL2.5 (chậm nhất, nên chia shard
# chạy song song nếu có nhiều GPU)
python build_captions.py --shard 0 --num-shards 1
cat index_data/captions.jsonl.shard* > index_data/captions.jsonl   # nếu chạy nhiều shard

# Bước 3: build BM25 từ caption + object + metadata
python build_bm25_index.py
```

Test nhanh trên tập nhỏ trước khi chạy full (tránh phí thời gian nếu config sai):

```bash
python build_dense_index.py --limit 200
```

## 3. Query (online, lúc thi)

```bash
python query_cli.py --query "Tìm cảnh một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh" \
    --top-k 100 --rerank --out-csv result.csv
```

- `--rerank`: bật VLM rerank tầng cuối (top 30 sau RRF) — tăng độ chính xác nhưng
  chậm hơn, cân nhắc bật/tắt tuỳ ngân sách thời gian mỗi truy vấn lúc thi.
- `result.csv` có cột `rank, video_id, frame_id, score` — nộp theo đúng thứ tự
  rank để tối ưu điểm `R@k` (BTC chấm theo Top-1/5/20/50/100, xem mục 2.2 đề bài).

## 4. Cấu trúc file

```
config.py            # đường dẫn dataset, tên model, tham số search
utils.py              # quét dataset, đọc object/metadata, tokenizer tiếng Việt
build_dense_index.py   # encode SigLIP2 -> FAISS
build_captions.py      # caption bằng InternVL2.5 (offline, chạy theo shard)
build_bm25_index.py    # ghép caption+object+metadata -> BM25Okapi
search.py              # DenseSearcher, BM25Searcher, rrf_fusion, hybrid_search
translate.py           # dịch query VI->EN cho nhánh SigLIP2
vlm_rerank.py           # rerank top-K bằng InternVL2.5
query_cli.py            # CLI end-to-end, xuất CSV nộp bài
```

## 5. Việc cần tự kiểm tra / điều chỉnh

- [ ] Xác nhận format thật của file map frame_id trong dataset BTC cấp, sửa
      `utils._read_frame_index_metadata()`.
- [ ] Xác nhận format Objects/*.json thật (đề nói theo chuẩn TensorFlow
      OpenImages) — `utils.load_objects()` đã hỗ trợ 2 format phổ biến, kiểm tra
      lại với 1-2 file mẫu.
- [ ] Đo thời gian caption InternVL2.5 trên 200 ảnh mẫu trước khi chạy full —
      đây là bước tốn thời gian nhất, cần ước lượng để lên kế hoạch batch 2.
- [ ] Tune `RRF_K`, ngưỡng `OBJECT_SCORE_THRESHOLD`, và bật/tắt `--rerank` dựa
      trên 1 tập validation query tự tạo (mô phỏng theo ví dụ đề bài) trước khi
      thi thật.
