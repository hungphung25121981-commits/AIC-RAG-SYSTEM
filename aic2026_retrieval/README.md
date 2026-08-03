# AIC 2026 — Pipeline Retrieval (SigLIP2 + BM25 hybrid + RRF)

Pipeline cho truy vấn **Textual KIS** (và làm nền cho Q&A). Tách rõ 2 giai đoạn:
build index (offline, chạy 1 lần) và query (online, chạy lúc thi).

**Chạy được trên mọi nền tảng (local / Kaggle / Colab)** bằng cách đổi 1 biến
môi trường `AIC_CONFIG` — code không cần sửa gì. Xem mục 0 và 1 bên dưới.

## 0. Test nhanh với dữ liệu mẫu (khuyến nghị làm TRƯỚC KHI đụng vào dataset thật)

Sinh 1 bộ dữ liệu giả nhỏ (vài video, vài chục keyframe, đúng format thật của
AIC) để kiểm tra toàn bộ pipeline chạy thông suốt từ đầu đến cuối trong vài phút,
tránh mất hàng giờ trên dataset thật mới phát hiện lỗi cấu hình:

```bash
pip install -r requirements.txt
python create_sample_dataset.py --num-videos 3 --frames-per-video 20

AIC_CONFIG=configs/sample.yaml python check_config.py
AIC_CONFIG=configs/sample.yaml python build_dense_index.py
AIC_CONFIG=configs/sample.yaml python dedup_keyframes.py
AIC_CONFIG=configs/sample.yaml python build_captions.py --dedup-map --max-new-tokens 32
AIC_CONFIG=configs/sample.yaml python build_bm25_index.py
AIC_CONFIG=configs/sample.yaml python query_cli.py --query "người mặc áo đỏ"
```

Nếu cả chuỗi lệnh trên chạy không lỗi và `query_cli.py` in ra kết quả (dù không
chính xác về nội dung vì ảnh là giả) — nghĩa là pipeline wiring đã đúng, có thể
yên tâm chuyển sang chạy dataset thật.

## 1. Cài đặt & chọn nền tảng

```bash
pip install -r requirements.txt
```

Chọn file config theo nền tảng đang chạy (không cần sửa code, chỉ set 1 biến
môi trường trước khi gọi bất kỳ script nào):

| Nền tảng | Cách chạy |
|---|---|
| Local | không set gì -- mặc định dùng `configs/base.yaml` |
| Kaggle | `AIC_CONFIG=configs/kaggle.yaml python build_dense_index.py` (hoặc set `os.environ["AIC_CONFIG"]` trong cell đầu notebook, TRƯỚC khi import bất kỳ script nào) |
| Colab | `AIC_CONFIG=configs/colab.yaml python build_dense_index.py` |
| Test nhanh | `AIC_CONFIG=configs/sample.yaml python build_dense_index.py` |

Mỗi file trong `configs/` định nghĩa đường dẫn 5 thư mục dataset
(Videos/Keyframes/Objects/CLIPFeatures/Metadata) + `map-keyframes` (file CSV
frame index thật) + model + tham số search. Muốn thêm nền tảng mới (server
riêng, Paperspace...) chỉ cần copy 1 file trong `configs/` rồi sửa path, không
đụng vào code.

## 2. Chỉnh đường dẫn dataset trong file config đang dùng

Trên Kaggle (`configs/kaggle.yaml`), mọi giá trị đã để sẵn `"auto"` — code tự
quét `/kaggle/input/*/` tìm đúng thư mục, miễn là bạn đã **Add Data** đủ 6
dataset (Videos/Keyframes/Objects/CLIPFeatures/Metadata/map-keyframes).

Luôn chạy kiểm tra trước khi build index:

```bash
AIC_CONFIG=configs/kaggle.yaml python check_config.py
```

Nếu báo `[LỖI]` ở dòng nào — sửa trực tiếp trong file `configs/kaggle.yaml`
(đổi `"auto"` thành path cụ thể), không cần set biến môi trường rời rạc nữa.

**Về frame_id:** pipeline ưu tiên đọc frame_id thật từ file
`map-keyframes/<video_id>.csv` (cột `n, pts_time, fps, frame_idx` — đúng format
chuẩn của dataset AIC). Nếu dataset của bạn không có thư mục này, kiểm tra lại
xem BTC có cấp kèm không (thường đi cùng Keyframes), vì đây là cách duy nhất
đảm bảo frame_id chính xác 100% khi nộp bài.

## 3. Build index (offline)

```bash
# Bước 1: encode dense bằng SigLIP2 -> FAISS + id_map.jsonl
python build_dense_index.py --batch-size 64

# Bước 1b (KHUYẾN NGHỊ MẠNH nếu chạy trên Kaggle/Colab có giới hạn thời gian):
# dedup các frame gần giống nhau (cảnh tĩnh kéo dài) trước khi caption, thường
# giảm 50-80% khối lượng caption -> tránh vượt giới hạn 9-12h của Kaggle
python dedup_keyframes.py --threshold 0.97

# Bước 2: caption keyframe bằng InternVL2.5 (chậm nhất trong pipeline)
python build_captions.py --shard 0 --num-shards 1 --dedup-map --max-new-tokens 48
cat index_data/captions.jsonl.shard* > index_data/captions.jsonl   # nếu chạy nhiều shard

# Bước 3: build BM25 từ caption + object + metadata
# (tự động dùng dedup_map.jsonl nếu tồn tại, các frame trùng dùng chung caption
#  với frame đại diện, không cần caption riêng từng frame)
python build_bm25_index.py
```

### Nếu vẫn có nguy cơ vượt giới hạn thời gian (Kaggle Save & Run All ~9-12h)

Bỏ `--dedup-map` không đủ giảm, cân nhắc thêm:
- Đổi `INTERNVL_MODEL_ID` trong `config.py` sang bản nhẹ hơn (`OpenGVLab/InternVL2_5-2B` hoặc `-4B`) — nhanh hơn 3-4 lần so với bản 8B, đủ dùng vì caption chỉ là tín hiệu phụ trợ cho BM25, không phải nhánh chính.
- Chạy song song nhiều GPU bằng `--shard`:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python build_captions.py --shard 0 --num-shards 2 --dedup-map &
  CUDA_VISIBLE_DEVICES=1 python build_captions.py --shard 1 --num-shards 2 --dedup-map &
  wait
  ```
- Nếu vẫn không đủ trong 1 session: chủ động dừng ở ~8-9h (chừa margin an toàn), publish `index_data/captions.jsonl.shard*` hiện có thành Kaggle Dataset checkpoint, mở session mới, copy checkpoint vào `index_data/` rồi chạy lại đúng lệnh cũ — code tự resume (bỏ qua `int_id` đã có trong file `.shard*`), lặp lại tới khi xong.

Test nhanh trên tập nhỏ trước khi chạy full (tránh phí thời gian nếu config sai):

```bash
python build_dense_index.py --limit 200
```

## 4. Query (online, lúc thi)

```bash
python query_cli.py --query "Tìm cảnh một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh" \
    --top-k 100 --rerank --out-csv result.csv
```

- `--rerank`: bật VLM rerank tầng cuối (top 30 sau RRF) — tăng độ chính xác nhưng
  chậm hơn, cân nhắc bật/tắt tuỳ ngân sách thời gian mỗi truy vấn lúc thi.
- `result.csv` có cột `rank, video_id, frame_id, score` — nộp theo đúng thứ tự
  rank để tối ưu điểm `R@k` (BTC chấm theo Top-1/5/20/50/100, xem mục 2.2 đề bài).

## 5. Cấu trúc file

```
configs/
  base.yaml           # config local (mặc định)
  kaggle.yaml          # config Kaggle (auto-detect /kaggle/input)
  colab.yaml           # config Google Colab
  sample.yaml          # config dùng với dữ liệu mẫu (create_sample_dataset.py)
config.py             # đọc file trong configs/ theo AIC_CONFIG, expose path/model/tham số
utils.py              # quét dataset, đọc map-keyframes CSV/object/metadata, tokenizer tiếng Việt
check_config.py         # kiểm tra nhanh path trước khi chạy job dài
create_sample_dataset.py # sinh dữ liệu giả nhỏ để smoke-test pipeline
build_dense_index.py   # encode SigLIP2 -> FAISS
dedup_keyframes.py      # gộp frame gần trùng nhau, giảm khối lượng caption
build_captions.py      # caption bằng InternVL2.5 (offline, chạy theo shard)
build_bm25_index.py    # ghép caption+object+metadata -> BM25Okapi
search.py              # DenseSearcher, BM25Searcher, rrf_fusion, hybrid_search
translate.py           # dịch query VI->EN cho nhánh SigLIP2
vlm_rerank.py           # rerank top-K bằng InternVL2.5
query_cli.py            # CLI end-to-end, xuất CSV nộp bài
debug_pipeline.py       # kiểm tra alignment index + cô lập lỗi dense/bm25/rrf
```

## 6. Việc cần tự kiểm tra / điều chỉnh

- [x] ~~Xác nhận format frame_id~~ -- đã fix: pipeline đọc `map-keyframes/<video_id>.csv`
      (cột `n, pts_time, fps, frame_idx`), đúng format chuẩn dataset AIC.
- [ ] Xác nhận format Objects/*.json thật (đề nói theo chuẩn TensorFlow
      OpenImages) — `utils.load_objects()` đã hỗ trợ 2 format phổ biến, kiểm tra
      lại với 1-2 file mẫu.
- [ ] Đo thời gian caption InternVL2.5 trên 200 ảnh mẫu trước khi chạy full —
      đây là bước tốn thời gian nhất, cần ước lượng để lên kế hoạch batch 2.
- [ ] Tune `RRF_K`, ngưỡng `OBJECT_SCORE_THRESHOLD`, và bật/tắt `--rerank` dựa
      trên 1 tập validation query tự tạo (mô phỏng theo ví dụ đề bài) trước khi
      thi thật.
