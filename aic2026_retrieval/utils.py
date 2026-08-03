"""
Hàm tiện ích dùng chung: quét cấu trúc dataset AIC, đọc/ghi jsonl,
tokenizer tiếng Việt cho BM25.
"""

import os
import json
import csv
import re
from dataclasses import dataclass, asdict
from typing import Iterator, List, Optional

import config


@dataclass
class KeyframeItem:
    """Một keyframe = 1 đơn vị trong index."""
    int_id: int          # id nội bộ, tăng dần, dùng làm hàng trong FAISS
    video_id: str         # vd "L01_V001"
    frame_id: int          # chỉ số frame trong video gốc (đọc từ map-keyframes CSV)
    keyframe_path: str    # đường dẫn ảnh keyframe trên đĩa
    object_json_path: Optional[str] = None
    stem: str = ""         # tên file không đuôi, vd "0000"
    pts_time: Optional[float] = None   # thời điểm (giây) trong video, nếu có


def _read_map_keyframes_csv(video_id: str) -> Optional[list]:
    """
    Đọc file map-keyframes/<video_id>.csv -- format THẬT của dataset AIC
    (xác nhận qua dataset mẫu tham khảo), gồm 4 cột:
        n, pts_time, fps, frame_idx
    Trong đó:
      - n         : thứ tự keyframe (1, 2, 3, ...), khớp với thứ tự file ảnh
                    trong Keyframes/<video_id>/ khi sort tăng dần.
      - pts_time  : thời điểm (giây) của keyframe trong video.
      - frame_idx : chỉ số frame THẬT trong video gốc -- đây chính là giá trị
                    cần dùng làm frame_id khi nộp bài.
    Trả về list (frame_idx, pts_time) theo đúng thứ tự n=1,2,3... hoặc None nếu không có file.
    """
    path = os.path.join(config.MAPKEYFRAMES_DIR, video_id + ".csv")
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(float(row["frame_idx"])), float(row["pts_time"])))
    return rows


def _read_frame_index_metadata(video_kf_dir: str) -> dict:
    """
    Fallback cũ: đọc map.json trong chính thư mục keyframe nếu có (một số
    dataset AIC dùng format này thay vì CSV). Chỉ dùng khi không tìm thấy
    file map-keyframes/<video_id>.csv (xem _read_map_keyframes_csv ở trên,
    đây mới là nguồn chính xác nên ưu tiên).
    """
    map_path = os.path.join(video_kf_dir, "map.json")
    if os.path.isfile(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def iter_all_keyframes() -> Iterator[KeyframeItem]:
    """
    Duyệt toàn bộ Keyframes/<video_id>/*.jpg theo đúng thứ tự tăng dần,
    sinh ra KeyframeItem cho từng ảnh. int_id sinh tuần tự -> dùng để
    map ngược lại (video_id, frame_id) sau khi search FAISS.

    Thứ tự ưu tiên xác định frame_id thật:
      1) map-keyframes/<video_id>.csv (n, pts_time, fps, frame_idx) -- ĐÚNG NHẤT,
         khớp format thật của dataset AIC. Cột 'n' tương ứng thứ tự ảnh keyframe
         khi sort tăng dần trong thư mục.
      2) map.json trong thư mục keyframe (một số dataset dùng format này).
      3) fallback cuối: lấy số trong tên file làm frame_id (CHỈ đúng nếu tên
         file thực sự là frame index -- kém tin cậy nhất, nên tránh).
    """
    int_id = 0
    video_ids = sorted(os.listdir(config.KEYFRAMES_DIR))
    for video_id in video_ids:
        video_kf_dir = os.path.join(config.KEYFRAMES_DIR, video_id)
        if not os.path.isdir(video_kf_dir):
            continue

        obj_dir = os.path.join(config.OBJECTS_DIR, video_id)
        filenames = sorted(
            f for f in os.listdir(video_kf_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        csv_rows = _read_map_keyframes_csv(video_id)   # ưu tiên #1 -- list[(frame_idx, pts_time)]
        json_frame_map = {} if csv_rows else _read_frame_index_metadata(video_kf_dir)  # ưu tiên #2

        if csv_rows and len(csv_rows) != len(filenames):
            print(f"[WARN] {video_id}: số dòng CSV ({len(csv_rows)}) khác số "
                  f"file keyframe ({len(filenames)}) -- kiểm tra lại dataset, "
                  f"tạm fallback sang phương án khác cho video này.")
            csv_rows = None

        for i, fname in enumerate(filenames):
            stem = os.path.splitext(fname)[0]
            pts_time = None

            if csv_rows:
                frame_id, pts_time = csv_rows[i]
            elif stem in json_frame_map:
                frame_id = int(json_frame_map[stem])
            else:
                digits = re.sub(r"\D", "", stem)
                frame_id = int(digits) if digits else int_id

            obj_json = os.path.join(obj_dir, stem + ".json")
            yield KeyframeItem(
                int_id=int_id,
                video_id=video_id,
                frame_id=frame_id,
                keyframe_path=os.path.join(video_kf_dir, fname),
                object_json_path=obj_json if os.path.isfile(obj_json) else None,
                stem=stem,
                pts_time=pts_time,
            )
            int_id += 1


def load_metadata(video_id: str) -> dict:
    """Đọc Metadata/<video_id>.json (title, description YouTube). Có thể không tồn tại."""
    path = os.path.join(config.METADATA_DIR, video_id + ".json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_objects(object_json_path: Optional[str], score_th: float = None) -> List[str]:
    """
    Đọc file Object detection (format TensorFlow OpenImages) và trả về list
    tên class đã lọc theo ngưỡng confidence.
    Format phổ biến: {"detection_class_entities": [...], "detection_scores": [...]}
    hoặc {"detections": [{"class": ..., "score": ...}, ...]}. Hỗ trợ cả hai.
    """
    if not object_json_path or not os.path.isfile(object_json_path):
        return []
    score_th = config.OBJECT_SCORE_THRESHOLD if score_th is None else score_th
    with open(object_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = []
    if "detections" in data:
        for d in data["detections"]:
            if d.get("score", 0) >= score_th:
                labels.append(str(d.get("class", "")))
    elif "detection_class_entities" in data:
        entities = data.get("detection_class_entities", [])
        scores = data.get("detection_scores", [1.0] * len(entities))
        for cls, sc in zip(entities, scores):
            if float(sc) >= score_th:
                labels.append(str(cls))
    return [l for l in labels if l]


# ---------------------------------------------------------------------
# jsonl helpers
# ---------------------------------------------------------------------
def write_jsonl(path: str, rows: Iterator[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------
# Vietnamese tokenizer cho BM25 (fallback nếu chưa cài pyvi)
# ---------------------------------------------------------------------
def tokenize_vi(text: str) -> List[str]:
    text = (text or "").lower()
    try:
        from pyvi import ViTokenizer
        text = ViTokenizer.tokenize(text)
        tokens = text.split()
    except ImportError:
        # fallback: tách theo khoảng trắng + bỏ dấu câu, không tách từ ghép
        text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
        tokens = text.split()
    return [t for t in tokens if len(t) > 1]
