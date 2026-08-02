"""
Hàm tiện ích dùng chung: quét cấu trúc dataset AIC, đọc/ghi jsonl,
tokenizer tiếng Việt cho BM25.
"""

import os
import json
import re
from dataclasses import dataclass, asdict
from typing import Iterator, List, Optional

import config


@dataclass
class KeyframeItem:
    """Một keyframe = 1 đơn vị trong index."""
    int_id: int          # id nội bộ, tăng dần, dùng làm hàng trong FAISS
    video_id: str         # vd "L01_V001"
    frame_id: int          # chỉ số frame trong video gốc (đọc từ metadata do BTC cấp)
    keyframe_path: str    # đường dẫn ảnh keyframe trên đĩa
    object_json_path: Optional[str] = None
    stem: str = ""         # tên file không đuôi, vd "0000"


def _read_frame_index_metadata(video_kf_dir: str) -> dict:
    """
    Đọc file metadata ánh xạ tên keyframe -> frame_id thật trong video gốc.
    BTC quy định: 'vị trí (frame index) tương ứng của mỗi keyframe được ghi
    trong file metadata'. Tuỳ format thực tế BTC cấp (thường là
    <video_id>.json hoặc map.json trong chính thư mục keyframe), chỉnh lại
    hàm này cho khớp. Ở đây hỗ trợ 2 dạng phổ biến:
      1) file "<video_kf_dir>/map.json"  -> {"0000": 1523, "0001": 1601, ...}
      2) nếu không có map.json, fallback: dùng chính tên file (int(stem))
         làm frame_id (chỉ đúng nếu keyframe được đặt tên = frame index thật)
    """
    map_path = os.path.join(video_kf_dir, "map.json")
    if os.path.isfile(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}  # fallback sẽ xử lý ở iter_all_keyframes()


def iter_all_keyframes() -> Iterator[KeyframeItem]:
    """
    Duyệt toàn bộ Keyframes/<video_id>/*.jpg theo đúng thứ tự tăng dần,
    sinh ra KeyframeItem cho từng ảnh. int_id sinh tuần tự -> dùng để
    map ngược lại (video_id, frame_id) sau khi search FAISS.
    """
    int_id = 0
    video_ids = sorted(os.listdir(config.KEYFRAMES_DIR))
    for video_id in video_ids:
        video_kf_dir = os.path.join(config.KEYFRAMES_DIR, video_id)
        if not os.path.isdir(video_kf_dir):
            continue

        frame_map = _read_frame_index_metadata(video_kf_dir)
        obj_dir = os.path.join(config.OBJECTS_DIR, video_id)

        filenames = sorted(
            f for f in os.listdir(video_kf_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        for fname in filenames:
            stem = os.path.splitext(fname)[0]
            if stem in frame_map:
                frame_id = int(frame_map[stem])
            else:
                # fallback: tên file chính là frame index (chỉnh nếu dataset khác)
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
