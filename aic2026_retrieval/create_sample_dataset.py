"""
Sinh 1 bộ dữ liệu MẪU NHỎ (vài video giả, mỗi video vài chục keyframe) đúng
format thật của dataset AIC -- dùng để kiểm tra toàn bộ pipeline chạy được
từ đầu đến cuối (build index -> dedup -> caption -> BM25 -> query) TRƯỚC KHI
chạy trên dataset thật.
"""

import argparse
import csv
import json
import os
import random

from PIL import Image, ImageDraw

import config
import utils

ROOT = "./sample_data"

SAMPLE_SCENES = [
    ("nguoi dan ong ao xanh dung noi cong vien", (40, 90, 200)),
    ("nguoi phu nu ao do phat bieu ngoai troi", (200, 40, 40)),
    ("hoc sinh mac dong phuc trang di bo", (230, 230, 230)),
    ("xe hoi mau vang dau tren duong pho", (230, 190, 30)),
    ("bau troi xanh voi may trang", (100, 180, 230)),
]


def make_dummy_image(path, text, color):
    img = Image.new("RGB", (384, 384), color=color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 180), text, fill=(255, 255, 255))
    img.save(path, quality=85)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-videos", type=int, default=3)
    parser.add_argument("--frames-per-video", type=int, default=20)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    dirs = {
        "Videos": os.path.join(ROOT, "Videos"),
        "Keyframes": os.path.join(ROOT, "Keyframes"),
        "Objects": os.path.join(ROOT, "Objects"),
        "CLIPFeatures": os.path.join(ROOT, "CLIPFeatures"),
        "Metadata": os.path.join(ROOT, "Metadata"),
        "map-keyframes": os.path.join(ROOT, "map-keyframes"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    random.seed(42)
    id_map_rows = []
    global_int_id = 0

    for vi in range(1, args.num_videos + 1):
        video_id = f"L01_V{vi:03d}"
        kf_dir = os.path.join(dirs["Keyframes"], video_id)
        obj_dir = os.path.join(dirs["Objects"], video_id)
        os.makedirs(kf_dir, exist_ok=True)
        os.makedirs(obj_dir, exist_ok=True)

        csv_rows = []
        cur_frame_idx = 0
        for n in range(1, args.frames_per_video + 1):
            text, color = random.choice(SAMPLE_SCENES)
            fname = f"{n:04d}.jpg"
            keyframe_path = os.path.join(kf_dir, fname)
            make_dummy_image(keyframe_path, text, color)

            cur_frame_idx += random.randint(15, 90)
            pts_time = round(cur_frame_idx / args.fps, 4)
            csv_rows.append({"n": n, "pts_time": pts_time, "fps": args.fps, "frame_idx": cur_frame_idx})

            # Lưu thông tin keyframe để xuất ra id_map.jsonl
            id_map_rows.append({
                "int_id": global_int_id,
                "video_id": video_id,
                "frame_name": fname,
                "frame_idx": cur_frame_idx,
                "pts_time": pts_time,
                "keyframe_path": keyframe_path
            })
            global_int_id += 1

            # Object detection giả
            with open(os.path.join(obj_dir, f"{n:04d}.json"), "w", encoding="utf-8") as f:
                json.dump({"detections": [
                    {"class": random.choice(["Person", "Car", "Sky", "Building"]), "score": round(random.uniform(0.4, 0.95), 2)}
                    for _ in range(random.randint(1, 3))
                ]}, f, ensure_ascii=False)

        with open(os.path.join(dirs["map-keyframes"], video_id + ".csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["n", "pts_time", "fps", "frame_idx"])
            writer.writeheader()
            writer.writerows(csv_rows)

        # Metadata giả
        with open(os.path.join(dirs["Metadata"], video_id + ".json"), "w", encoding="utf-8") as f:
            json.dump({
                "title": f"[SAMPLE] Video thu {vi} - ban tin thoi su",
                "description": "Day la metadata gia dung de test pipeline, khong phai du lieu that.",
            }, f, ensure_ascii=False)

        open(os.path.join(dirs["Videos"], video_id + ".mp4"), "wb").close()

        print(f"Đã sinh {video_id}: {args.frames_per_video} keyframe, "
              f"frame_idx cuối = {cur_frame_idx}")

    # === BƯỚC BỔ SUNG QUAN TRỌNG: Ghi id_map.jsonl vào thư mục INDEX_DIR ===
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    utils.write_jsonl(config.ID_MAP_PATH, id_map_rows)
    print(f"-> Đã tạo xong file id_map.jsonl tại: {config.ID_MAP_PATH} ({len(id_map_rows)} items)")

    print(f"\nXong. Dữ liệu mẫu nằm ở: {os.path.abspath(ROOT)}")


if __name__ == "__main__":
    main()
