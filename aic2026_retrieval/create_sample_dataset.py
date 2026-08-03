"""
Sinh 1 bộ dữ liệu MẪU NHỎ (vài video giả, mỗi video vài chục keyframe) đúng
format thật của dataset AIC -- dùng để kiểm tra toàn bộ pipeline chạy được
từ đầu đến cuối (build index -> dedup -> caption -> BM25 -> query) TRƯỚC KHI
chạy trên dataset thật (tránh mất hàng giờ mới phát hiện lỗi cấu hình).

Lưu ý: ảnh sinh ra là ảnh giả (màu + chữ ngẫu nhiên), CHỈ để test luồng chạy
(pipeline wiring) chứ không phản ánh chất lượng search thật -- muốn test độ
chính xác phải dùng dữ liệu thật.

Chạy:
    python create_sample_dataset.py --num-videos 3 --frames-per-video 20

Sau đó chạy pipeline với config mẫu:
    AIC_CONFIG=configs/sample.yaml python build_dense_index.py --limit 60
    AIC_CONFIG=configs/sample.yaml python dedup_keyframes.py
    AIC_CONFIG=configs/sample.yaml python build_captions.py --dedup-map --max-new-tokens 32
    AIC_CONFIG=configs/sample.yaml python build_bm25_index.py
    AIC_CONFIG=configs/sample.yaml python query_cli.py --query "người mặc áo đỏ"
"""

import argparse
import csv
import json
import os
import random

from PIL import Image, ImageDraw

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

    for vi in range(1, args.num_videos + 1):
        video_id = f"L01_V{vi:03d}"
        kf_dir = os.path.join(dirs["Keyframes"], video_id)
        obj_dir = os.path.join(dirs["Objects"], video_id)
        os.makedirs(kf_dir, exist_ok=True)
        os.makedirs(obj_dir, exist_ok=True)

        # --- sinh keyframe ảnh giả + map-keyframes CSV (đúng format thật) ---
        csv_rows = []
        cur_frame_idx = 0
        for n in range(1, args.frames_per_video + 1):
            text, color = random.choice(SAMPLE_SCENES)
            fname = f"{n:04d}.jpg"
            make_dummy_image(os.path.join(kf_dir, fname), text, color)

            cur_frame_idx += random.randint(15, 90)  # mô phỏng keyframe cách nhau ~15-90 frame
            pts_time = round(cur_frame_idx / args.fps, 4)
            csv_rows.append({"n": n, "pts_time": pts_time, "fps": args.fps, "frame_idx": cur_frame_idx})

            # object detection giả (format "detections")
            with open(os.path.join(obj_dir, f"{n:04d}.json"), "w", encoding="utf-8") as f:
                json.dump({"detections": [
                    {"class": random.choice(["Person", "Car", "Sky", "Building"]), "score": round(random.uniform(0.4, 0.95), 2)}
                    for _ in range(random.randint(1, 3))
                ]}, f, ensure_ascii=False)

        with open(os.path.join(dirs["map-keyframes"], video_id + ".csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["n", "pts_time", "fps", "frame_idx"])
            writer.writeheader()
            writer.writerows(csv_rows)

        # --- metadata giả ---
        with open(os.path.join(dirs["Metadata"], video_id + ".json"), "w", encoding="utf-8") as f:
            json.dump({
                "title": f"[SAMPLE] Video thu {vi} - ban tin thoi su",
                "description": "Day la metadata gia dung de test pipeline, khong phai du lieu that.",
            }, f, ensure_ascii=False)

        # video thật không cần thiết cho pipeline retrieval (chỉ cần keyframe),
        # tạo file rỗng placeholder để giữ đúng cấu trúc thư mục
        open(os.path.join(dirs["Videos"], video_id + ".mp4"), "wb").close()

        print(f"Đã sinh {video_id}: {args.frames_per_video} keyframe, "
              f"frame_idx cuối = {cur_frame_idx}")

    print(f"\nXong. Dữ liệu mẫu nằm ở: {os.path.abspath(ROOT)}")
    print("Chạy tiếp với: AIC_CONFIG=configs/sample.yaml python build_dense_index.py")


if __name__ == "__main__":
    main()
