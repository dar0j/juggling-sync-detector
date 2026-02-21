#!/usr/bin/env python3
"""
pipeline_detect.py
Detecta pelotas con YOLO en todos los videos de las carpetas "already *"
y guarda detecciones en cache NDJSON.

Uso:
  python pipeline_detect.py --model "100ep 1800imgs/best.pt" \
    --source "../Datasets/used to track" \
    --out_dir runs/dets_cache_all \
    --conf 0.25 --iou 0.7
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def find_videos(source_dir: Path):
    """Busca todos los videos en carpetas 'already *' recursivamente."""
    videos = []
    for folder in sorted(source_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("already"):
            for f in sorted(folder.rglob("*")):
                if f.suffix.lower() in VIDEO_EXTS:
                    videos.append(f)
    return videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="100ep 1800imgs/best.pt")
    ap.add_argument("--source", default="../Datasets/used to track")
    ap.add_argument("--out_dir", default="runs/dets_cache_all")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    videos = find_videos(Path(args.source))
    print(f"Encontrados {len(videos)} videos en carpetas 'already *'")

    for vi, video_path in enumerate(videos, 1):
        out_path = out_dir / f"{video_path.stem}.ndjson"
        if args.skip_existing and out_path.exists():
            print(f"[{vi}/{len(videos)}] SKIP: {video_path.name}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[{vi}/{len(videos)}] ERROR: no se pudo abrir {video_path}")
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        predict_kwargs = dict(
            source=str(video_path),
            stream=True,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )
        if args.imgsz is not None:
            predict_kwargs["imgsz"] = args.imgsz

        with out_path.open("w", encoding="utf-8") as f:
            meta = {
                "video": str(video_path),
                "fps": fps, "w": w, "h": h,
                "folder": video_path.parent.name,
            }
            f.write(json.dumps({"meta": meta}) + "\n")

            frame_i = 0
            for r in model.predict(**predict_kwargs):
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    dets = []
                else:
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    conf = boxes.conf.detach().cpu().numpy()
                    cls = boxes.cls.detach().cpu().numpy()
                    dets = [
                        [float(x1), float(y1), float(x2), float(y2), float(s), int(c)]
                        for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls)
                    ]
                f.write(json.dumps({"frame": frame_i, "dets": dets}) + "\n")
                frame_i += 1

        print(f"[{vi}/{len(videos)}] OK: {video_path.name} -> {out_path.name} ({frame_i} frames)")

    print(f"\nListo. {len(videos)} videos procesados en {out_dir}")


if __name__ == "__main__":
    main()