import cv2
import os
import glob
import csv
import numpy as np
import sys

sys.path.append("../../../old rasmus/juggling-vision-py")
from gridmodel import GridModel

for nBalls in [3, 4, 5, 6]:
    input_dir = f"60 fps/{nBalls}b frames 60"
    patternscsvfolder = f"CSVs/60fps64/{nBalls}b csv 60 64"
    os.makedirs(patternscsvfolder, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    grid_model = GridModel(f"../../../old rasmus/grid_models/grid_model_submovavg_64x64.h5", nBalls=nBalls)

    # Agrupa los frames por video usando el nombre base antes de "_frame"
    videos = {}
    for frame_path in frame_files:
        base = os.path.basename(frame_path)
        vid_name = base.split("_frame")[0]
        videos.setdefault(vid_name, []).append(frame_path)

    for vid_name, frames in videos.items():
        annotations = []
        frames = sorted(frames)  # Asegura orden correcto
        for frame_path in frames:
            image = cv2.imread(frame_path)
            if image is None:
                continue
            #resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            balls_and_hands = grid_model.predict(image)
            row = []
            row.extend(balls_and_hands["rhand"])
            row.extend(balls_and_hands["lhand"])
            row.extend(balls_and_hands["balls"].flatten())
            annotations.append(row)

        with open(os.path.join(patternscsvfolder, f"{vid_name}_annotations.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            for row in annotations:
                writer.writerow(row)