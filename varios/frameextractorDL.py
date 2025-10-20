import cv2
import os
import glob
import csv
import numpy as np
import sys

sys.path.append("../../../old rasmus/juggling-vision-py")
from gridmodel import GridModel

for nBalls in [3, 4, 5, 6]:
    input_dir = f"../../Datasets/{nBalls}b"
    output_dir = f"60 fps/{nBalls}b frames 60"
    patternscsvfolder = f"{nBalls}b csv 60 128"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(patternscsvfolder, exist_ok=True)

    video_files = glob.glob(os.path.join(input_dir, "*.mp4")) # gif o mp4

    # Ajusta el path del modelo si tienes uno distinto por cantidad de bolas
    grid_model = GridModel(f"../../../old rasmus/grid_models/grid_model_submovavg_128x128.h5", nBalls=nBalls)

    for video_path in video_files:
        vid_name = os.path.splitext(os.path.basename(video_path))[0]
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        #frame_idx = 0
        annotations = []
        while success:
            #if frame_idx % 2 == 0: # skippear/alternar frames para que tome de un video de 60 fps muestras en 30 fps
            resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            out_path = os.path.join(output_dir, f"{vid_name}_frame{count:05d}.png")
            cv2.imwrite(out_path, resized)
            # Predice posiciones
            balls_and_hands = grid_model.predict(resized)
            # Guarda anotaciones en el formato esperado
            row = []
            row.extend(balls_and_hands["rhand"])
            row.extend(balls_and_hands["lhand"])
            row.extend(balls_and_hands["balls"].flatten())
            annotations.append(row)
            count += 1 # Desde aqui hay que indentar para usar el intercalador de frames para los videos de 60 fps (y sacar solo 30)
            success, image = vidcap.read()
            #frame_idx += 1
        vidcap.release()
        # Guarda las anotaciones en CSV
        with open(os.path.join(patternscsvfolder, f"{vid_name}_annotations.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            for row in annotations:
                writer.writerow(row)