import os
import re

frames_dir = "extracted_frames"
frame_files = os.listdir(frames_dir)

# Diccionario para guardar el máximo índice de frame por video
max_frames = {}

for fname in frame_files:
    match = re.match(r"(.+)_frame_(\d+)\.png", fname)
    if match:
        video = match.group(1)
        frame_num = int(match.group(2))
        if video not in max_frames or frame_num > max_frames[video]:
            max_frames[video] = frame_num

# Mostrar resultados
for video, last_frame in max_frames.items():
    print(f"{video}: último frame = {last_frame}")