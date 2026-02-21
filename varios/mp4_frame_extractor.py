import cv2
import os

video_path = "../../../PROJECT/Datasets/used to track/already nico/3_(2x,4x)_2.mp4"
vidcap = cv2.VideoCapture(video_path)

#save individual frames in FRAMES folder
os.makedirs("FRAMES", exist_ok=True)
frame_count = 0
success, frame = vidcap.read()
while success:
    cv2.imwrite(f"FRAMES/frame_{frame_count:3d}.png", frame)
    success, frame = vidcap.read()
    frame_count += 1



