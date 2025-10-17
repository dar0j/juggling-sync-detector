import time
import cv2

cap = cv2.VideoCapture(1)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera FPS: {fps}")
frame_count = 0
start = time.time()
while frame_count < 100:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
end = time.time()
cap.release()
actual_fps = frame_count / (end - start)
print(f"Actual FPS: {actual_fps:.2f}")