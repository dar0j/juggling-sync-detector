import os
import cv2
import pandas as pd
import numpy as np

for nBalls in [3, 4, 5, 6]:
    listfile = f"{nBalls}balls"
    if not os.path.exists(listfile):
        print(f"{listfile} not found, skipping.")
        continue
    with open(listfile, "r") as f:
        csv_files = [line.strip() for line in f if line.strip()]
    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            print(f"{csv_path} not found, skipping.")
            continue
        print(f"Mostrando: {csv_path}")
        recording = pd.read_csv(csv_path, header=None).values
        for i in range(recording.shape[0]):
            canvas = np.zeros((256,256,3), dtype=np.uint8)
            cv2.line(canvas, (recording[i,0]-10, recording[i,1]), (recording[i,0]+10, recording[i,1]), (0,255,0), 2)
            cv2.line(canvas, (recording[i,2]-10, recording[i,3]), (recording[i,2]+10, recording[i,3]), (0,0,255), 2)
            j_values = list(range(4, recording.shape[1], 2))
            max_index = len(j_values) - 1 if len(j_values) > 1 else 1
            for idx, j in enumerate(j_values):
                colorshift = int((idx / max_index) * 255) if max_index > 0 else 0
                cv2.circle(canvas, (recording[i,j], recording[i,j+1]), 10, (colorshift, 255 - colorshift, colorshift), 2)
            cv2.imshow('PlayPattern', canvas)
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break
        # Espera una tecla para pasar al siguiente archivo
        print("Presiona cualquier tecla para continuar con el siguiente archivo...")
        cv2.waitKey(0)
cv2.destroyAllWindows()