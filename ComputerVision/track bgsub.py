import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def track_balls_bg_subtraction(video_path, num_balls, output_csv, 
                                min_area=100, max_cost=150, visualize=False):
    """
    Trackea pelotas usando sustracción de fondo + Hungarian.
    
    Args:
        video_path: ruta al video
        num_balls: número de pelotas a trackear
        output_csv: ruta del CSV de salida
        min_area: área mínima de contorno (píxeles²)
        max_cost: umbral para Hungarian assignment
        visualize: mostrar tracking en tiempo real
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Crear sustractor de fondo (MOG2 es el mejor para objetos en movimiento)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=100,           # frames para modelo de fondo
        varThreshold=40,       # sensibilidad (menor = más sensible)
        detectShadows=False    # desactivar detección de sombras (más rápido)
    )
    
    # Pre-alocar trayectorias
    trajectories = np.full((total_frames, num_balls, 2), np.nan, dtype=np.float32)
    prev_positions = None
    
    # Kernel para operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    frame_idx = 0
    print(f"Procesando {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Aplicar sustracción de fondo
        fg_mask = bg_subtractor.apply(frame)
        
        # 2. Limpiar máscara con morfología
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)   # eliminar ruido
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # cerrar huecos
        
        # 3. Encontrar contornos
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4. Extraer centros filtrados por área
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append((cx, cy, area))
        
        # Ordenar por área (descendente) y tomar top N
        centers = sorted(centers, key=lambda x: x[2], reverse=True)[:num_balls]
        centers = [(x, y) for x, y, _ in centers]
        
        # 5. Hungarian assignment para mantener IDs consistentes
        if prev_positions is None:
            # Primer frame
            current_positions = [None] * num_balls
            for i, center in enumerate(centers):
                current_positions[i] = center
        else:
            current_positions = hungarian_assignment(prev_positions, centers, max_cost)
        
        # 6. Guardar posiciones
        for ball_id in range(num_balls):
            pos = current_positions[ball_id]
            if pos is not None:
                trajectories[frame_idx, ball_id, 0] = pos[0]
                trajectories[frame_idx, ball_id, 1] = pos[1]
        
        prev_positions = current_positions
        
        # Visualización
        if visualize:
            vis = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            for i, pos in enumerate(current_positions):
                if pos:
                    pos_s = (pos[0]//2, pos[1]//2)
                    cv2.circle(vis, pos_s, 5, (0, 255, 0), -1)
                    cv2.putText(vis, str(i+1), (pos_s[0]+8, pos_s[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Tracking', vis)
            cv2.imshow('Mask', cv2.resize(fg_mask, (0, 0), fx=0.5, fy=0.5))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total_frames} frames...")
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Guardar CSV
    data = trajectories[:frame_idx].reshape(frame_idx, num_balls * 2)
    np.savetxt(output_csv, data, delimiter=',', fmt='%.1f')
    print(f"CSV guardado: {output_csv}")


def hungarian_assignment(prev_positions, new_centers, max_cost):
    """Hungarian assignment vectorizado"""
    n_balls = len(prev_positions)
    n_detections = len(new_centers)
    
    if n_detections == 0:
        return [None] * n_balls
    
    cost_matrix = np.full((n_balls, n_detections), max_cost * 2, dtype=np.float32)
    
    for i, prev_pos in enumerate(prev_positions):
        if prev_pos is None:
            continue
        prev_arr = np.array(prev_pos, dtype=np.float32)
        centers_arr = np.array(new_centers, dtype=np.float32)
        dists = np.linalg.norm(centers_arr - prev_arr, axis=1)
        cost_matrix[i, :] = dists
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    result = [None] * n_balls
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < max_cost:
            result[i] = new_centers[j]
    
    return result


if __name__ == "__main__":
    VIDEO_PATH = "/home/dar0j/Documentos/2025/intro trabajo titulo el E/PROJECT/Datasets/5b/5_(6x,4x)_12.mp4"
    NUM_BALLS = 5
    OUTPUT_CSV = "tracking_bg_5b_id12.csv"
    
    track_balls_bg_subtraction(
        video_path=VIDEO_PATH,
        num_balls=NUM_BALLS,
        output_csv=OUTPUT_CSV,
        min_area=50,        # Ajustar según tamaño de pelotas en píxeles
        max_cost=150,       # Máxima distancia para asociar pelota entre frames
        visualize=True      # Ver tracking en tiempo real
    )