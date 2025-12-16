import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def track_balls_to_csv(video_path, hsv_range, num_balls, output_csv, 
                       nms_threshold=40, max_cost=150, visualize=False):
    """
    Trackea pelotas en video y exporta CSV con formato: x_ball1,y_ball1,x_ball2,y_ball2,...
    
    Args:
        video_path: ruta al video
        hsv_range: tupla (h_min, s_min, v_min, h_max, s_max, v_max)
        num_balls: número de pelotas a trackear
        output_csv: ruta del CSV de salida
        nms_threshold: distancia mínima entre detecciones (pixeles)
        max_cost: costo máximo para Hungarian (si supera, crea nuevo ID)
        visualize: mostrar tracking en tiempo real
    """
    cap = cv2.VideoCapture(video_path)
    h_min, s_min, v_min, h_max, s_max, v_max = hsv_range
    
    # Almacena posiciones: {ball_id: [(x,y), ...]}
    trajectories = {i: [] for i in range(num_balls)}
    prev_positions = None  # posiciones en frame anterior
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Segmentación por color HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), 
                                np.array([h_max, s_max, v_max]))
        
        # 2. Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Extraer centros y aplicar NMS
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append((cx, cy))
        
        # NMS: eliminar detecciones muy cercanas
        centers = non_max_suppression(centers, nms_threshold)
        centers = centers[:num_balls]  # limitar a num_balls
        
        # 4. Asociación con Hungarian (mantener IDs consistentes)
        if prev_positions is None:
            # Primer frame: asignar IDs en orden
            current_positions = [None] * num_balls
            for i, center in enumerate(centers):
                current_positions[i] = center
        else:
            current_positions = hungarian_assignment(prev_positions, centers, max_cost)
        
        # 5. Guardar posiciones (None si pelota no detectada)
        for ball_id in range(num_balls):
            pos = current_positions[ball_id] if ball_id < len(current_positions) else None
            trajectories[ball_id].append(pos)
        
        prev_positions = current_positions
        
        # Visualización opcional
        if visualize:
            vis_frame = frame.copy()
            for i, pos in enumerate(current_positions):
                if pos:
                    cv2.circle(vis_frame, pos, 8, (0, 255, 0), -1)
                    cv2.putText(vis_frame, f"B{i+1}", (pos[0]+10, pos[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow('Tracking', vis_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                break
        
        frame_idx += 1
    
    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    
    # 6. Exportar a CSV
    save_to_csv(trajectories, output_csv)
    print(f"CSV guardado en: {output_csv}")


def non_max_suppression(centers, threshold):
    """Elimina centros muy cercanos entre sí"""
    if len(centers) == 0:
        return []
    
    centers = sorted(centers, key=lambda c: c[1])  # ordenar por Y
    keep = []
    
    for c in centers:
        too_close = False
        for kept in keep:
            dist = np.linalg.norm(np.array(c) - np.array(kept))
            if dist < threshold:
                too_close = True
                break
        if not too_close:
            keep.append(c)
    
    return keep


def hungarian_assignment(prev_positions, new_centers, max_cost):
    """
    Asocia detecciones nuevas con IDs previos usando algoritmo Húngaro.
    Retorna lista de posiciones ordenada por ball_id (None si no detectada).
    """
    n_balls = len(prev_positions)
    n_detections = len(new_centers)
    
    # Matriz de costos (distancias euclídeas)
    cost_matrix = np.full((n_balls, n_detections), max_cost * 2, dtype=float)
    
    for i, prev_pos in enumerate(prev_positions):
        if prev_pos is None:
            continue
        for j, new_pos in enumerate(new_centers):
            dist = np.linalg.norm(np.array(prev_pos) - np.array(new_pos))
            cost_matrix[i, j] = dist
    
    # Resolver asignación
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Construir resultado
    result = [None] * n_balls
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < max_cost:  # solo asignar si costo razonable
            result[i] = new_centers[j]
    
    return result


def save_to_csv(trajectories, output_path):
    """Convierte trajectories a DataFrame y guarda CSV sin cabecera"""
    num_balls = len(trajectories)
    num_frames = len(trajectories[0])
    
    # Construir matriz: cada fila = frame, columnas = x1,y1,x2,y2,...
    data = []
    for frame_idx in range(num_frames):
        row = []
        for ball_id in range(num_balls):
            pos = trajectories[ball_id][frame_idx]
            if pos is None:
                row.extend([np.nan, np.nan])  # o usar 0, 0 si prefieres
            else:
                row.extend([pos[0], pos[1]])
        data.append(row)
    
    # Crear DataFrame sin cabecera
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, header=False)


# Ejemplo de uso
if __name__ == "__main__":
    # Configuración
    VIDEO_PATH = "/home/dar0j/Documentos/2025/intro trabajo titulo el E/PROJECT/Datasets/5b/5_(6x,4x)_12.mp4"
    HSV_RANGE =  (0, 140, 87, 9, 196, 203)  # Ajustar con hsv_color_picker.py
    NUM_BALLS = 5
    OUTPUT_CSV = "tracking_half5bR_id12.csv"
    
    track_balls_to_csv(
        video_path=VIDEO_PATH,
        hsv_range=HSV_RANGE,
        num_balls=NUM_BALLS,
        output_csv=OUTPUT_CSV,
        visualize=True  # cambiar a False para procesamiento rápido
    )