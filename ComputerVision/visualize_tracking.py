#!/usr/bin/env python3
"""
visualize_tracking_csv.py
Visualiza en tiempo real las trayectorias de pelotas desde un CSV.
CSV: sin header, columnas x1,y1,x2,y2,x3,y3,...
"""

import numpy as np
import pandas as pd
import cv2
import argparse
import time
from pathlib import Path


def visualize_tracking_realtime(csv_path, fps=30, canvas_size=(800, 600), 
                                ball_radius=8, trail_length=30, 
                                start_ball=0, end_ball=None):
    """
    Visualiza tracking en tiempo real desde CSV.
    
    Args:
        csv_path: ruta al CSV
        fps: frames por segundo de reproducción
        canvas_size: tamaño del canvas (ancho, alto)
        ball_radius: radio de las pelotas en píxeles
        trail_length: longitud de la estela (frames anteriores)
        start_ball: índice de primera pelota a mostrar (0-indexed)
        end_ball: índice de última pelota a mostrar (None = todas)
    """
    # Cargar datos
    data = pd.read_csv(csv_path, header=None).values
    n_frames, n_cols = data.shape
    n_balls = n_cols // 2
    
    # Validar rango de pelotas
    if end_ball is None:
        end_ball = n_balls - 1
    
    start_ball = max(0, min(start_ball, n_balls - 1))
    end_ball = max(start_ball, min(end_ball, n_balls - 1))
    
    selected_balls = list(range(start_ball, end_ball + 1))
    n_selected = len(selected_balls)
    
    print(f"CSV: {csv_path}")
    print(f"Total frames: {n_frames}")
    print(f"Total pelotas: {n_balls}")
    print(f"Mostrando pelotas: {start_ball+1} a {end_ball+1} ({n_selected} pelotas)")
    print(f"FPS: {fps}")
    print(f"\nControles:")
    print("  ESPACIO: pausar/reanudar")
    print("  ESC: salir")
    print("  +/-: acelerar/desacelerar")
    print("  R: reiniciar")
    
    # Encontrar límites para normalización
    x_coords = []
    y_coords = []
    for i in selected_balls:
        x_coords.extend(data[:, i*2][data[:, i*2] != -1])
        y_coords.extend(data[:, i*2 + 1][data[:, i*2 + 1] != -1])
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # Margen
    margin = 50
    
    # Colores para cada pelota (BGR)
    colors = [
        (0, 0, 255),      # Rojo
        (0, 255, 0),      # Verde
        (255, 0, 0),      # Azul
        (0, 255, 255),    # Amarillo
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (128, 0, 255),    # Rosa
        (255, 128, 0),    # Azul claro
        (0, 128, 255),    # Naranja
        (128, 255, 0),    # Verde lima
        (255, 0, 128),    # Púrpura
        (0, 255, 128),    # Verde agua
    ]
    
    # Función de normalización
    def normalize(x, y):
        if x == -1 or y == -1:
            return None
        nx = int(margin + (x - x_min) / (x_max - x_min + 1) * (canvas_size[0] - 2*margin))
        ny = int(margin + (y - y_min) / (y_max - y_min + 1) * (canvas_size[1] - 2*margin))
        return (nx, ny)
    
    # Estado
    frame_idx = 0
    paused = False
    current_fps = fps
    
    # Buffer de trayectorias
    trajectories = {i: [] for i in selected_balls}
    
    while True:
        # Crear canvas
        canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        
        # Dibujar estelas
        for ball_id in selected_balls:
            color = colors[ball_id % len(colors)]
            trail = trajectories[ball_id]
            
            # Dibujar línea de estela
            for j in range(1, len(trail)):
                if trail[j-1] is not None and trail[j] is not None:
                    alpha = j / len(trail)  # fade out
                    thickness = max(1, int(3 * alpha))
                    cv2.line(canvas, trail[j-1], trail[j], 
                            tuple(int(c * alpha * 0.5) for c in color), 
                            thickness)
        
        # Dibujar pelotas actuales
        if frame_idx < n_frames:
            for ball_id in selected_balls:
                x = data[frame_idx, ball_id * 2]
                y = data[frame_idx, ball_id * 2 + 1]
                
                pos = normalize(x, y)
                
                if pos is not None:
                    color = colors[ball_id % len(colors)]
                    
                    # Pelota
                    cv2.circle(canvas, pos, ball_radius, color, -1)
                    cv2.circle(canvas, pos, ball_radius + 2, (255, 255, 255), 1)
                    
                    # Etiqueta
                    label = f"B{ball_id + 1}"
                    cv2.putText(canvas, label, 
                               (pos[0] + ball_radius + 5, pos[1] + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Actualizar trayectoria
                    trajectories[ball_id].append(pos)
                    if len(trajectories[ball_id]) > trail_length:
                        trajectories[ball_id].pop(0)
                else:
                    trajectories[ball_id].append(None)
                    if len(trajectories[ball_id]) > trail_length:
                        trajectories[ball_id].pop(0)
        
        # Info overlay
        info_text = f"Frame: {frame_idx}/{n_frames} | FPS: {current_fps:.1f}"
        if paused:
            info_text += " [PAUSED]"
        
        cv2.putText(canvas, info_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Leyenda de colores
        y_offset = 50
        for i, ball_id in enumerate(selected_balls):
            color = colors[ball_id % len(colors)]
            label = f"Ball {ball_id + 1}"
            cv2.circle(canvas, (15, y_offset + i*25), 8, color, -1)
            cv2.putText(canvas, label, (30, y_offset + i*25 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Mostrar
        cv2.imshow('Tracking Visualization', canvas)
        
        # Control de tiempo
        delay = int(1000 / current_fps) if not paused else 0
        key = cv2.waitKey(delay) & 0xFF
        
        # Controles
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # ESPACIO
            paused = not paused
        elif key == ord('+') or key == ord('='):
            current_fps = min(120, current_fps * 1.5)
            print(f"FPS: {current_fps:.1f}")
        elif key == ord('-') or key == ord('_'):
            current_fps = max(1, current_fps / 1.5)
            print(f"FPS: {current_fps:.1f}")
        elif key == ord('r'):  # Reiniciar
            frame_idx = 0
            trajectories = {i: [] for i in selected_balls}
            print("Reiniciado")
        
        # Avanzar frame
        if not paused:
            frame_idx += 1
            if frame_idx >= n_frames:
                print("Fin del video. Reiniciando...")
                frame_idx = 0
                trajectories = {i: [] for i in selected_balls}
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizar tracking CSV en tiempo real')
    parser.add_argument('csv', help='Ruta al archivo CSV')
    parser.add_argument('--fps', type=float, default=30, help='FPS de reproducción')
    parser.add_argument('--size', type=int, nargs=2, default=[800, 600], 
                       help='Tamaño canvas (ancho alto)')
    parser.add_argument('--radius', type=int, default=8, help='Radio de pelotas')
    parser.add_argument('--trail', type=int, default=30, help='Longitud de estela')
    parser.add_argument('--start-ball', type=int, default=0, 
                       help='Primera pelota a mostrar (0-indexed, ej: 5 para pelota 6)')
    parser.add_argument('--end-ball', type=int, default=None, 
                       help='Última pelota a mostrar (0-indexed, ej: 11 para pelota 12)')
    
    args = parser.parse_args()
    
    visualize_tracking_realtime(
        csv_path=args.csv,
        fps=args.fps,
        canvas_size=tuple(args.size),
        ball_radius=args.radius,
        trail_length=args.trail,
        start_ball=args.start_ball,
        end_ball=args.end_ball
    )