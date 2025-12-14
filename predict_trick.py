"""
Script de inferencia end-to-end para clasificación de trucos de malabarismo.

Uso:
    python predict_trick.py --video path/to/video.mp4 --model checkpoints/fold_3_best.h5
    python predict_trick.py --video path/to/video.mp4 --fps 30 --create_overlay result.mp4

Output:
    Truco predicho con probabilidad
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import json
import sys
import os

# Importar GridModel (ajustar path si es necesario)
sys.path.append("./")
from rasmus.gridmodel import GridModel

# Parámetros del modelo (deben coincidir con entrenamiento)
MAX_BALLS = 6
HAND_FEATS = 4
MAX_FEATURES = HAND_FEATS + MAX_BALLS * 2 + 1  # 17
MASK_VALUE = -1.0
IMAGE_SIZE = 256  # GridModel espera 256x256
TARGET_FPS = 60  # muestrear a 60 fps (default)

def extract_coordinates_from_video(video_path, grid_model, target_fps=60, max_frames=None):
    """
    Extrae coordenadas de manos y pelotas de un video usando GridModel.
    
    Args:
        video_path: ruta al video MP4
        grid_model: instancia de GridModel
        target_fps: fps objetivo (submuestrea si el video tiene más)
        max_frames: máximo de frames a procesar (None = todos)
    
    Returns:
        np.array: (n_frames, 4 + 2*n_balls) coordenadas
        int: número de pelotas detectadas
    """
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    
    original_fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(original_fps / target_fps))
    
    annotations = []
    frame_idx = 0
    processed = 0
    
    print(f"Procesando video: {os.path.basename(video_path)}")
    print(f"FPS original: {original_fps:.1f}, submuestreo cada {frame_skip} frames")
    
    while True:
        success, image = vidcap.read()
        if not success:
            break
        
        # Submuestrear frames
        if frame_idx % frame_skip == 0:
            # Redimensionar y hacer cuadrada (crop central)
            h, w = image.shape[:2]
            if w > h:
                crop = (w - h) // 2
                image = image[:, crop:crop+h]
            elif h > w:
                crop = (h - w) // 2
                image = image[crop:crop+w, :]
            
            resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            
            # Predecir posiciones
            balls_and_hands = grid_model.predict(resized)
            
            # Formatear: [x_rhand, y_rhand, x_lhand, y_lhand, x_ball1, y_ball1, ...]
            row = []
            row.extend(balls_and_hands["rhand"])
            row.extend(balls_and_hands["lhand"])
            row.extend(balls_and_hands["balls"].flatten())
            annotations.append(row)
            
            processed += 1
            if processed % 30 == 0:
                print(f"  Procesados {processed} frames...")
            
            if max_frames and processed >= max_frames:
                break
        
        frame_idx += 1
    
    vidcap.release()
    
    if len(annotations) == 0:
        raise ValueError("No se extrajeron coordenadas del video")
    
    coords = np.array(annotations)
    n_balls = (coords.shape[1] - 4) // 2  # (total_cols - 4_manos) / 2
    
    print(f"✓ Extraídos {len(annotations)} frames con {n_balls} pelotas detectadas")
    return coords, n_balls


def preprocess_sequence(coords, n_balls):
    """
    Preprocesa secuencia de coordenadas para modelo TCN.
    
    Args:
        coords: (n_frames, 4 + 2*n_balls)
        n_balls: número de pelotas
    
    Returns:
        np.array: (1, n_frames, 17) tensor listo para modelo
    """
    n_frames = coords.shape[0]
    seq = np.full((n_frames, MAX_FEATURES), MASK_VALUE, dtype=np.float32)
    
    # Copiar coordenadas reales
    real_cols = 4 + 2 * n_balls
    seq[:, :real_cols] = coords[:, :real_cols]
    
    # Ball count normalizado
    seq[:, -1] = n_balls / MAX_BALLS
    
    # Normalización min-max (igual que en entrenamiento)
    real_mask = np.any(seq[:, :HAND_FEATS] != MASK_VALUE, axis=1)
    real_frames = seq[real_mask, :real_cols]
    
    if real_frames.size == 0:
        raise ValueError("No se encontraron frames válidos en la secuencia")
    
    minv = real_frames.min(axis=0, keepdims=True)
    maxv = real_frames.max(axis=0, keepdims=True)
    rangev = maxv - minv
    rangev[rangev == 0] = 1.0
    scaled = (real_frames - minv) / rangev
    
    seq[real_mask, :real_cols] = scaled
    
    # Expandir dimensión de batch: (1, n_frames, 17)
    return np.expand_dims(seq, axis=0)


def predict_trick(model, preprocessed_seq, label_map, top_k=3):
    """
    Predice truco usando modelo entrenado.
    
    Args:
        model: modelo Keras cargado
        preprocessed_seq: (1, n_frames, 17)
        label_map: dict {trick_name: label_id}
        top_k: número de predicciones top a retornar
    
    Returns:
        list: [(trick_name, probability), ...] ordenado por probabilidad
    """
    # Predicción
    probs = model.predict(preprocessed_seq, verbose=0)[0]  # (27,)
    
    # Invertir label_map
    idx_to_label = {v: k for k, v in label_map.items()}
    
    # Top-k predicciones
    top_indices = np.argsort(probs)[::-1][:top_k]
    predictions = [
        (idx_to_label[idx], float(probs[idx]))
        for idx in top_indices
    ]
    
    return predictions


def create_overlay_video(video_path, output_path, grid_model, predictions, fps=30):
    """
    Crea video con overlay de detecciones y predicción.
    
    Args:
        video_path: ruta al video original
        output_path: ruta para guardar video con overlay
        grid_model: instancia de GridModel para detectar posiciones
        predictions: lista [(trick, prob), ...]
        fps: fps objetivo para procesamiento
    """
    vidcap = cv2.VideoCapture(video_path)
    original_fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(original_fps / fps))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nGenerando video con overlay...")
    print(f"  Resolución: {width}x{height}")
    print(f"  FPS original: {original_fps:.1f}, output: {fps}")
    
    # VideoWriter para guardar
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    best_trick, best_prob = predictions[0]
    
    frame_idx = 0
    written = 0
    
    while True:
        success, frame = vidcap.read()
        if not success:
            break
        
        if frame_idx % frame_skip == 0:
            # Preparar imagen cuadrada para GridModel
            h, w = frame.shape[:2]
            if w > h:
                crop = (w - h) // 2
                square = frame[:, crop:crop+h]
                offset_x = crop
                offset_y = 0
            elif h > w:
                crop = (h - w) // 2
                square = frame[crop:crop+w, :]
                offset_x = 0
                offset_y = crop
            else:
                square = frame.copy()
                offset_x = 0
                offset_y = 0
            
            resized = cv2.resize(square, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            
            # Detectar posiciones
            detections = grid_model.predict(resized)
            
            # Escalar coordenadas de vuelta al tamaño original
            square_size = min(w, h)
            scale = square_size / IMAGE_SIZE
            
            # Dibujar manos
            rhand = detections["rhand"]
            lhand = detections["lhand"]
            
            rx = int(rhand[0] * scale + offset_x)
            ry = int(rhand[1] * scale + offset_y)
            lx = int(lhand[0] * scale + offset_x)
            ly = int(lhand[1] * scale + offset_y)
            
            cv2.circle(frame, (rx, ry), 10, (0, 255, 0), -1)  # Verde - mano derecha
            cv2.circle(frame, (lx, ly), 10, (255, 255, 0), -1)  # Cyan - mano izquierda
            cv2.putText(frame, "R", (rx+12, ry+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "L", (lx+12, ly+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Dibujar pelotas
            balls = detections["balls"].reshape(-1, 2)
            for i, ball in enumerate(balls):
                bx = int(ball[0] * scale + offset_x)
                by = int(ball[1] * scale + offset_y)
                cv2.circle(frame, (bx, by), 8, (0, 0, 255), -1)  # Rojo
                cv2.putText(frame, str(i+1), (bx+10, by+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Overlay con predicción (panel superior semi-transparente)
            overlay = frame.copy()
            panel_height = 150
            cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Texto principal
            cv2.putText(frame, f"Truco: {best_trick}", (15, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
            cv2.putText(frame, f"Confianza: {best_prob:.1%}", (15, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            
            # Top-3 predicciones pequeñas
            y_pos = 120
            for i, (trick, prob) in enumerate(predictions[:3]):
                text = f"{i+1}. {trick} ({prob:.1%})"
                cv2.putText(frame, text, (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                y_pos += 30
            
            out.write(frame)
            written += 1
            
            if written % 30 == 0:
                print(f"  Escritos {written} frames...")
        
        frame_idx += 1
    
    vidcap.release()
    out.release()
    print(f"✓ Video guardado: {output_path} ({written} frames)")


def main():
    parser = argparse.ArgumentParser(description="Clasificador de trucos de malabarismo")
    parser.add_argument("--video", required=True, help="Ruta al video MP4")
    parser.add_argument("--model", default="fold_3_best.h5", help="Ruta al modelo entrenado")
    parser.add_argument("--grid_model", default="rasmus/grid_models/grid_model_submovavg_128x128.h5", 
                        help="Ruta al modelo GridModel")
    parser.add_argument("--label_map", default="label_map.json", help="Ruta al label_map.json")
    parser.add_argument("--top_k", type=int, default=3, help="Número de predicciones top a mostrar")
    parser.add_argument("--fps", type=int, default=60, help="FPS para procesamiento (default: 60)")
    parser.add_argument("--max_frames", type=int, default=None, help="Máximo de frames a procesar")
    parser.add_argument("--save_csv", default=None, help="Guardar coordenadas en CSV")
    parser.add_argument("--create_overlay", default=None, help="Crear video con overlay (ruta output)")
    
    args = parser.parse_args()
    
    # Verificar archivos
    if not os.path.exists(args.video):
        print(f"❌ Video no encontrado: {args.video}")
        return
    if not os.path.exists(args.model):
        print(f"❌ Modelo no encontrado: {args.model}")
        return
    if not os.path.exists(args.label_map):
        print(f"❌ Label map no encontrado: {args.label_map}")
        return
    
    print("\n" + "="*60)
    print("CLASIFICADOR DE TRUCOS DE MALABARISMO")
    print("="*60 + "\n")
    
    # 1. Cargar GridModel
    print("[1/5] Cargando GridModel...")
    try:
        # Detectar número de pelotas del video (asumiendo nombre formato N_trick.mp4)
        basename = os.path.basename(args.video)
        try:
            n_balls_hint = int(basename[0])
        except:
            n_balls_hint = 3  # default
        
        grid_model = GridModel(args.grid_model, nBalls=n_balls_hint)
        print(f"✓ GridModel cargado (configurado para {n_balls_hint} pelotas)")
    except Exception as e:
        print(f"❌ Error cargando GridModel: {e}")
        return
    
    # 2. Extraer coordenadas
    print(f"\n[2/5] Extrayendo coordenadas del video (FPS: {args.fps})...")
    try:
        coords, n_balls = extract_coordinates_from_video(
            args.video, grid_model, 
            target_fps=args.fps,
            max_frames=args.max_frames
        )
        
        # Guardar CSV si se solicita
        if args.save_csv:
            pd.DataFrame(coords).to_csv(args.save_csv, header=False, index=False)
            print(f"  → Coordenadas guardadas en {args.save_csv}")
    
    except Exception as e:
        print(f"❌ Error extrayendo coordenadas: {e}")
        return
    
    # 3. Preprocesar secuencia
    print("\n[3/5] Preprocesando secuencia...")
    try:
        preprocessed = preprocess_sequence(coords, n_balls)
        print(f"✓ Secuencia preprocesada: {preprocessed.shape}")
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {e}")
        return
    
    # 4. Cargar modelo y label_map
    print("\n[4/5] Cargando modelo clasificador...")
    try:
        from tensorflow.keras.models import load_model
        model = load_model(args.model)
        print(f"✓ Modelo cargado: {args.model}")
        
        with open(args.label_map, "r") as f:
            label_map = json.load(f)
        print(f"✓ Label map cargado: {len(label_map)} clases")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # 5. Predecir
    print("\n[5/5] Clasificando truco...")
    try:
        predictions = predict_trick(model, preprocessed, label_map, top_k=args.top_k)
        
        print("\n" + "="*60)
        print("RESULTADOS")
        print("="*60)
        for i, (trick, prob) in enumerate(predictions, 1):
            bar = "█" * int(prob * 40)
            print(f"{i}. {trick:30s} {prob:6.2%} {bar}")
        
        # Mejor predicción
        best_trick, best_prob = predictions[0]
        print("\n" + "="*60)
        print(f"🎯 TRUCO PREDICHO: {best_trick.upper()}")
        print(f"   Confianza: {best_prob:.1%}")
        print("="*60 + "\n")
        
        # Crear video con overlay si se solicita
        if args.create_overlay:
            try:
                create_overlay_video(
                    args.video, 
                    args.create_overlay, 
                    grid_model, 
                    predictions, 
                    fps=args.fps
                )
            except Exception as e:
                print(f"❌ Error creando video overlay: {e}")
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return


if __name__ == "__main__":
    main()