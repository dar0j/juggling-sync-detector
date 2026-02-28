"""
Paso 2: Tunea hiperparámetros de BoxMOTJugglingTracker usando Optuna.

Usa el dataset MOT creado por prepare_gt_dataset.py para evaluar
diferentes configuraciones de preprocesamiento + tracker.

Uso:
    python boxmot_tuner.py --mot_dir datasets/juggling-mot --tracker_type ocsort --n_trials 50
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import optuna
from tqdm import tqdm
import yaml
import json

from boxmot_tracker import BoxMOTJugglingTracker


def load_mot_gt(gt_path: str) -> pd.DataFrame:
    """
    Carga GT en formato MOT.
    Returns DataFrame con: frame, id, bb_left, bb_top, bb_width, bb_height, ...
    """
    df = pd.read_csv(gt_path, header=None, 
                     names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height',
                            'conf', 'cls', 'visibility'])
    return df


def gt_to_centers_per_frame(gt_df: pd.DataFrame) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """
    Convierte GT MOT a dict: {frame: {ball_id: (cx, cy)}}
    """
    result = {}
    for _, row in gt_df.iterrows():
        frame = int(row['frame'])
        ball_id = int(row['id'])
        cx = row['bb_left'] + row['bb_width'] / 2
        cy = row['bb_top'] + row['bb_height'] / 2
        
        if frame not in result:
            result[frame] = {}
        result[frame][ball_id] = (cx, cy)
    
    return result


def compute_mota(predictions: Dict[int, Dict[int, Tuple[float, float]]],
                 ground_truth: Dict[int, Dict[int, Tuple[float, float]]],
                 distance_threshold: float = 30.0) -> Dict[str, float]:
    """
    Calcula MOTA simplificado usando distancia euclidiana.
    
    Args:
        predictions: {frame: {track_id: (cx, cy)}}
        ground_truth: {frame: {gt_id: (cx, cy)}}
        distance_threshold: distancia máxima para considerar match (px)
    
    Returns:
        Dict con MOTA, num_misses, num_fps, num_switches
    """
    from scipy.optimize import linear_sum_assignment
    
    total_gt = 0
    total_fp = 0
    total_miss = 0
    total_switch = 0
    
    # Mapping GT_id -> last assigned track_id
    gt_to_track = {}
    
    all_frames = sorted(set(list(ground_truth.keys()) + list(predictions.keys())))
    
    for frame in all_frames:
        gt_objects = ground_truth.get(frame, {})
        pred_objects = predictions.get(frame, {})
        
        total_gt += len(gt_objects)
        
        if len(gt_objects) == 0:
            total_fp += len(pred_objects)
            continue
        
        if len(pred_objects) == 0:
            total_miss += len(gt_objects)
            continue
        
        # Cost matrix: GT vs predictions
        gt_ids = list(gt_objects.keys())
        pred_ids = list(pred_objects.keys())
        
        cost = np.full((len(gt_ids), len(pred_ids)), distance_threshold * 2)
        
        for i, gid in enumerate(gt_ids):
            for j, pid in enumerate(pred_ids):
                gx, gy = gt_objects[gid]
                px, py = pred_objects[pid]
                dist = np.sqrt((gx - px)**2 + (gy - py)**2)
                cost[i, j] = dist
        
        row_ind, col_ind = linear_sum_assignment(cost)
        
        matched_gt = set()
        matched_pred = set()
        
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < distance_threshold:
                matched_gt.add(r)
                matched_pred.add(c)
                
                gid = gt_ids[r]
                pid = pred_ids[c]
                
                # Check for switch
                if gid in gt_to_track and gt_to_track[gid] != pid:
                    total_switch += 1
                gt_to_track[gid] = pid
        
        total_miss += len(gt_ids) - len(matched_gt)
        total_fp += len(pred_ids) - len(matched_pred)
    
    mota = 1.0 - (total_miss + total_fp + total_switch) / max(total_gt, 1)
    
    return {
        'MOTA': mota,
        'misses': total_miss,
        'false_positives': total_fp,
        'switches': total_switch,
        'total_gt': total_gt
    }


def evaluate_on_sequence(tracker: BoxMOTJugglingTracker, 
                         video_path: str,
                         gt_path: str,
                         distance_threshold: float = 30.0) -> Dict[str, float]:
    """
    Evalúa tracker en una secuencia.
    """
    gt_df = load_mot_gt(gt_path)
    gt_centers = gt_to_centers_per_frame(gt_df)
    
    cap = cv2.VideoCapture(video_path)
    
    tracker.reset()
    
    pred_centers = {}
    frame_idx = 1  # MOT es 1-based
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = tracker.detect_balls(frame)
        raw = tracker.tracker.update(detections, frame)
        
        # BoxMOT v16: devuelve (N,8) con tracks, o (0,) vacío 1D
        # reshape garantiza que siempre sea 2D antes de iterar
        tracks = np.array(raw, dtype=np.float64)
        if tracks.ndim != 2 or tracks.shape[1] < 5:
            frame_idx += 1
            continue
        
        # tracks[:, 4] = track_id  (confirmado con debug)
        frame_preds = {}
        for track in tracks:
            x1, y1, x2, y2 = track[0], track[1], track[2], track[3]
            track_id = int(track[4])
            frame_preds[track_id] = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        pred_centers[frame_idx] = frame_preds
        frame_idx += 1
    
    cap.release()
    return compute_mota(pred_centers, gt_centers, distance_threshold)


def objective_function(trial, mot_dir, tracker_type, distance_threshold=30.0):
    """
    Función objetivo para Optuna.
    Tunea TANTO preprocesamiento como parámetros del tracker.
    """
    # === Parámetros de PREPROCESAMIENTO ===
    min_contour_area = trial.suggest_int('min_contour_area', 50, 50)       # era 50-500,  bajar mín
    max_contour_area = trial.suggest_int('max_contour_area', 4500, 4500)    # era 1000-10000, ok
    var_threshold    = trial.suggest_int('var_threshold', 50, 60)           # era 10-50,   subir máx
    history          = trial.suggest_int('history', 75, 115)                # era 50-300,  bajar ambos
    morph_kernel_size = trial.suggest_int('morph_kernel_size', 3, 3, step=2) # era 3-7, 3 siempre gana
    enclosing_area_diff = trial.suggest_float('enclosing_area_diff', 0.25, 0.4) # era 0.3-0.9, bajar
    use_blur    = trial.suggest_categorical('use_blur', [True])#, False])
    blur_kernel = trial.suggest_int('blur_kernel', 3, 9, step=2)            # impar: 3,5,7,9
    bg_method   = trial.suggest_categorical('bg_method', ['MOG2'])#, 'KNN'])
    morph_ops   = trial.suggest_categorical('morph_ops', [
        'open', 'open_close'
    ])
    morph_operations = {
        'open':       ['open'],
        'close':      ['close'],
        'open_close': ['open', 'close'],
        'close_open': ['close', 'open'],
    }[morph_ops]
    
    # Confidence score variable según circularidad (simula det_threshold)
    # No se tunea directamente, se calcula en detect_balls
    
    # === Parámetros del TRACKER ===
    if tracker_type in ['ocsort', 'bytetrack']:
        det_thresh    = trial.suggest_float('det_thresh', 0.9, 0.9)        # era 0.1-0.9, subir mín
        max_age       = trial.suggest_int('max_age', 65, 65)                # era 10-60,   subir máx
        min_hits      = trial.suggest_int('min_hits', 1, 3)                 # era 1-5, sin cambio
        iou_threshold = trial.suggest_float('iou_threshold', -0.8, -0.3)    # era 0.1-0.7, bajar máx
        
        # ✅ Parámetros adicionales de OcSort
        asso_func = trial.suggest_categorical('asso_func', ['giou'])
        # delta_t: cuántos frames hacia atrás para estimar velocidad
        # bajo (1-2) = reacciona rápido a cambios de dirección (pelotas en arco)
        # alto (3-5) = más suave pero puede perder picos de trayectoria
        delta_t   = trial.suggest_int('delta_t', 1, 3)
        # inertia: peso de la predicción Kalman vs observación
        # alto = confía más en predicción (útil si hay oclusiones breves)
        # bajo = confía más en detección (útil si BG subtraction es limpio)
        inertia   = trial.suggest_float('inertia', 0.6, 0.6)

    else:  # deepocsort, botsort
        track_high_thresh = trial.suggest_float('track_high_thresh', 0.2, 0.8)
        track_low_thresh = trial.suggest_float('track_low_thresh', 0.05, 0.3)
        new_track_thresh = trial.suggest_float('new_track_thresh', 0.3, 0.8)
        track_buffer = trial.suggest_int('track_buffer', 10, 60)
        match_thresh = trial.suggest_float('match_thresh', 0.5, 0.95)
    
    # Construir params
    params = {
        'tracker_type': tracker_type,
        'bg_method': bg_method,
        'min_contour_area': min_contour_area,
        'max_contour_area': max_contour_area,
        'var_threshold': var_threshold,
        'history': history,
        'morph_kernel_size': morph_kernel_size,
        'enclosing_area_diff': enclosing_area_diff,
        'detect_shadows': False,
        'use_blur': use_blur,
        'blur_kernel': blur_kernel,
        'morph_operations': morph_operations,
    }
    
    if tracker_type in ['ocsort', 'bytetrack']:
        params['track_high_thresh'] = det_thresh
        params['track_buffer'] = max_age
        params['match_thresh'] = iou_threshold
        params['min_hits'] = min_hits
        # ✅ Añadir los nuevos
        params['asso_func'] = asso_func
        params['delta_t'] = delta_t
        params['inertia'] = inertia

    else:
        params['track_high_thresh'] = track_high_thresh
        params['track_low_thresh'] = track_low_thresh
        params['new_track_thresh'] = new_track_thresh
        params['track_buffer'] = track_buffer
        params['match_thresh'] = match_thresh
    
    # Crear tracker
    try:
        tracker = BoxMOTJugglingTracker(**params)
    except Exception as e:
        print(f"⚠️ Error creando tracker: {e}")
        return -1.0
    
    # Evaluar en todas las secuencias de train
    mot_dir = Path(mot_dir)
    train_dir = mot_dir / 'train'
    
    if not train_dir.exists():
        print(f"❌ No existe {train_dir}")
        return -1.0
    
    sequences = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    
    if len(sequences) == 0:
        print("❌ No hay secuencias en train/")
        return -1.0
    
    motas = []
    
    for seq_dir in sequences:
        gt_path = seq_dir / 'gt' / 'gt.txt'
        
        # Reconstruir video desde frames
        img_dir = seq_dir / 'img1'
        frames = sorted(img_dir.glob('*.jpg'))
        
        if not gt_path.exists() or len(frames) == 0:
            continue
        
        # Usar el video original si existe, sino reconstituir desde frames
        video_candidates = list(seq_dir.parent.parent.parent.glob(
            f"gts+vids/{seq_dir.name}.mp4"
        ))
        
        # Intentar encontrar el video original
        # Recorremos posibles ubicaciones
        video_path = None
        for pattern in [
            seq_dir.parent.parent.parent / 'gts+vids' / f"{seq_dir.name}.mp4",
            seq_dir.parent.parent.parent.parent / 'gts+vids' / f"{seq_dir.name}.mp4",
        ]:
            if pattern.exists():
                video_path = str(pattern)
                break
        
        if video_path is None:
            # Crear video temporal desde frames
            import tempfile
            first_frame = cv2.imread(str(frames[0]))
            h, w = first_frame.shape[:2]
            tmp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(tmp_video, fourcc, 30, (w, h))
            for f in frames:
                writer.write(cv2.imread(str(f)))
            writer.release()
            video_path = tmp_video
        
        try:
            metrics = evaluate_on_sequence(tracker, video_path, str(gt_path), distance_threshold)
            motas.append(metrics['MOTA'])
        except Exception as e:
            print(f"⚠️ Error en {seq_dir.name}: {e}")
            motas.append(-1.0)
    
    if len(motas) == 0:
        return -1.0
    
    avg_mota = np.mean(motas)
    
    trial.set_user_attr('per_sequence_mota', motas)
    
    return avg_mota


def tune_tracker(mot_dir: str,
                 tracker_type: str = 'ocsort',
                 n_trials: int = 50,
                 output_config: str = 'configs/best_config.yaml',
                 distance_threshold: float = 30.0):
    """
    Ejecuta tuning con Optuna.
    """
    print(f"🎯 Tuneando {tracker_type} con {n_trials} trials")
    print(f"📁 Dataset MOT: {mot_dir}")
    
    study = optuna.create_study(direction='maximize', study_name=f'juggling_{tracker_type}')
    
    study.optimize(
        lambda trial: objective_function(
            trial, mot_dir, tracker_type, distance_threshold
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Guardar mejor config
    best_params = study.best_trial.params
    best_params['tracker_type'] = tracker_type
    best_params['best_mota'] = study.best_value
    # best_params['bg_method'] = 'MOG2'
    # best_params['detect_shadows'] = False
    # best_params['use_blur'] = True
    # best_params['blur_kernel'] = 5
    # best_params['morph_operations'] = ['open']
    
    Path(output_config).parent.mkdir(parents=True, exist_ok=True)
    with open(output_config, 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False)
    
    print(f"\n{'='*50}")
    print(f"🏆 Mejor MOTA: {study.best_value:.4f}")
    print(f"📄 Config guardada: {output_config}")
    print(f"{'='*50}")
    print(f"\nMejores parámetros:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    return best_params, study


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tunea BoxMOT tracker para juggling")
    parser.add_argument('--mot_dir', default='datasets/juggling-mot',
                       help='Directorio del dataset MOT')
    parser.add_argument('--tracker_type', default='ocsort',
                       choices=['ocsort', 'deepocsort', 'botsort', 'bytetrack'])
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--output_config', default='configs/best_config.yaml')
    parser.add_argument('--distance_threshold', type=float, default=30.0,
                       help='Distancia máxima para match (px)')
    args = parser.parse_args()
    
    tune_tracker(
        mot_dir=args.mot_dir,
        tracker_type=args.tracker_type,
        n_trials=args.n_trials,
        output_config=args.output_config,
        distance_threshold=args.distance_threshold
    )