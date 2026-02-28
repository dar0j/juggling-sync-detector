"""
Paso 3: Tracking batch con parámetros óptimos.
Guarda CSVs en formato autocolortrack (sin header, -1 para lost tracks)
para que pasen directamente por nohandlebars.py

Uso:
    python batch_track.py --config configs/best_config.yaml 
        --video_dir ../../../../PROJECT/Datasets/used\ to\ track 
        --output_dir datasets/tracked 
        --num_balls 5
    
    # O un solo video:
    python batch_track.py --config configs/best_config.yaml --video path/to/video.mp4 --output_dir datasets/tracked --num_balls 3
"""
import cv2
import numpy as np
import yaml
import argparse
import re
from pathlib import Path
from tqdm import tqdm

from boxmot_tracker import BoxMOTJugglingTracker
import sys
sys.path.insert(0, str(Path(__file__).parent / 'bitrack'))
from auto_color_detect import detect_ball_colors, save_color_config


def track_video_to_autocolortrack_format(tracker: BoxMOTJugglingTracker,
                                          video_path: str,
                                          num_balls: int,
                                          output_csv: str,
                                          visualize: bool = False):
    """
    Trackea un video y guarda en formato idéntico a autocolortrack.py:
    - Sin header
    - Columnas: x_ball1,y_ball1,x_ball2,y_ball2,...
    - -1 para frames sin detección
    - Separado por comas
    - Una fila por frame
    
    Este formato es leído directamente por nohandlebars.py
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    trajectories = np.full((total_frames, num_balls, 2), np.nan, dtype=np.float32)
    
    tracker.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=tracker.history,
        varThreshold=tracker.var_threshold,
        detectShadows=tracker.detect_shadows
    )
    tracker.tracker.reset()
    
    track_id_to_ball = {}       # track_id -> ball_id (0-based)
    last_known_center = {}      # ball_id -> (cx, cy) último centro conocido
    next_ball_id = 0
    MAX_REASSIGN_DIST = 80      # píxeles máximos para reasignar track perdido
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = tracker.detect_balls(frame)
        tracks = tracker.tracker.update(detections, frame)
        
        if len(tracks) > 0:
            for track in tracks:
                x1, y1, x2, y2, track_id = track[:5]
                track_id = int(track_id)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Mapear track_id a ball_id (0-based, hasta num_balls-1)
                if track_id not in track_id_to_ball:
                    # Intentar reasignar a ball_id perdido por distancia
                    best_ball = None
                    best_dist = MAX_REASSIGN_DIST
                    
                    occupied_balls = set(track_id_to_ball.values())
                    
                    for ball_id, (lx, ly) in last_known_center.items():
                        if ball_id in occupied_balls:
                            continue  # Ya tiene un track activo
                        dist = np.sqrt((cx - lx)**2 + (cy - ly)**2)
                        if dist < best_dist:
                            best_dist = dist
                            best_ball = ball_id
                    
                    if best_ball is not None:
                        track_id_to_ball[track_id] = best_ball
                    elif next_ball_id < num_balls:
                        track_id_to_ball[track_id] = next_ball_id
                        next_ball_id += 1
                    else:
                        continue
                
                ball_id = track_id_to_ball[track_id]
                if ball_id < num_balls:
                    trajectories[frame_idx, ball_id, 0] = cx
                    trajectories[frame_idx, ball_id, 1] = cy
                    last_known_center[ball_id] = (cx, cy)
        
        frame_idx += 1
    
    cap.release()
    
    # Guardar en formato autocolortrack: reshape a (frames, balls*2), NaN -> -1
    data = trajectories[:frame_idx].reshape(frame_idx, num_balls * 2)
    data_int = np.where(np.isnan(data), -1, data).astype(int)
    
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_csv, data_int, delimiter=',', fmt='%d')
    
    # Estadísticas
    valid_per_ball = np.sum(~np.isnan(trajectories[:frame_idx, :, 0]), axis=0)
    
    print(f"✅ {Path(output_csv).name}: {frame_idx} frames, {num_balls} pelotas")
    for i in range(num_balls):
        pct = (valid_per_ball[i] / frame_idx) * 100
        print(f"   Pelota {i+1}: {pct:.1f}% detectada")


def extract_num_balls_from_filename(filename: str, fallback: int = 3) -> int:
    """
    Extrae num_balls del nombre de archivo con convención 'numballs_trickname_id.mp4'.
    
    Ejemplos:
        ss3_id_110.mp4      → 3
        ss642_id_990.mp4    → 642 ← siteswap, no num_balls directamente
        5_cascade_001.mp4   → 5
    
    Convención asumida: el PRIMER número del nombre es num_balls.
    Si el nombre empieza con 'ss', num_balls = cantidad de dígitos distintos
    en el siteswap (ej: ss531 → {5,3,1} → 3 pelotas = max//2+1... 
    o simplemente usar fallback y dejar que el usuario lo especifique).
    
    Para nombres tipo '5_cascade_001': extrae el primer número como num_balls.
    """
    stem = Path(filename).stem  # sin extensión
    
    # Patrón: nombre empieza con dígito(s) seguido de _ o fin
    # Ej: '5_cascade_001', '3_shower_002', '7_mills_003'
    match = re.match(r'^(\d+)_', stem)
    if match:
        return int(match.group(1))
    
    # Fallback: usar el argumento --num_balls
    return fallback


def batch_track(config_path: str,
                video_dir: str = None,
                video_path: str = None,
                output_dir: str = 'datasets/tracked',
                num_balls: int = None,           # ← ahora opcional
                visualize: bool = False,
                auto_color=False,          # ✅ nuevo
                color_config_dir=None):    # ✅ nuevo: guardar colores por video

    """
    Tracking batch o individual con parámetros óptimos.
    Si num_balls es None, lo extrae del nombre de archivo.
    """
    # Cargar config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📄 Config cargada: {config_path}")
    print(f"   Tracker: {config.get('tracker_type', 'unknown')}")
    print(f"   MOTA del tuning: {config.pop('best_mota', 'N/A')}")

    # Mapear nombres del yaml (Optuna) → nombres del constructor
    KEY_MAP = {
        'det_thresh':    'track_high_thresh',
        'max_age':       'track_buffer',
        'iou_threshold': 'match_thresh',
        'morph_ops':     None,   # se reconstruye abajo
    }
    for yaml_key, init_key in KEY_MAP.items():
        if yaml_key in config:
            val = config.pop(yaml_key)
            if init_key is not None:
                config[init_key] = val

    # Reconstruir morph_operations si viene como string 'open' / 'open_close'
    if 'morph_operations' not in config:
        config['morph_operations'] = ['open']  # default seguro

    # Crear tracker
    tracker = BoxMOTJugglingTracker(**config)

    # Encontrar videos
    if video_path:
        video_files = [Path(video_path)]
    elif video_dir:
        video_dir = Path(video_dir)
        video_files = sorted(
            list(video_dir.glob('**/*.mp4')) +
            list(video_dir.glob('**/*.avi')) +
            list(video_dir.glob('**/*.gif'))
        )
    else:
        print("❌ Especifica --video o --video_dir")
        return
    
    print(f"\n🎬 {len(video_files)} videos a procesar")
    if num_balls is None:
        print("   ℹ️  num_balls se extraerá del nombre de cada archivo")
    else:
        print(f"   ℹ️  num_balls fijo: {num_balls} (override manual)")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    skipped = []
    
    for i, vf in enumerate(video_files):
        # Extraer num_balls del nombre si no se especificó manualmente
        if num_balls is not None:
            n_balls = num_balls
            source = "manual"
        else:
            n_balls = extract_num_balls_from_filename(vf.name)
            source = f"del nombre '{vf.stem}'"
        
        if n_balls is None:
            print(f"\n⚠️  [{i+1}/{len(video_files)}] {vf.name}: "
                  f"no se pudo extraer num_balls, saltando")
            skipped.append(vf.name)
            continue
        
        print(f"\n[{i+1}/{len(video_files)}] {vf.name}  →  {n_balls} pelotas ({source})")
        
        output_csv = output_dir / f"{vf.stem}.csv"
        
        # ✅ Detección automática de color si se pide
        if auto_color:
            print(f"  🎨 Detectando colores...")
            colors = detect_ball_colors(str(vf), config, n_balls)
            if colors and color_config_dir:
                color_out = Path(color_config_dir) / f"{vf.stem}_colors.yaml"
                save_color_config(colors, str(color_out))
            elif colors is None:
                print(f"  ⚠️  No se detectaron colores, continuando sin color")

        track_video_to_autocolortrack_format(
            tracker=tracker,
            video_path=str(vf),
            num_balls=n_balls,
            output_csv=str(output_csv),
            visualize=visualize
        )
    
    print(f"\n{'='*50}")
    print(f"✅ Tracking completo. CSVs en: {output_dir}")
    if skipped:
        print(f"⚠️  Saltados ({len(skipped)}): {', '.join(skipped)}")
    print(f"   Formato: autocolortrack (sin header, -1=lost)")
    print(f"   Compatible con: nohandlebars.py")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch tracking con formato autocolortrack")
    parser.add_argument('--config', required=True, help='YAML con parámetros óptimos')
    parser.add_argument('--video_dir', default=None, help='Carpeta con videos')
    parser.add_argument('--video', default=None, help='Video individual')
    parser.add_argument('--output_dir', default='datasets/tracked',
                        help='Carpeta de salida para CSVs')
    parser.add_argument('--num_balls', type=int, default=None,
                        help='Override manual de num_balls. '
                             'Si no se especifica, se extrae del nombre del archivo.')
    parser.add_argument('--visualize', action='store_true',
                        help='Mostrar ventana en vivo con detecciones')
    parser.add_argument('--auto_color', action='store_true',        # ✅ añadir
                        help='Detectar colores de pelotas automáticamente (k-means HSV)')
    parser.add_argument('--color_config_dir', default=None,         # ✅ añadir
                        help='Carpeta donde guardar YAMLs de colores detectados por video')
    args = parser.parse_args()

    batch_track(
        config_path=args.config,
        video_dir=args.video_dir,
        video_path=args.video,
        output_dir=args.output_dir,
        num_balls=args.num_balls,
        visualize=args.visualize,
        auto_color=args.auto_color,             # ✅ pasar desde args
        color_config_dir=args.color_config_dir  # ✅ pasar desde args
    )