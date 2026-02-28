"""
Paso 1: Convierte videos + ground truth de datasets/gts+vids/ a formato MOT
para BoxMOT tuning.

Ground truth esperado: CSV con header 'ball_1_x,ball_1_y,ball_2_x,...'
Videos: .mp4 con mismo nombre base que el CSV

Uso:
    python prepare_gt_dataset.py --input_dir datasets/gts+vids --output_dir datasets/juggling-mot --ball_radius 15
"""
import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import shutil


def detect_csv_format(csv_path: str) -> dict:
    """
    Detecta el formato del CSV y retorna info para parsearlo correctamente.
    
    Formatos soportados:
    1. header: ball_1_x,ball_1_y,ball_2_x,...   → num_balls = cols/2
    2. header: x_position,y_position repetido    → num_balls = cols/2
    3. header: 0,1,2,3,...                        → num_balls = cols/2
    """
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
    
    cols = [c.strip() for c in first_line.split(',')]
    num_cols = len(cols)
    num_balls = num_cols // 2
    
    info = {
        'num_balls': num_balls,
        'has_header': True,
        'format': None
    }
    
    # Formato 1: ball_1_x, ball_1_y, ...
    if any('ball_' in c for c in cols):
        info['format'] = 'ball_named'
        return info
    
    # Formato 2: x_position, y_position repetido
    if any('position' in c.lower() or 'x_pos' in c.lower() for c in cols):
        info['format'] = 'x_position'
        return info
    
    # Formato 3: 0,1,2,3,... (numérico)
    try:
        [int(c) for c in cols]
        info['format'] = 'numeric_index'
        return info
    except ValueError:
        pass
    
    # Sin header: primera línea son datos
    try:
        [float(c) for c in cols]
        info['has_header'] = False
        info['format'] = 'no_header'
        return info
    except ValueError:
        pass
    
    # Fallback
    info['format'] = 'unknown'
    return info


def load_centers_csv(csv_path: str) -> tuple[pd.DataFrame, int]:
    """
    Carga CSV de centros independientemente del formato del header.
    
    Returns:
        df: DataFrame con columnas normalizadas ball_1_x, ball_1_y, ...
        num_balls: número de pelotas detectado
    """
    fmt_info = detect_csv_format(csv_path)
    num_balls = fmt_info['num_balls']
    
    # Leer datos según formato
    if fmt_info['has_header']:
        raw = pd.read_csv(csv_path, header=0)
    else:
        raw = pd.read_csv(csv_path, header=None)
    
    # Renombrar columnas a formato estándar
    new_cols = []
    for i in range(num_balls):
        new_cols.append(f'ball_{i+1}_x')
        new_cols.append(f'ball_{i+1}_y')
    
    if len(raw.columns) != len(new_cols):
        raise ValueError(
            f"CSV {Path(csv_path).name}: esperaba {len(new_cols)} columnas, "
            f"encontró {len(raw.columns)}"
        )
    
    raw.columns = new_cols
    
    # Convertir a numérico, errores → NaN
    raw = raw.apply(pd.to_numeric, errors='coerce')
    
    print(f"  Formato: '{fmt_info['format']}' | {num_balls} pelotas | {len(raw)} filas")
    
    return raw, num_balls


def is_all_zeros_or_invalid(row: pd.Series, min_valid: int = 5) -> bool:
    """Devuelve True si una fila tiene todas las coordenadas inválidas (0, -1, NaN)."""
    values = row.values
    for v in values:
        if pd.notna(v) and float(v) > min_valid:
            return False
    return True


def centers_csv_to_mot_gt(csv_path: str, 
                           output_txt: str, 
                           ball_radius: int = 15,
                           min_valid_coord: int = 5) -> int:
    """
    Convierte CSV de centros (cualquier formato) a MOT gt.txt.
    
    Formato MOT: frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility
    - Ignora filas donde TODAS las coordenadas son 0/-1/NaN
    - Ignora coordenadas individuales ≤ min_valid_coord
    
    Returns:
        num_balls detectado
    """
    df, num_balls = load_centers_csv(csv_path)
    
    mot_lines = []
    skipped_frames = 0
    
    for frame_idx, row in df.iterrows():
        frame_num = frame_idx + 1  # MOT es 1-based
        
        # Ignorar fila si TODAS son inválidas (ej: primera fila 0,0,0,0,...)
        if is_all_zeros_or_invalid(row, min_valid=min_valid_coord):
            skipped_frames += 1
            continue
        
        for ball_id in range(1, num_balls + 1):
            cx = row[f'ball_{ball_id}_x']
            cy = row[f'ball_{ball_id}_y']
            
            # Filtrar coordenadas inválidas individualmente
            if pd.isna(cx) or pd.isna(cy):
                continue
            if float(cx) <= min_valid_coord or float(cy) <= min_valid_coord:
                continue
            if float(cx) < 0 or float(cy) < 0:
                continue
            
            bb_left = max(0, int(cx - ball_radius))
            bb_top = max(0, int(cy - ball_radius))
            bb_width = ball_radius * 2
            bb_height = ball_radius * 2
            
            # Formato MOT: frame,id,bb_left,bb_top,w,h,conf,class,visibility
            mot_lines.append(
                f"{frame_num},{ball_id},{bb_left},{bb_top},"
                f"{bb_width},{bb_height},1,1,1.0"
            )
    
    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, 'w') as f:
        f.write('\n'.join(mot_lines) + '\n')
    
    print(f"  ✅ GT MOT: {Path(output_txt).name} | "
          f"{len(mot_lines)} dets | "
          f"{skipped_frames} frames saltados (all-zero)")
    
    return num_balls


def extract_frames(video_path: str, output_dir: str) -> tuple[int, float, int, int]:
    """Extrae frames como 000001.jpg, 000002.jpg, ..."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(output_dir / f"{idx:06d}.jpg"), frame)
        idx += 1
    
    cap.release()
    return total, fps, width, height


def create_seqinfo(seq_dir: Path, name: str, fps: float,
                   length: int, width: int, height: int):
    """Crea seqinfo.ini requerido por MOT format."""
    content = f"""[Sequence]
name={name}
imDir=img1
frameRate={int(fps)}
seqLength={length}
imWidth={width}
imHeight={height}
imExt=.jpg
"""
    with open(seq_dir / 'seqinfo.ini', 'w') as f:
        f.write(content)


def prepare_dataset(input_dir: str, 
                    output_dir: str, 
                    ball_radius: int = 15,
                    train_ratio: float = 0.8,
                    min_valid_coord: int = 5):
    """
    Prepara dataset MOT completo desde carpeta con videos + GTs.
    
    input_dir/
        video1.mp4  +  video1.csv   (cualquier formato de header)
        video2.mp4  +  video2.csv
        ...
    
    output_dir/
        train/
            video1/
                gt/gt.txt       ← MOT format
                img1/*.jpg      ← frames extraídos
                det/det.txt     ← copia del GT (para BoxMOT)
                seqinfo.ini
        test/
            ...
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Buscar pares video + csv
    csv_files = sorted(input_dir.glob('*.csv'))
    pairs = []
    
    for csv_file in csv_files:
        video_file = None
        for ext in ['.MP4', '.mp4', '.avi', '.mov', '.gif']:
            candidate = csv_file.with_suffix(ext)
            if candidate.exists():
                video_file = candidate
                break
        
        if video_file is None:
            print(f"  ⚠️  Sin video para {csv_file.name}, saltando")
            continue
        
        pairs.append((video_file, csv_file))
    
    print(f"\n📁 {len(pairs)} pares video+GT encontrados en {input_dir}")
    
    if len(pairs) == 0:
        print("❌ No se encontraron pares. Verifica nombres de archivo.")
        return
    
    n_train = max(1, int(len(pairs) * train_ratio))
    
    for i, (video_path, csv_path) in enumerate(pairs):
        split = 'train' if i < n_train else 'test'
        seq_name = video_path.stem
        seq_dir = output_dir / split / seq_name
        
        print(f"\n{'='*55}")
        print(f"[{i+1}/{len(pairs)}] {seq_name}  →  {split}/")
        print(f"{'='*55}")
        
        # 1. Convertir GT
        gt_dir = seq_dir / 'gt'
        num_balls = centers_csv_to_mot_gt(
            csv_path=str(csv_path),
            output_txt=str(gt_dir / 'gt.txt'),
            ball_radius=ball_radius,
            min_valid_coord=min_valid_coord
        )
        
        # 2. Extraer frames
        img_dir = seq_dir / 'img1'
        total_frames, fps, width, height = extract_frames(str(video_path), str(img_dir))
        print(f"  🎬 {total_frames} frames  |  {width}x{height}  |  {fps:.1f}fps")
        
        # 3. seqinfo.ini
        create_seqinfo(seq_dir, seq_name, fps, total_frames, width, height)
        
        # 4. det/det.txt  (BoxMOT lo necesita; usamos GT como detecciones perfectas)
        det_dir = seq_dir / 'det'
        det_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(gt_dir / 'gt.txt'), str(det_dir / 'det.txt'))
        
        print(f"  📋 {num_balls} pelotas  |  seqinfo.ini y det.txt listos")
    
    print(f"\n{'='*55}")
    print(f"✅ Dataset listo en: {output_dir}")
    print(f"   Train: {n_train} secuencias")
    print(f"   Test : {len(pairs) - n_train} secuencias")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepara dataset MOT desde videos + GT (cualquier formato de header)"
    )
    parser.add_argument('--input_dir', default='datasets/gts+vids',
                        help='Carpeta con videos y CSVs')
    parser.add_argument('--output_dir', default='datasets/juggling-mot',
                        help='Carpeta de salida MOT')
    parser.add_argument('--ball_radius', type=int, default=15,
                        help='Radio de pelota para bbox (px)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Proporción train/test (0.0-1.0)')
    parser.add_argument('--min_valid_coord', type=int, default=5,
                        help='Coordenadas ≤ este valor se ignoran')
    args = parser.parse_args()
    
    prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ball_radius=args.ball_radius,
        train_ratio=args.train_ratio,
        min_valid_coord=args.min_valid_coord
    )