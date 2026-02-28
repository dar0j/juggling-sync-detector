#!/usr/bin/env python3
"""
siteswap_from_peaks.py
Detecta siteswap síncrono a partir de tracks de pelotas (CSV) sin usar manos.
CSV: columnas sin header: x_ball1,y_ball1,x_ball2,y_ball2,...
Salida: periodos y siteswap estimado.
NUEVO: Procesa automáticamente todos los CSVs en carpetas "Nb TRACK 60"
VALIDACIÓN: Solo considera válidos los patrones cuyo promedio == número de pelotas
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import json
import re

# ----------- Utilities -----------
def load_tracks(csv_path, has_header=False):
    if has_header:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None)
        ncols = df.shape[1]
        assert ncols % 2 == 0, "CSV debe tener pares x,y por pelota"
        n_balls = ncols // 2
        cols = []
        for i in range(1, n_balls+1):
            cols += [f"x_ball{i}", f"y_ball{i}"]
        df.columns = cols
    return df

def smooth_signal(y, window=9, poly=3):
    if len(y) < window:
        return y
    return savgol_filter(y, window_length=window, polyorder=poly)

def interpolate_missing_detections(df):
    """
    Interpola valores faltantes (-1 o NaN) linealmente para reconstruir trayectorias
    """
    df_interp = df.copy()
    for col in df_interp.columns:
        # Reemplazar -1 con NaN
        df_interp[col] = df_interp[col].replace(-1, np.nan)
        # Interpolación lineal
        df_interp[col] = df_interp[col].interpolate(method='linear', limit_direction='both')
        # Forward/backward fill para extremos
        df_interp[col] = df_interp[col].ffill().bfill()  # ✅ fix FutureWarning
    return df_interp

# ----------- Siteswap validation (from ss_sorter.py) -----------
def siteswap_char_to_int(ch):
    """Convierte un caracter de siteswap a su valor numérico"""
    if ch.isdigit():
        return int(ch)
    elif ch.isalpha():
        return 10 + ord(ch.lower()) - ord('a')
    else:
        raise ValueError(f"Invalid character: {ch}")

def verificar_ss(ss, b):
    """
    Verificar que el promedio del siteswap sea EXACTAMENTE igual al número de pelotas.
    Esta es la condición NECESARIA (aunque no suficiente) para que sea un siteswap válido.
    
    Args:
        ss (str): El siteswap a verificar, ej: "(6x,2x)(2x,2x)"
        b (int): Número de pelotas esperado
    
    Returns:
        bool: True si promedio == b, False en caso contrario
    """
    try:
        # Limpiar el string: remover paréntesis y 'x'
        trim = ss.strip("()")
        limpio = trim.replace("x", "")
        
        # Separar por comas y paréntesis
        resultado = re.split(r'[,)(]+', limpio)
        resultado = [r for r in resultado if r]  # Remover strings vacíos
        
        if not resultado:
            return False
        
        # Convertir a valores numéricos
        values = [siteswap_char_to_int(ch) for ch in resultado]
        
        # Calcular promedio
        avg = sum(values) / len(values)
        
        # Verificar si coincide EXACTAMENTE con el número de pelotas
        return abs(avg - b) < 0.01  # Tolerancia para errores de punto flotante
    except Exception:
        return False

# ----------- Peak detection per ball (ahora detecta MÁXIMOS) -----------
def detect_y_maxima_per_ball(df, ball_index,
                             prominence=5, distance=8,
                             smooth_window=9, smooth_poly=3):
    xb = f"x_ball{ball_index}"
    yb = f"y_ball{ball_index}"
    y = df[yb].values.astype(float)
    y_s = smooth_signal(y, window=smooth_window, poly=smooth_poly)
    # MÁXIMOS => peaks on y (sin negativo)
    peaks, props = find_peaks(y_s, prominence=prominence, distance=distance)
    return [(int(p), f"ball{ball_index}",
             float(df.loc[p, xb]), float(y_s[p]),
             float(props["prominences"][i]) if "prominences" in props else 0.0)
            for i,p in enumerate(peaks)]

# ----------- Nueva función de agrupación de beats -----------
def group_peaks_into_beats_new(all_peaks, x_center, frame_window=7):
    """
    Agrupa máximos en beats según la nueva lógica:
    - Cada máximo se asigna a L o R según su posición x respecto a x_center
    - Si otro máximo aparece dentro de frame_window frames, se agrega al beat anterior
      en la mano correspondiente
    """
    if not all_peaks:
        return []
    
    # Convertir a lista de diccionarios y ordenar por frame
    entries = [{'frame': p[0], 'ball': p[1], 'x': p[2], 'y': p[3], 'prom': p[4]}
               for p in all_peaks]
    entries = sorted(entries, key=lambda e: e['frame'])
    
    beats = []
    i = 0
    n = len(entries)
    
    while i < n:
        current_entry = entries[i]
        current_frame = current_entry['frame']
        
        # Determinar mano según posición x
        diff_x = current_entry['x'] - x_center
        is_left = diff_x < 0  # negativo = izquierda, positivo = derecha
        
        # Inicializar beat
        beat = {
            'frame': current_frame,
            'left': None,
            'right': None
        }
        
        # Asignar el máximo actual a su mano
        if is_left:
            beat['left'] = current_entry
        else:
            beat['right'] = current_entry
        
        # Buscar otros máximos dentro de frame_window
        j = i + 1
        while j < n and entries[j]['frame'] - current_frame <= frame_window:
            next_entry = entries[j]
            diff_x_next = next_entry['x'] - x_center
            is_left_next = diff_x_next < 0
            
            # Agregar a la mano correspondiente si aún no está ocupada
            if is_left_next and beat['left'] is None:
                beat['left'] = next_entry
            elif not is_left_next and beat['right'] is None:
                beat['right'] = next_entry
            
            j += 1
        
        beats.append(beat)
        i = j  # Avanzar al siguiente máximo no procesado
    
    return beats

# ----------- Convertir beats a pares (L, R) -----------
def beats_to_pairs(beats):
    """
    Convierte los beats con estructura {'left': ..., 'right': ...}
    a pares (ball_left, ball_right) para mantener compatibilidad
    """
    pairs = []
    for beat in beats:
        left_ball = beat['left']['ball'] if beat['left'] else None
        right_ball = beat['right']['ball'] if beat['right'] else None
        pairs.append((left_ball, right_ball))
    return pairs

# ----------- Compute siteswap from pairs -----------
def compute_siteswap_from_pairs(pairs):
    appearances=[]
    for bi,p in enumerate(pairs):
        appearances.append((bi, p[0], 'L'))
        appearances.append((bi, p[1], 'R'))

    ball_indices={}
    for idx,(bi,ball,side) in enumerate(appearances):
        if ball is None: continue
        ball_indices.setdefault(ball, []).append(idx)

    nb = len(pairs)
    beat_pairs=[]
    for bi in range(nb):
        pair=[]
        for side in [0,1]:
            idx = 2*bi + side
            if idx >= len(appearances):
                pair.append('0')
                continue
            ball = appearances[idx][1]
            hand = appearances[idx][2]
            if ball is None:
                pair.append('0')
                continue

            lst = ball_indices.get(ball, [])
            next_idx = None
            for j in lst:
                if j > idx:
                    next_idx = j
                    break
            if next_idx is None:
                next_idx = lst[0]
            delta_beats = (appearances[next_idx][0] - appearances[idx][0]) % nb
            if delta_beats == 0:
                delta_beats = nb
            num = delta_beats * 2
            cross = (appearances[next_idx][2] != hand)
            pair.append(f"{num}x" if cross else str(num))
        beat_pairs.append(tuple(pair))
    return beat_pairs

def minimal_period(pairs):
    seq = ['(' + p[0] + ',' + p[1] + ')' for p in pairs]
    n = len(seq)
    for k in range(1, n+1):
        ok = True
        for i in range(n):
            if seq[i] != seq[i % k]:
                ok = False
                break
        if ok:
            return seq[:k]
    return seq

def rotate_sequence(seq, k):
    """Rota la secuencia k posiciones a la izquierda"""
    return seq[k:] + seq[:k]

def maximal_rotation(seq):
    """Devuelve la rotación alfabéticamente mayor de la secuencia"""
    n = len(seq)
    rotations = [rotate_sequence(seq, k) for k in range(n)]
    return max(rotations)

def find_largest_repeating_pattern(siteswap_str, num_balls):
    """
    Encuentra el subpatrón VÁLIDO que más se repite, PRIORIZANDO repeticiones consecutivas.
    
    VALIDEZ: Solo considera válido un patrón si su promedio == num_balls
    
    Args:
        siteswap_str: string del siteswap completo
        num_balls: número de pelotas (REQUERIDO para validación)
    
    Returns:
        dict con:
        - pattern: str del mejor patrón encontrado
        - repetitions: int número de apariciones totales
        - consecutive_reps: int número de repeticiones consecutivas máximas
        - is_valid: bool si promedio == num_balls
        - avg: float promedio calculado del patrón
    
    Lógica:
    1. Calcula AMBAS métricas: consecutivas Y totales
    2. PRIORIDAD: consecutivas > totales > longitud
    3. Si empate en consecutivas, usa totales como desempate
    """
    pairs = re.findall(r'\([^)]+\)', siteswap_str)
    n = len(pairs)
    
    if n == 0:
        return {'pattern': "", 'repetitions': 0, 'consecutive_reps': 0, 'is_valid': False, 'avg': 0.0}
    
    best_pattern = ""
    best_total_reps = 0
    best_consecutive_reps = 0
    best_length = 0
    best_is_valid = False
    best_avg = 0.0
    
    # Probar cada longitud de subpatrón posible (de 1 hasta n)
    for pattern_len in range(1, n + 1):
        # Diccionario para guardar info de cada patrón único
        pattern_info = {}
        
        # Recorrer todas las posiciones
        for start_pos in range(n - pattern_len + 1):
            pattern = ''.join(pairs[start_pos:start_pos + pattern_len])
            
            if pattern not in pattern_info:
                # Primera vez que vemos este patrón
                is_valid = verificar_ss(pattern, num_balls)
                
                try:
                    trim = pattern.strip("()")
                    limpio = trim.replace("x", "")
                    resultado = re.split(r'[,)(]+', limpio)
                    resultado = [r for r in resultado if r]
                    values = [siteswap_char_to_int(ch) for ch in resultado]
                    avg = sum(values) / len(values) if values else 0.0
                except:
                    avg = 0.0
                
                # Calcular repeticiones CONSECUTIVAS desde esta posición
                consecutive = 0
                i = start_pos
                while i + pattern_len <= n:
                    candidate = ''.join(pairs[i:i+pattern_len])
                    if candidate == pattern:
                        consecutive += 1
                        i += pattern_len
                    else:
                        break
                
                pattern_info[pattern] = {
                    'total_count': 1,
                    'max_consecutive': consecutive,
                    'is_valid': is_valid,
                    'avg': avg
                }
            else:
                # Ya vimos este patrón, actualizar
                pattern_info[pattern]['total_count'] += 1
                
                # Calcular consecutivas desde ESTA posición
                consecutive = 0
                i = start_pos
                while i + pattern_len <= n:
                    candidate = ''.join(pairs[i:i+pattern_len])
                    if candidate == pattern:
                        consecutive += 1
                        i += pattern_len
                    else:
                        break
                
                # Actualizar máximo de consecutivas
                pattern_info[pattern]['max_consecutive'] = max(
                    pattern_info[pattern]['max_consecutive'],
                    consecutive
                )
        
        # Evaluar todos los patrones de esta longitud
        for pattern, info in pattern_info.items():
            total_reps = info['total_count']
            consecutive_reps = info['max_consecutive']
            is_valid = info['is_valid']
            avg = info['avg']
            
            # Actualizar mejor patrón con PRIORIDAD:
            # 1. Válido > inválido
            # 2. Más repeticiones CONSECUTIVAS
            # 3. Si empate consecutivas: más repeticiones TOTALES
            # 4. Si empate totales: más largo
            is_better = False
            
            if is_valid and not best_is_valid:
                # Este es válido y el anterior no -> MEJOR
                is_better = True
            elif is_valid == best_is_valid:
                # Ambos válidos o ambos inválidos
                if consecutive_reps > best_consecutive_reps:
                    # Más repeticiones consecutivas -> MEJOR
                    is_better = True
                elif consecutive_reps == best_consecutive_reps:
                    # Empate en consecutivas, usar totales
                    if total_reps > best_total_reps:
                        is_better = True
                    elif total_reps == best_total_reps and pattern_len > best_length:
                        is_better = True
            
            if is_better:
                best_pattern = pattern
                best_total_reps = total_reps
                best_consecutive_reps = consecutive_reps
                best_length = pattern_len
                best_is_valid = is_valid
                best_avg = avg
    
    # Si no encontramos patrón que aparezca ≥2 veces, devolver secuencia completa
    if best_total_reps < 2:
        is_valid_full = verificar_ss(siteswap_str, num_balls)
        try:
            trim = siteswap_str.strip("()")
            limpio = trim.replace("x", "")
            resultado = re.split(r'[,)(]+', limpio)
            resultado = [r for r in resultado if r]
            values = [siteswap_char_to_int(ch) for ch in resultado]
            avg_full = sum(values) / len(values) if values else 0.0
        except:
            avg_full = 0.0
            
        return {
            'pattern': siteswap_str,
            'repetitions': 1,
            'consecutive_reps': 1,
            'is_valid': is_valid_full,
            'avg': avg_full
        }
    
    return {
        'pattern': best_pattern,
        'repetitions': best_total_reps,
        'consecutive_reps': best_consecutive_reps,
        'is_valid': best_is_valid,
        'avg': best_avg
    }

def substring_match_metric(expected, detected, num_balls):
    """
    Métrica mejorada: compara el esperado con el PATRÓN MÁS REPETIDO Y VÁLIDO del detectado.
    
    VALIDEZ: Solo considera válido si promedio == num_balls
    
    Args:
        expected: patrón esperado
        detected: patrón detectado
        num_balls: número de pelotas (REQUERIDO)
    
    Returns:
        dict con:
        - exact_match: bool (coincidencia exacta)
        - pattern_match: bool (patrón más grande coincide con esperado)
        - substring_match: bool (esperado está contenido en detectado)
        - coverage: float (0-1, qué porcentaje del esperado está en detectado)
        - largest_pattern: str (patrón más grande detectado)
    """
    if not expected or not detected:
        return {
            'exact_match': False, 
            'pattern_match': False,
            'substring_match': False, 
            'coverage': 0.0,
            'largest_pattern': '',
            'pattern_valid': False,
            'repetitions': 0,
            'consecutive_reps': 0,      # ✅ añadir
            'pattern_avg': 0.0
        }
    
    exact = (expected == detected)
    
    # Encontrar patrón más repetido VÁLIDO en el detectado
    pattern_info = find_largest_repeating_pattern(detected, num_balls)
    largest_pattern = pattern_info['pattern']
    pattern_valid = pattern_info['is_valid']
    repetitions = pattern_info['repetitions']
    consecutive_reps = pattern_info['consecutive_reps']  # ✅ extraer
    pattern_avg = pattern_info['avg']
    
    # Verificar si el patrón más grande coincide con el esperado
    # (considerando todas las rotaciones del patrón)
    expected_pairs = re.findall(r'\([^)]+\)', expected)
    pattern_pairs = re.findall(r'\([^)]+\)', largest_pattern)
    
    pattern_match = False
    if len(expected_pairs) == len(pattern_pairs):
        # Probar todas las rotaciones del patrón
        for k in range(len(pattern_pairs)):
            rotated = rotate_sequence(pattern_pairs, k)
            if ''.join(rotated) == expected:
                pattern_match = True
                break
    
    # Verificar si esperado está como substring en detectado repetido
    detected_extended = detected * 3
    substring = expected in detected_extended
    
    # Calcular coverage: cuántos pares del esperado aparecen en el detectado
    detected_pairs = re.findall(r'\([^)]+\)', detected)
    
    if not expected_pairs:
        coverage = 0.0
    else:
        matches = sum(1 for ep in expected_pairs if ep in detected_pairs)
        coverage = matches / len(expected_pairs)
    
    return {
        'exact_match': exact,
        'pattern_match': pattern_match,
        'substring_match': substring,
        'coverage': coverage,
        'largest_pattern': largest_pattern,
        'pattern_valid': pattern_valid,
        'repetitions': repetitions,
        'consecutive_reps': consecutive_reps,   # ✅ incluir en return
        'pattern_avg': pattern_avg
    }

# ----------- Visualización mejorada (SIN holds, más limpia) -----------
def visualize_ball_trajectories(df, all_peaks, x_center=None, title=""):
    """
    HOLDS ELIMINADOS: Los holds detectaban siteswap=2 (pelotas en manos),
    pero para detectar el patrón solo importan los MÁXIMOS (lanzamientos).
    Los holds añadían ruido visual sin utilidad para el análisis.
    """
    nballs = df.shape[1] // 2
    
    # Crear figura con dos subplots (Y y X)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ========== Subplot 1: Trayectorias Y (INVERTIDAS) ==========
    max_y = max(df[f"y_ball{i}"].max() for i in range(1, nballs+1))
    
    for i in range(1, nballs+1):
        y = df[f"y_ball{i}"].values
        y_inverted = max_y - y
        ax1.plot(y_inverted, label=f"ball{i}", alpha=0.7)
    
    # Marcar SOLO máximos (lanzamientos) en rojo
    for (fr, ball, x, yv, prom) in all_peaks:
        yv_inverted = max_y - yv
        ax1.scatter(fr, yv_inverted, color='red', s=50, marker='o', zorder=5)

    ax1.set_title(f"{title}\nTrayectorias Y (invertidas) - Máximos = Lanzamientos (rojo)")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Y (invertido)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== Subplot 2: Trayectorias X ==========
    for i in range(1, nballs+1):
        x = df[f"x_ball{i}"].values
        ax2.plot(x, label=f"ball{i}", alpha=0.7)
    
    # Marcar máximos en X
    for (fr, ball, xv, yv, prom) in all_peaks:
        ax2.scatter(fr, xv, color='red', s=50, marker='o', zorder=5)
    
    # Línea del x_center (MEDIANA, más robusta ante outliers)
    if x_center is not None:
        ax2.axhline(y=x_center, color='purple', linestyle='--', 
                    linewidth=2, label=f'x_center={x_center:.1f} (mediana)')
    
    ax2.set_title("Trayectorias X - Máximos proyectados (rojo)")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("X")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ----------- Main pipeline -----------
def pipeline(csv_path, n_balls=None,
             smooth_window=9, smooth_poly=3,
             prominence=6, distance=8,
             frame_window=7,
             use_median=True,
             interpolate=True,
             visualize=False):

    df = load_tracks(csv_path, has_header=False)
    
    # INTERPOLACIÓN: reconstruir trayectorias con detecciones faltantes
    if interpolate:
        df = interpolate_missing_detections(df)

    total_cols = df.shape[1]
    if n_balls is None:
        n_balls = total_cols // 2

    all_peaks = []

    for i in range(1, n_balls+1):
        peaks = detect_y_maxima_per_ball(df, i,
                                         prominence=prominence,
                                         distance=distance,
                                         smooth_window=smooth_window,
                                         smooth_poly=smooth_poly)
        all_peaks.extend(peaks)

    if not all_peaks:
        return {'pairs':[], 'period': [], 'siteswap':'', 'siteswap_canonical': '', 
                'period_length': 0, 'x_center': None, 'n_balls': n_balls}

    # MEDIANA vs PROMEDIO: mediana es más robusta ante outliers
    all_x = [p[2] for p in all_peaks]
    if use_median:
        x_center = float(np.median(all_x)) if all_x else 0.0
    else:
        x_center = float(np.mean(all_x)) if all_x else 0.0

    beats = group_peaks_into_beats_new(all_peaks, x_center, frame_window=frame_window)
    pairs = beats_to_pairs(beats)
    
    beat_pairs = compute_siteswap_from_pairs(pairs)
    period = minimal_period(beat_pairs)
    
    # Rotación canónica (máxima alfabéticamente)
    canonical_period = maximal_rotation(period)
    siteswap_str = ''.join(period)
    siteswap_canonical = ''.join(canonical_period)
    period_length = 2 * len(period)

    if visualize:
        visualize_ball_trajectories(df, all_peaks, x_center, title=Path(csv_path).stem)

    return {
        'pairs': beat_pairs,
        'period': period,
        'period_canonical': canonical_period,
        'siteswap': siteswap_str,
        'siteswap_canonical': siteswap_canonical,
        'period_length': period_length,
        'x_center': x_center,
        'beats': beats,
        'num_peaks': len(all_peaks),
        'n_balls': n_balls
    }


# ----------- Batch processing -----------
def process_all_folders(base_dir='.', nballs_list=[3,4,5,6], visualize=False, 
                       output_json='results.json',
                       smooth_window=9, smooth_poly=3,
                       prominence=6, distance=8, frame_window=7,
                       use_median=True, interpolate=True):
    """
    Procesa todos los CSVs en carpetas "Nb TRACK 60"
    
    PARÁMETROS CLAVE PARA GENERALIZACIÓN:
    - smooth_window: ventana de suavizado Savitzky-Golay (↑ = más suave, ↓ = más detalle)
    - smooth_poly: orden polinomial del suavizado (2-3 típico)
    - prominence: prominencia mínima de picos (↑ = solo picos altos, ↓ = más sensible)
    - distance: distancia mínima entre picos en frames (↑ = evita duplicados, ↓ = más detecciones)
    - frame_window: ventana para agrupar catches en beats (↑ = beats más largos, ↓ = más estricto)
    - use_median: True = mediana (robusto), False = promedio (sensible a outliers)
    - interpolate: True = rellena detecciones faltantes, False = usa datos crudos
    """
    base_path = Path(base_dir)
    results = {}
    
    for nballs in nballs_list:
        folder_name = f"{nballs}b" #f"{nballs}b gifs TRACK 60"
        folder_path = base_path / folder_name
        
        if not folder_path.exists():
            print(f"⚠ Carpeta no encontrada: {folder_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Procesando carpeta: {folder_name}")
        print(f"{'='*60}")
        
        csv_files = sorted(folder_path.glob("*.csv"))
        
        if not csv_files:
            print(f"  No se encontraron CSVs en {folder_path}")
            continue
        
        for csv_file in csv_files:
            print(f"\n  Procesando: {csv_file.name}")
            
            try:
                result = pipeline(
                    str(csv_file),
                    n_balls=nballs,
                    smooth_window=smooth_window,
                    smooth_poly=smooth_poly,
                    prominence=prominence,
                    distance=distance,
                    frame_window=frame_window,
                    use_median=use_median,
                    interpolate=interpolate,
                    visualize=visualize
                )
                
                # Extraer siteswap esperado del nombre
                filename = csv_file.stem
                expected = filename.split('_')[1] if '_' in filename else 'unknown'
                
                # MÉTRICAS DE COMPARACIÓN (usando patrón más repetido VÁLIDO)
                metrics_simple = substring_match_metric(expected, result['siteswap'], nballs)
                metrics_canonical = substring_match_metric(expected, result['siteswap_canonical'], nballs)
                
                # Tomar la mejor métrica PRIORIZANDO VALIDEZ
                if metrics_canonical['pattern_valid'] and not metrics_simple['pattern_valid']:
                    best_metrics = metrics_canonical
                elif metrics_simple['pattern_valid'] and not metrics_canonical['pattern_valid']:
                    best_metrics = metrics_simple
                elif metrics_canonical['coverage'] > metrics_simple['coverage']:
                    best_metrics = metrics_canonical
                elif metrics_canonical['coverage'] == metrics_simple['coverage']:
                    # Si coverage igual, preferir más repeticiones
                    best_metrics = metrics_canonical if metrics_canonical['repetitions'] > metrics_simple['repetitions'] else metrics_simple
                else:
                    best_metrics = metrics_simple
                
                results[csv_file.name] = {
                    'expected': expected,
                    'detected': result['siteswap'],
                    'detected_canonical': result['siteswap_canonical'],
                    'largest_pattern': best_metrics['largest_pattern'],
                    'pattern_valid': best_metrics['pattern_valid'],
                    'pattern_avg': best_metrics['pattern_avg'],
                    'repetitions': best_metrics['repetitions'],
                    'consecutive_reps': best_metrics['consecutive_reps'],
                    'period_length': result['period_length'],
                    'x_center': result['x_center'],
                    'num_peaks': result['num_peaks'],
                    'num_beats': len(result['pairs']),
                    'num_balls': nballs,
                    # MÉTRICAS
                    'exact_match': best_metrics['exact_match'],
                    'pattern_match': best_metrics['pattern_match'],
                    'substring_match': best_metrics['substring_match'],
                    'coverage': best_metrics['coverage']
                }
                
                # Símbolos de resultado CONSIDERANDO VALIDEZ
                if best_metrics['exact_match']:
                    status = "✓✓✓"  # exacto
                elif best_metrics['pattern_match'] and best_metrics['pattern_valid']:
                    status = "✓✓"   # patrón válido (avg == nballs) que coincide
                elif best_metrics['pattern_match']:
                    status = "✓"    # patrón coincide pero inválido (avg ≠ nballs)
                elif best_metrics['substring_match']:
                    status = "~"    # substring
                elif best_metrics['coverage'] > 0.5:
                    status = "~"    # parcial
                else:
                    status = "✗"    # fallo
                
                valid_symbol = "✓" if best_metrics['pattern_valid'] else "✗"
                
                print(f"    Esperado:        {expected}")
                print(f"    Patrón grande:   {best_metrics['largest_pattern']} {status}")
                print(f"    Válido:          {valid_symbol} (avg={best_metrics['pattern_avg']:.1f}, esperado={nballs})")
                print(f"    Repeticiones:    {best_metrics['repetitions']}x")
                print(f"    Detectado full:  {result['siteswap']}")
                print(f"    Canónico:        {result['siteswap_canonical']}")
                print(f"    Coverage:        {best_metrics['coverage']*100:.0f}%")
                print(f"    Picos:           {result['num_peaks']}, Beats: {len(result['pairs'])}")
                
            except Exception as e:
                print(f"    ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                results[csv_file.name] = {
                    'expected': 'unknown',
                    'detected': 'ERROR',
                    'error': str(e)
                }
    
    # Guardar resultados en JSON
    output_path = base_path / output_json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Resultados guardados en: {output_path}")
    print(f"{'='*60}")
    
    # Estadísticas detalladas
    total = len(results)
    exact = sum(1 for r in results.values() if r.get('exact_match', False))
    pattern_valid = sum(1 for r in results.values() 
                       if r.get('pattern_match', False) and r.get('pattern_valid', False) 
                       and not r.get('exact_match', False))
    pattern_invalid = sum(1 for r in results.values() 
                         if r.get('pattern_match', False) and not r.get('pattern_valid', False)
                         and not r.get('exact_match', False))
    substring = sum(1 for r in results.values() 
                   if r.get('substring_match', False) and not r.get('pattern_match', False))
    partial = sum(1 for r in results.values() 
                 if r.get('coverage', 0) > 0.5 and not r.get('substring_match', False))
    errors = sum(1 for r in results.values() if 'error' in r)
    
    print(f"\nEstadísticas:")
    print(f"  Total procesados:            {total}")
    print(f"  Exactas (✓✓✓):               {exact} ({exact/total*100:.1f}%)")
    print(f"  Patrón válido (✓✓):          {pattern_valid} ({pattern_valid/total*100:.1f}%)")
    print(f"  Patrón inválido (✓):         {pattern_invalid} ({pattern_invalid/total*100:.1f}%)")
    print(f"  Substring (~):               {substring} ({substring/total*100:.1f}%)")
    print(f"  Parciales (>50%, ~):         {partial} ({partial/total*100:.1f}%)")
    print(f"  Errores (✗):                 {errors}")
    
    # Accuracy ESTRICTO: solo exactos + patrones válidos
    accuracy = (exact + pattern_valid) / total if total > 0 else 0
    print(f"  Accuracy (exacto+válido):    {accuracy*100:.1f}%")
    
    avg_coverage = np.mean([r.get('coverage', 0) for r in results.values() if 'coverage' in r])
    print(f"  Coverage promedio:           {avg_coverage*100:.1f}%")
    
    valid_patterns = sum(1 for r in results.values() if r.get('pattern_valid', False))
    print(f"  Patrones válidos totales:    {valid_patterns} ({valid_patterns/total*100:.1f}%)")
    
    return results


# ----------- CLI -----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='Procesar un solo CSV')
    parser.add_argument('--nballs', type=int, default=None)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--base-dir', default='.')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--output', default='siteswap_results.json')
    
    # PARÁMETROS AJUSTABLES
    parser.add_argument('--smooth-window', type=int, default=9, help='Ventana suavizado (7-15)')
    parser.add_argument('--prominence', type=float, default=6.0, help='Prominencia picos (3-10)')
    parser.add_argument('--distance', type=int, default=8, help='Distancia entre picos (5-15)')
    parser.add_argument('--frame-window', type=int, default=7, help='Ventana agrupación beats (5-10)')
    parser.add_argument('--use-mean', action='store_true', help='Usar promedio en vez de mediana')
    parser.add_argument('--no-interpolate', action='store_true', help='No interpolar detecciones faltantes')
    
    args = parser.parse_args()
    
    if args.batch:
        process_all_folders(
            base_dir=args.base_dir,
            nballs_list=[3, 4, 5, 6],
            visualize=False,
            output_json=args.output,
            smooth_window=args.smooth_window,
            prominence=args.prominence,
            distance=args.distance,
            frame_window=args.frame_window,
            use_median=not args.use_mean,
            interpolate=not args.no_interpolate
        )
    elif args.csv:
        if not args.nballs:
            print("ERROR: --nballs es requerido para validación")
            parser.print_help()
            exit(1)
            
        out = pipeline(args.csv, n_balls=args.nballs, 
                      smooth_window=args.smooth_window,
                      prominence=args.prominence,
                      distance=args.distance,
                      frame_window=args.frame_window,
                      use_median=not args.use_mean,
                      interpolate=not args.no_interpolate,
                      visualize=args.visualize or True)
        print("\n" + "="*60)
        print("RESULTADO")
        print("="*60)
        print(f"Siteswap detectado:  {out['siteswap']}")
        print(f"Siteswap canónico:   {out['siteswap_canonical']}")
        print(f"Periodo (beats):     {out['period_length']}")
        print(f"x_center usado:      {out['x_center']:.1f}")
        print(f"Picos detectados:    {out['num_peaks']}")
        
        # Validar patrón detectado
        print(f"\nPATRÓN MÁS REPETIDO:")
        pattern_info = find_largest_repeating_pattern(out['siteswap'], args.nballs)
        print(f"  Patrón:            {pattern_info['pattern']}")
        print(f"  Repeticiones:      {pattern_info['repetitions']}x")
        print(f"  Promedio:          {pattern_info['avg']:.1f}")
        print(f"  Esperado:          {args.nballs}")
        print(f"  Válido:            {'✓' if pattern_info['is_valid'] else '✗'} (avg {'==' if pattern_info['is_valid'] else '!='} nballs)")
    else:
        parser.print_help()