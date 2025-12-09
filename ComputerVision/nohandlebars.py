#!/usr/bin/env python3
"""
siteswap_from_peaks.py
Detecta siteswap síncrono a partir de tracks de pelotas (CSV) sin usar manos.
CSV: columnas sin header: x_ball1,y_ball1,x_ball2,y_ball2,...
Salida: periodos y siteswap estimado.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import argparse
import matplotlib.pyplot as plt

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

def detect_holds(df, ball_index, vy_thresh=0.8, flat_window=5):
    """
    Detecta frames donde la pelota está en 'hold' (siteswap = 2).
    """
    y = df[f"y_ball{ball_index}"].values.astype(float)
    vy = np.gradient(y)

    holds = []
    for i in range(len(y)):
        if abs(vy[i]) < vy_thresh:
            start = max(0, i-flat_window)
            end   = min(len(y), i+flat_window)
            local_std = np.std(y[start:end])
            if local_std < 1.5:
                holds.append(i)
    return holds

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
            # Si ya hay una pelota en esa mano, se podría sobrescribir
            # o ignorar según lógica deseada
            
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
                pair.append('0')   # ← cuando no hay pelota asignada
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

# ----------- Visualización mejorada (gráficos invertidos) -----------
def visualize_ball_trajectories(df, all_peaks, holds=None, x_center=None):
    nballs = df.shape[1] // 2
    
    # Crear figura con dos subplots (Y y X)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ========== Subplot 1: Trayectorias Y (INVERTIDAS) ==========
    for i in range(1, nballs+1):
        y = df[f"y_ball{i}"].values
        ax1.plot(-y, label=f"ball{i}", alpha=0.7)  # Invertir: -y
    
    # Marcar máximos en Y (invertidos)
    for (fr, ball, x, yv, prom) in all_peaks:
        ax1.scatter(fr, -yv, color='red', s=50, marker='o', zorder=5)  # Invertir: -yv

    if holds is not None:
        for ball, frames in holds.items():
            for fr in frames:
                if fr < len(df):
                    yv = df.loc[fr, f"y_{ball}"]
                    ax1.scatter(fr, -yv, color='green', s=80, marker='s', zorder=5)  # Invertir: -yv

    ax1.set_title("Trayectorias Y (invertidas)\nmáximos (rojo) y holds/2 (verde)")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("-Y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Invertir eje Y para que arriba sea positivo
    
    # ========== Subplot 2: Trayectorias X ==========
    for i in range(1, nballs+1):
        x = df[f"x_ball{i}"].values
        ax2.plot(x, label=f"ball{i}", alpha=0.7)
    
    # Marcar máximos en X (en los frames donde ocurren máximos en Y)
    for (fr, ball, xv, yv, prom) in all_peaks:
        ax2.scatter(fr, xv, color='red', s=50, marker='o', zorder=5)
    
    # Línea del x_center
    if x_center is not None:
        ax2.axhline(y=x_center, color='purple', linestyle='--', 
                    linewidth=2, label=f'x_center={x_center:.1f}')
    
    ax2.set_title("Trayectorias X\nmáximos en Y proyectados (rojo)")
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
             frame_window=7):  # Cambiado a 7 frames

    df = load_tracks(csv_path, has_header=False)

    total_cols = df.shape[1]
    if n_balls is None:
        n_balls = total_cols // 2

    all_peaks = []
    holds = {}

    for i in range(1, n_balls+1):
        # Detectar MÁXIMOS en vez de mínimos
        peaks = detect_y_maxima_per_ball(df, i,
                                         prominence=prominence,
                                         distance=distance,
                                         smooth_window=smooth_window,
                                         smooth_poly=smooth_poly)
        all_peaks.extend(peaks)

        h = detect_holds(df, i)
        holds[f"ball{i}"] = h

    if not all_peaks:
        return {'pairs':[], 'period': [], 'siteswap':'', 'x_center': None}

    # Calcular x_center (media de todas las posiciones x)
    all_x = [p[2] for p in all_peaks]
    x_center = float(np.mean(all_x)) if all_x else 0.0

    # Nueva agrupación de beats
    beats = group_peaks_into_beats_new(all_peaks, x_center, frame_window=frame_window)
    
    # Convertir a pares para mantener compatibilidad con compute_siteswap
    pairs = beats_to_pairs(beats)
    
    beat_pairs = compute_siteswap_from_pairs(pairs)
    period = minimal_period(beat_pairs)
    siteswap_str = ''.join(period)

    # Visualización con ambos ejes
    visualize_ball_trajectories(df, all_peaks, holds, x_center)

    return {
        'pairs': beat_pairs,
        'period': period,
        'siteswap': siteswap_str,
        'x_center': x_center,
        'beats': beats,
        'holds': holds
    }


# ----------- CLI -----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--nballs', type=int, default=None)
    args = parser.parse_args()
    out = pipeline(args.csv, n_balls=args.nballs)
    print("Siteswap estimado:", out['siteswap'])
    print("Pairs (sample):", out['pairs'][:12])
    print("x_center usado:", out['x_center'])
    print("Period detected:", out['period'])