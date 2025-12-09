#!/usr/bin/env python3
"""
extract_sync_siteswap.py
Entrada: CSV con columnas:
 x_righthand, y_righthand, x_lefthand, y_lefthand, x_ball1, y_ball1, x_ball2, y_ball2, ...
Salida: siteswap síncrono estimado (ej. "(4x,2x)"), la secuencia completa de pares, y el periodo mínimo.
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

# ---------- utilidades ----------
def load_positions(csv_path):
    # Intentar leer con cabecera
    df = pd.read_csv(csv_path)
    
    # Si no tiene cabecera, asignar nombres manualmente
    if not all(col.startswith('x_') or col.startswith('y_') for col in df.columns[:4]):
        # Asumir formato: x_rh, y_rh, x_lh, y_lh, x_b1, y_b1, ...
        n_cols = len(df.columns)
        n_balls = (n_cols - 4) // 2
        cols = ['x_righthand', 'y_righthand', 'x_lefthand', 'y_lefthand']
        for i in range(1, n_balls + 1):
            cols.extend([f'x_ball{i}', f'y_ball{i}'])
        df.columns = cols[:n_cols]
    
    return df

def smooth_and_interp(df, window=7, poly=2):
    # Interpolar NaNs y -1 (valores faltantes)
    df = df.replace(-1, np.nan)
    df = df.interpolate(limit_direction='both', axis=0)
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    # suavizar (Savgol) por columna numérica
    for col in df.columns:
        try:
            arr = df[col].values.astype(float)
            if len(arr) >= window:
                df[col] = savgol_filter(arr, window, poly)
        except Exception:
            pass
    return df

def compute_velocities(df, fps=30):
    # Devuelve dict: vel[col] = np.array(dx/dt,dy/dt) stacked
    dt = 1.0/fps
    vel = {}
    cols = list(df.columns)
    for c in cols:
        if c.startswith('x_'):
            name = c[2:]
            y_col = f'y_{name}'
            if y_col not in df.columns:
                continue
            x = df[c].values
            y = df[y_col].values
            vx = np.gradient(x, dt)
            vy = np.gradient(y, dt)
            vel[name] = np.vstack((vx, vy)).T
    return vel

def detect_catches(df, vel, dist_thresh=60, v_thresh=5.0, debug=False):
    n_frames = len(df)
    balls = sorted({col.split('_',1)[1] for col in df.columns if col.startswith('x_ball')})
    
    if debug:
        print(f"\n=== DEBUG: detect_catches ===")
        print(f"Frames: {n_frames}, Balls: {balls}")
        print(f"Thresholds: dist={dist_thresh}, v={v_thresh}")
    
    catches = []
    
    for b in balls:
        bx_col = f'x_{b}'
        by_col = f'y_{b}'
        if bx_col not in df.columns or by_col not in df.columns:
            continue
            
        bx = df[bx_col].values
        by = df[by_col].values
        vb = vel.get(b, np.zeros((n_frames, 2)))
        speed = np.sqrt(vb[:,0]**2 + vb[:,1]**2)
        
        for hand in ['lefthand', 'righthand']:
            hx = df[f'x_{hand}'].values
            hy = df[f'y_{hand}'].values
            dist = np.sqrt((bx - hx)**2 + (by - hy)**2)
            
            # Detectar mínimos locales de distancia
            for t in range(2, n_frames - 2):
                # Mínimo local
                if dist[t] < dist_thresh:
                    is_local_min = (dist[t] <= dist[t-1] and dist[t] <= dist[t+1])
                    
                    # Condición relajada: solo requiere cercanía + velocidad baja
                    if is_local_min or speed[t] < v_thresh:
                        # Evitar duplicados cercanos
                        if not any(c['ball'] == b and c['hand'] == hand[0].upper() and abs(c['frame'] - t) < 5 
                                  for c in catches):
                            catches.append({
                                'frame': t,
                                'ball': b,
                                'hand': 'L' if hand == 'lefthand' else 'R',
                                'dist': dist[t],
                                'speed': speed[t]
                            })
                            
                            if debug and len(catches) <= 10:
                                print(f"  Catch: frame={t}, ball={b}, hand={hand[0].upper()}, "
                                      f"dist={dist[t]:.1f}, speed={speed[t]:.1f}")
    
    if debug:
        print(f"Total catches detected: {len(catches)}")
    
    return catches

def group_into_beats(catches, frame_window=8, debug=False):
    if not catches:
        return []
    
    catches_sorted = sorted(catches, key=lambda x: x['frame'])
    beats = []
    i = 0
    n = len(catches_sorted)
    
    while i < n:
        base = catches_sorted[i]
        group = [base]
        j = i + 1
        
        # Agrupar catches dentro de frame_window
        while j < n and (catches_sorted[j]['frame'] - base['frame']) <= frame_window:
            # Evitar duplicar misma pelota en mismo beat
            if not any(g['ball'] == catches_sorted[j]['ball'] for g in group):
                group.append(catches_sorted[j])
            j += 1
        
        # Aceptar beats con al menos 1 catch
        if len(group) >= 1:
            # Preferir 2 catches (L y R), pero aceptar 1
            if len(group) > 2:
                # Tomar los 2 con menor distancia
                group = sorted(group, key=lambda x: x['dist'])[:2]
            beats.append(group)
        
        i = j
    
    if debug:
        print(f"\n=== DEBUG: group_into_beats ===")
        print(f"Total beats: {len(beats)}")
        for idx, b in enumerate(beats[:5]):
            print(f"  Beat {idx}: frame={b[0]['frame']}, entries={len(b)}")
    
    beats_out = []
    for g in beats:
        frame_mean = int(np.round(np.mean([c['frame'] for c in g])))
        beats_out.append({'frame': frame_mean, 'entries': g})
    
    return beats_out

def compute_siteswap_from_beats(beats, debug=False):
    if not beats:
        return []
    
    # Construir mapa de apariciones: ball -> [(beat_idx, hand), ...]
    ball_appearances = {}
    for bi, b in enumerate(beats):
        for e in b['entries']:
            ball = e['ball']
            hand = e['hand']
            ball_appearances.setdefault(ball, []).append((bi, hand))
    
    beat_pairs = []
    
    for bi, b in enumerate(beats):
        left_entry = next((e for e in b['entries'] if e['hand'] == 'L'), None)
        right_entry = next((e for e in b['entries'] if e['hand'] == 'R'), None)
        
        pair = []
        for entry, hand in [(left_entry, 'L'), (right_entry, 'R')]:
            if entry is None:
                pair.append('0')
                continue
            
            ball = entry['ball']
            appearances = ball_appearances.get(ball, [])
            
            # Encontrar próxima aparición después de este beat
            current_idx = None
            for idx, (beat_i, h) in enumerate(appearances):
                if beat_i == bi and h == hand:
                    current_idx = idx
                    break
            
            if current_idx is None or current_idx + 1 >= len(appearances):
                # Wrap around o no hay siguiente
                if len(appearances) > 1:
                    next_beat, next_hand = appearances[0]
                else:
                    pair.append('-')
                    continue
            else:
                next_beat, next_hand = appearances[current_idx + 1]
            
            # Delta beats
            delta = (next_beat - bi) % len(beats)
            if delta == 0:
                delta = len(beats)
            
            num = delta * 2
            cross = (next_hand != hand)
            
            pair.append(f"{num}x" if cross else str(num))
        
        beat_pairs.append(tuple(pair))
    
    if debug:
        print(f"\n=== DEBUG: compute_siteswap_from_beats ===")
        print(f"Beat pairs (first 10): {beat_pairs[:10]}")
    
    return beat_pairs

def find_minimal_period(pairs):
    if not pairs:
        return []
    
    seq = ['(' + (p[0] if p[0] else '-') + ',' + (p[1] if p[1] else '-') + ')' for p in pairs]
    n = len(seq)
    
    for k in range(1, n + 1):
        if all(seq[i] == seq[i % k] for i in range(n)):
            return seq[:k]
    
    return seq

def pipeline(csv_path, fps=30, dist_thresh=60, v_thresh=5.0, debug=False):
    df = load_positions(csv_path)
    
    if debug:
        print(f"\n=== CSV Info ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First row:\n{df.iloc[0]}")
    
    df = smooth_and_interp(df)
    vel = compute_velocities(df, fps=fps)
    catches = detect_catches(df, vel, dist_thresh=dist_thresh, v_thresh=v_thresh, debug=debug)
    
    if not catches:
        print("\n⚠ WARNING: No catches detected. Try:")
        print(f"  - Increase --dist-thresh (current: {dist_thresh})")
        print(f"  - Increase --v-thresh (current: {v_thresh})")
        print(f"  - Check CSV format with --debug")
        return {'beat_pairs': [], 'siteswap': '', 'period_list': []}
    
    beats = group_into_beats(catches, debug=debug)
    beat_pairs = compute_siteswap_from_beats(beats, debug=debug)
    period = find_minimal_period(beat_pairs)
    siteswap = ''.join(period)
    
    return {
        'beat_pairs': beat_pairs,
        'siteswap': siteswap,
        'period_list': period
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--fps', type=float, default=60.0)
    parser.add_argument('--dist-thresh', type=float, default=80.0)  # aumentado
    parser.add_argument('--v-thresh', type=float, default=10.0)     # aumentado
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    out = pipeline(args.csv, fps=args.fps, dist_thresh=args.dist_thresh, 
                   v_thresh=args.v_thresh, debug=args.debug)
    
    print("\n" + "="*60)
    print("RESULTADO")
    print("="*60)
    print(f"Siteswap (periodo mínimo): {out['siteswap']}")
    print(f"Beat pairs totales: {len(out['beat_pairs'])}")
    if out['beat_pairs']:
        print(f"Primeros 10 beats: {out['beat_pairs'][:10]}")
