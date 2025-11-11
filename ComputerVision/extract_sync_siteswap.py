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
    df = pd.read_csv(csv_path)
    return df

def smooth_and_interp(df, window=7, poly=2):
    # Interpolar NaNs
    df = df.interpolate(limit_direction='both', axis=0)
    # suavizar (Savgol) por columna numérica
    for col in df.columns:
        try:
            arr = df[col].values
            # requiere tamaño >= window
            if len(arr) >= window:
                df[col] = savgol_filter(arr, window, poly)
        except Exception:
            pass
    return df

def compute_velocities(df, fps=30):
    # Devuelve dict: vel[col] = np.array(dx/dt,dy/dt) stacked
    dt = 1.0/fps
    vel = {}
    # encontrar pares x,y por nombre
    cols = list(df.columns)
    for c in cols:
        if c.startswith('x_'):
            name = c[2:]
            x = df[f'x_{name}'].values
            y = df.get(f'y_{name}').values
            if y is None:
                continue
            vx = np.gradient(x, dt)
            vy = np.gradient(y, dt)
            vel[name] = np.vstack((vx, vy)).T  # shape (Nframes, 2)
    return vel

# Hungarian-based matching between sets of detections (frame t to t+1)
def match_ids_between_frames(pts_prev, pts_next, max_cost=200):
    # pts_prev, pts_next: arrays Nx2, Mx2
    if pts_prev.size == 0 or pts_next.size == 0:
        return {}
    cost = np.linalg.norm(pts_prev[:, None, :] - pts_next[None, :, :], axis=2)
    # clamp cost large
    cost[cost > max_cost] = max_cost * 2
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {}
    for r,c in zip(row_ind, col_ind):
        if cost[r,c] <= max_cost:
            mapping[r] = c
    return mapping

# Detectar catches: un catch lo definimos como el frame donde la pelota está cerca de una mano
# y la distancia pasa por un mínimo (valle) y/o su velocidad relativa cambia de acercamiento a alejamiento.
def detect_catches(df, vel, dist_thresh=60, v_thresh=0.5):
    # df columns: x_righthand, y_righthand, x_lefthand, y_lefthand, x_ball1, y_ball1, ...
    n_frames = len(df)
    # enumerar balls encontradas
    balls = sorted({col.split('_',1)[1] for col in df.columns if col.startswith('x_ball')})
    catches = []  # cada catch: dict(frame, ball, hand('left'/'right'), pos)
    for b in balls:
        bx = df[f'x_{b}'].values
        by = df[f'y_{b}'].values
        vx = vel.get(b, np.zeros((n_frames,2)))[:,0]
        vy = vel.get(b, np.zeros((n_frames,2)))[:,1]
        speed = np.sqrt(vx**2 + vy**2)
        for hand in ['lefthand','righthand']:
            hx = df[f'x_{hand}'].values
            hy = df[f'y_{hand}'].values
            dist = np.sqrt((bx-hx)**2 + (by-hy)**2)
            # detectar mínimos locales de distancia donde dist < dist_thresh
            for t in range(1, n_frames-1):
                if dist[t] < dist_thresh and dist[t] <= dist[t-1] and dist[t] <= dist[t+1]:
                    # chequeo de velocidad: venía acercándose y luego alejándose
                    # aproximación: dot product between velocity and vector (hand->ball)
                    vdot_prev = ( (bx[t-1]-hx[t-1])* (vx[t-1]) + (by[t-1]-hy[t-1])*(vy[t-1]) )
                    vdot_next = ( (bx[t+1]-hx[t+1])* (vx[t+1]) + (by[t+1]-hy[t+1])*(vy[t+1]) )
                    # si vdot_prev < 0 (acercamiento) y vdot_next > 0 (alejamiento) ==> catch/throw
                    if vdot_prev < 0 and vdot_next > 0:
                        catches.append({'frame': t, 'ball': b, 'hand': 'L' if hand=='lefthand' else 'R',
                                        'dist': dist[t], 'speed': speed[t]})
                    else:
                        # alternativa: si speed is low at t (near stop)
                        if speed[t] < v_thresh:
                            catches.append({'frame': t, 'ball': b, 'hand': 'L' if hand=='lefthand' else 'R',
                                            'dist': dist[t], 'speed': speed[t]})
    # deduplicate catches (misdetections) keep earliest within small window per ball
    catches_sorted = sorted(catches, key=lambda x: (x['ball'], x['frame']))
    filtered = []
    for c in catches_sorted:
        if not any((c['ball']==f['ball'] and abs(c['frame']-f['frame'])<=4) for f in filtered):
            filtered.append(c)
    return filtered

# Agrupar catches en beats síncronos: catches que ocurren en frames cercanos se consideran el par
def group_into_beats(catches, frame_window=4):
    # entrada: lista de dicts con 'frame','ball','hand'
    if not catches:
        return []
    # ordenar por frame
    catches_sorted = sorted(catches, key=lambda x: x['frame'])
    beats = []
    i = 0
    n = len(catches_sorted)
    while i < n:
        base = catches_sorted[i]
        # buscar otros catches dentro de frame_window
        group = [base]
        j = i+1
        while j < n and abs(catches_sorted[j]['frame'] - base['frame']) <= frame_window:
            group.append(catches_sorted[j])
            j += 1
        # si group tiene 2, perfecto. Si mas o menos, heurística:
        if len(group) == 1:
            # intentar buscar otra catch cercana (maybe missed detection)
            beats.append(group)
        else:
            # ordenar left then right for determinismo
            # si hay más de 2, escoger 2 con menor dist (mejores)
            group_sorted = sorted(group, key=lambda x: x['dist'])
            chosen = group_sorted[:2]
            beats.append(chosen)
        i = j
    # transform beats to list of (frame_mean, [entries])
    beats_out = []
    for g in beats:
        frame_mean = int(np.round(np.mean([c['frame'] for c in g])))
        beats_out.append({'frame': frame_mean, 'entries': g})
    return beats_out

# Calcular siteswap síncrono a partir de beats:
# Para cada beat, determinamos la pelota en la mano L y en la mano R (si faltan, usamos None)
def compute_siteswap_from_beats(beats):
    # construir lista plana de appearances: for each beat, left then right
    appearances = []  # each elem: (beat_index, ball, hand)
    for bi, b in enumerate(beats):
        # try to assign L and R
        ent = b['entries']
        left = next((e for e in ent if e['hand']=='L'), None)
        right = next((e for e in ent if e['hand']=='R'), None)
        appearances.append( (bi, left['ball'] if left else None, 'L') )
        appearances.append( (bi, right['ball'] if right else None, 'R') )
    # Now for each appearance position p (beat_i, hand), find next appearance of same ball
    N = len(appearances)
    # create result pairs per beat index -> (left_num, right_num)
    beat_pairs = []
    # map ball -> list of appearance indices (index in appearances list)
    ball_appear_indices = {}
    for idx, (bi, ball, hand) in enumerate(appearances):
        if ball is None: continue
        ball_appear_indices.setdefault(ball, []).append(idx)
    # helper to find next appearance index after idx
    def next_index_for_ball(ball, cur_idx):
        lst = ball_appear_indices.get(ball, [])
        for x in lst:
            if x > cur_idx:
                return x
        return None
    # For each beat (we have two appearances per beat at positions 2*bi, 2*bi+1)
    for bi in range(len(beats)):
        left_idx = 2*bi
        right_idx = 2*bi + 1
        left_ball = appearances[left_idx][1]
        right_ball = appearances[right_idx][1]
        pair = []
        for cur_idx, ball in [(left_idx, left_ball), (right_idx, right_ball)]:
            if ball is None:
                pair.append(None)
                continue
            nxt = next_index_for_ball(ball, cur_idx)
            if nxt is None:
                # assume wraps around: look from start
                lst = ball_appear_indices.get(ball, [])
                if len(lst) > 0:
                    nxt = lst[0]
                else:
                    pair.append(None)
                    continue
            # delta beats: compute how many beat-pairs until reappearance.
            # appearances index -> beat index = idx//2
            delta_beats = (appearances[nxt][0] - appearances[cur_idx][0]) % (len(beats))
            if delta_beats == 0:
                # full wrap
                delta_beats = len(beats)
            num = delta_beats * 2  # regla: número = beats_until * 2
            # check cross: if next appearance's hand != current hand => add 'x'
            cross = (appearances[nxt][2] != appearances[cur_idx][2])
            if cross:
                pair.append(f"{num}x")
            else:
                pair.append(str(num))
        beat_pairs.append(tuple(pair))
    return beat_pairs

# Encontrar periodo mínimo de la secuencia de beat_pairs
def find_minimal_period(pairs):
    # Representamos la secuencia como list of strings like "(4x,2)"
    seq = ['(' + (p[0] if p[0] is not None else '-') + ',' + (p[1] if p[1] is not None else '-') + ')' for p in pairs]
    s = ''.join(seq)
    # buscar el menor k tal que seq == seq[0:k] repeated
    n = len(seq)
    for k in range(1, n+1):
        if n % k != 0:
            # still can check smaller period even if not exact division (take prefix)
            candidate = seq[:k]
            ok = True
            for i in range(n):
                if seq[i] != candidate[i % k]:
                    ok = False
                    break
            if ok:
                return candidate
        else:
            candidate = seq[:k]
            if all(seq[i] == candidate[i % k] for i in range(n)):
                return candidate
    return seq  # si no repetido, devolver toda secuencia

# ---------- pipeline ----------
def pipeline(csv_path, fps=30, dist_thresh=60, debug=False):
    df = load_positions(csv_path)
    df = smooth_and_interp(df)
    vel = compute_velocities(df, fps=fps)
    catches = detect_catches(df, vel, dist_thresh=dist_thresh)
    beats = group_into_beats(catches)
    beat_pairs = compute_siteswap_from_beats(beats)
    period = find_minimal_period(beat_pairs)
    # format period as siteswap notation
    period_str = ''.join(period)
    # convert to readable tuple form: if period is like ['(4x,2)'] then return '(4x,2)'
    if isinstance(period, list):
        siteswap = ''.join(period)
    else:
        siteswap = str(period)
    return {
        'beat_pairs': beat_pairs,
        'siteswap': siteswap,
        'period_list': period
    }

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='CSV path with positions per frame')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--dist-thresh', type=float, default=60.0)
    args = parser.parse_args()
    out = pipeline(args.csv, fps=args.fps, dist_thresh=args.dist_thresh)
    print("Siteswap estimado (periodo mínimo):", out['siteswap'])
    print("Beat pairs (sample):", out['beat_pairs'][:10])
