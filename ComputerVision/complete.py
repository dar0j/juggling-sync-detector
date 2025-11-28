import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

def _moving_avg(a, w=5):
    if w <= 1: return a
    pad = w // 2
    a_pad = np.pad(a, ((pad, pad), (0, 0)), mode='edge')
    kernel = np.ones((w, 1)) / w
    return np.apply_along_axis(lambda x: np.convolve(x, kernel.ravel(), mode='valid'), 0, a_pad)

def load_positions(csv_path):
    # Columnas esperadas:
    # x_righthand, y_righthand, x_lefthand, y_lefthand, x_ball1, y_ball1, x_ball2, y_ball2, ...
    df = pd.read_csv(csv_path, header=None)
    df.columns = [
        "x_righthand","y_righthand",
        "x_lefthand","y_lefthand",
        "x_ball1","y_ball1",
        "x_ball2","y_ball2",
        "x_ball3","y_ball3"
    ]
    cols = df.columns.tolist()
    #assert cols[0:4] == ['x_righthand','y_righthand','x_lefthand','y_lefthand'], "Cabecera CSV inesperada"
    # manos
    rh = df[['x_righthand','y_righthand']].to_numpy()
    lh = df[['x_lefthand','y_lefthand']].to_numpy()
    # pelotas
    balls = []
    i = 4
    ball_idx = 0
    while i+1 < len(cols):
        bx = cols[i]; by = cols[i+1]
        if not (bx.startswith('x_ball') and by.startswith('y_ball')): break
        balls.append(df[[bx,by]].to_numpy())
        i += 2; ball_idx += 1
    return rh, lh, balls

def smooth_series(rh, lh, balls, window=5):
    rh_s = _moving_avg(rh, window)
    lh_s = _moving_avg(lh, window)
    balls_s = [_moving_avg(b, window) for b in balls]
    return rh_s, lh_s, balls_s

def compute_velocities(arr, fps):
    # diferencia simple (pix/frame) -> (pix/s) si multiplicas por fps; aquí basta criterio relativo
    vel = np.vstack([np.zeros((1,2)), np.diff(arr, axis=0)])
    return vel

def detect_catches(rh, lh, balls, fps=60, d_thresh=50.0, v_thresh=2.0):
    # Devuelve lista de catches: (frame, 'L'|'R', ball_id)
    catches = []
    nframes = rh.shape[0]
    nb = len(balls)
    # precompute velocidades de pelotas
    ball_vel = [compute_velocities(b, fps) for b in balls]
    for t in range(1, nframes-1):
        for b_id in range(nb):
            b = balls[b_id]
            v = np.linalg.norm(ball_vel[b_id][t])
            # distancias a manos
            dL = np.linalg.norm(b[t] - lh[t])
            dR = np.linalg.norm(b[t] - rh[t])
            # mínimos locales de distancia (aprox. "contacto") + velocidad baja
            # izquierda
            if dL < d_thresh:
                prev = np.linalg.norm(b[t-1] - lh[t-1])
                nxt = np.linalg.norm(b[t+1] - lh[t+1])
                if dL <= prev and dL <= nxt and v < v_thresh:
                    catches.append((t, 'L', b_id))
                    continue
            # derecha
            if dR < d_thresh:
                prev = np.linalg.norm(b[t-1] - rh[t-1])
                nxt = np.linalg.norm(b[t+1] - rh[t+1])
                if dR <= prev and dR <= nxt and v < v_thresh:
                    catches.append((t, 'R', b_id))
    catches.sort(key=lambda x: x[0])
    return catches

def group_into_beats(catches, frame_window=4):
    # Agrupa catches cercanos en el mismo beat síncrono
    # Retorna lista de beats: [(L_ball_id|None, R_ball_id|None), ...]
    beats = []
    if not catches: return beats
    cur_start = catches[0][0]
    L, R = None, None
    for (t, side, b_id) in catches:
        if t - cur_start > frame_window:
            beats.append((L, R))
            cur_start = t
            L, R = None, None
        if side == 'L': L = b_id
        else: R = b_id
    beats.append((L, R))
    return beats

def compute_sync_siteswap_from_beats(beats):
    # Para cada beat i, para L y R, busca la próxima aparición del mismo ball_id (en L o R)
    # altura = 2 * (j - i); si cambia de mano entre i y j, añade 'x'
    n = len(beats)
    # mapa de ocurrencias por pelota
    occurrences = {}  # ball_id -> list of (beat_index, 'L'|'R')
    for i,(L,R) in enumerate(beats):
        if L is not None: occurrences.setdefault(L, []).append((i,'L'))
        if R is not None: occurrences.setdefault(R, []).append((i,'R'))
    # para búsqueda rápida
    occ_by_side = {}
    for b_id, occs in occurrences.items():
        occ_by_side[b_id] = occs

    pairs = []
    for i,(L,R) in enumerate(beats):
        def value_for(side, b_id):
            if b_id is None: return '0'  # beat vacío
            occs = occ_by_side[b_id]
            # localizar esta ocurrencia (i, side) en la lista y tomar la siguiente
            # si hay duplicados en el mismo beat por ruido, usamos la primera coincidencia exacta
            idx = None
            for k,(ii,ss) in enumerate(occs):
                if ii == i and ss == side:
                    idx = k; break
            if idx is None:
                # fallback: buscar por beat i sin importar side
                for k,(ii,ss) in enumerate(occs):
                    if ii == i: idx = k; side = ss; break
            # próxima ocurrencia
            j = None; side_j = side
            for k in range((idx or 0)+1, len(occs)):
                j, side_j = occs[k]
                if j > i: break
            if j is None or j <= i:
                return '-'  # no cierra ciclo dentro del clip
            height = 2 * (j - i)
            cross = (side != side_j)
            return f"{height}{'x' if cross else ''}"
        pairs.append((value_for('L',L), value_for('R',R)))
    return pairs

def minimal_period_pairs(pairs):
    # Equivalente a testparts.minimal_period pero sobre pares formateados "(a,b)"
    seq = ['(' + (p[0] if p[0] is not None else '-') + ',' + (p[1] if p[1] is not None else '-') + ')' for p in pairs]
    n = len(seq)
    for k in range(1, n+1):
        if all(seq[i] == seq[i % k] for i in range(n)):
            return seq[:k]
    return seq

def canonical_rotation(seq):
    # rotación lexicográficamente mínima
    rots = [seq[k:] + seq[:k] for k in range(len(seq))]
    return min(rots)

class BallKF:
    def __init__(self, x, y):
        self.state = np.array([x, y, 0., 0.], dtype=float)  # x,y,vx,vy
        self.P = np.eye(4) * 50.
    def predict(self, dt):
        F = np.array([[1,0,dt,0],
                      [0,1,0,dt],
                      [0,0,1,0],
                      [0,0,0,1]], dtype=float)
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + np.eye(4) * 0.5
        return self.state[:2]
    def update(self, z):
        H = np.array([[1,0,0,0],
                      [0,1,0,0]], dtype=float)
        y = z - H @ self.state
        S = H @ self.P @ H.T + np.eye(2) * 2.
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

def reidentify_balls_kf(balls, fps=60, max_cost=150):
    """
    balls: lista de arrays (frames,2) en orden inicial (puede variar).
    Devuelve lista reordenada estable usando predicción Kalman + Hungarian.
    """
    if not balls:
        return balls
    nb = len(balls)
    nframes = balls[0].shape[0]
    # stack detections
    detections = np.stack(balls, axis=1)  # (frames, nb, 2)
    # inicializar filtros
    kfs = [BallKF(detections[0,i,0], detections[0,i,1]) for i in range(nb)]
    assigned = np.zeros_like(detections)
    assigned[0] = detections[0]
    dt = 1.0 / fps
    prev_pred = [kf.state[:2].copy() for kf in kfs]

    for t in range(1, nframes):
        cur = detections[t]  # (nb,2)
        # predicciones
        preds = np.array([kf.predict(dt) for kf in kfs])  # (nb,2)
        # matriz de costos
        cost = np.linalg.norm(preds[:, None, :] - cur[None, :, :], axis=2)
        # gating
        gated = cost.copy()
        gated[gated > max_cost] = max_cost * 5
        rows, cols = linear_sum_assignment(gated)
        # reordenar según asignación (row = filtro, col = detección)
        frame_assigned = np.zeros_like(cur)
        for r, c in zip(rows, cols):
            frame_assigned[r] = cur[c]
            kfs[r].update(cur[c])
        # fallback si alguna fila quedó en cero (detección inválida)
        for r in range(nb):
            if np.all(frame_assigned[r] == 0):
                frame_assigned[r] = preds[r]
        assigned[t] = frame_assigned

    # separar por bola
    out = [assigned[:, i, :] for i in range(nb)]
    return out

def pipeline(csv_path, fps=60, smooth_window=5, d_thresh=50.0, v_thresh=2.0, frame_window=4, use_kf=False, max_cost=150):
    rh, lh, balls = load_positions(csv_path)
    if use_kf:
        balls = reidentify_balls_kf(balls, fps=fps, max_cost=max_cost)
    rh, lh, balls = smooth_series(rh, lh, balls, window=smooth_window)
    catches = detect_catches(rh, lh, balls, fps=fps, d_thresh=d_thresh, v_thresh=v_thresh)
    beats = group_into_beats(catches, frame_window=frame_window)
    pairs = compute_sync_siteswap_from_beats(beats)
    seq_str = ''.join([f"({a},{b})" for a,b in pairs])
    period_list = minimal_period_pairs(pairs)
    period_canon = canonical_rotation(period_list)
    period_str = ''.join(period_canon)
    return {
        'beats': beats,
        'pairs': pairs,
        'siteswap_full': seq_str,
        'siteswap_period': period_str
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--fps', type=int, default=60)
    ap.add_argument('--smooth', type=int, default=5)
    ap.add_argument('--d', type=float, default=50.0)
    ap.add_argument('--v', type=float, default=2.0)
    ap.add_argument('--win', type=int, default=4)
    ap.add_argument('--kf', action='store_true', help='Activar reidentificación Kalman+Hungarian')
    ap.add_argument('--maxcost', type=float, default=150.0, help='Umbral distancia para gating')
    args = ap.parse_args()
    res = pipeline(args.csv, fps=args.fps, smooth_window=args.smooth, d_thresh=args.d, v_thresh=args.v, frame_window=args.win, use_kf=args.kf, max_cost=args.maxcost)
    print("FULL:", res['siteswap_full'])
    print("PERIOD:", res['siteswap_period'])