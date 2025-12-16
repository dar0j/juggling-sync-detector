import numpy as np
import pandas as pd

def detect_catches_robust(df, fps=30,
                          dist_thresh=60,
                          speed_thresh_pxsec=120,
                          min_dwell_frames=3,
                          win_pre=4, win_post=6):
    """
    Detecta catches usando:
      - dist < dist_thresh
      - speed_smooth < speed_thresh (px/s)
      - dwell mínimo en ventana (min_dwell_frames)
      - comprobación opcional de acercamiento previo (win_pre) y alejamiento posterior (win_post)
    Devuelve lista de dicts: {'frame':, 'ball':, 'hand':, 'frame_start', 'frame_end'}
    """

    # convertir umbral velocidad a px/frame
    speed_thresh = speed_thresh_pxsec / fps

    n = len(df)
    balls = [f"ball{i}" for i in range(1,4)]
    catches = []

    # calcular velocidades por pelota (central diff) y suavizar (movil)
    vel = {}
    for b in balls:
        bx = df[f'x_{b}'].values
        by = df[f'y_{b}'].values
        vx = np.gradient(bx)  # px/frame
        vy = np.gradient(by)
        speed = np.sqrt(vx*vx + vy*vy)
        # suavizar speed con mediana móvil de 3
        speed_smooth = pd.Series(speed).rolling(window=3, center=True, min_periods=1).median().values
        vel[b] = {'vx': vx, 'vy': vy, 'speed': speed_smooth}

    for b in balls:
        bx = df[f'x_{b}'].values
        by = df[f'y_{b}'].values
        for hand in ['lefthand', 'righthand']:
            hx = df[f'x_{hand}'].values
            hy = df[f'y_{hand}'].values
            dist = np.sqrt((bx-hx)**2 + (by-hy)**2)
            t = 0
            while t < n:
                if dist[t] < dist_thresh and vel[b]['speed'][t] < speed_thresh:
                    # posible inicio de dwell
                    t0 = t
                    # extender mientras se cumpla
                    te = t0
                    while te+1 < n and dist[te+1] < dist_thresh and vel[b]['speed'][te+1] < speed_thresh:
                        te += 1
                    dwell_len = te - t0 + 1
                    if dwell_len >= min_dwell_frames:
                        # comprobar pre-condición: en win_pre frames previos hubo acercamiento promedio
                        pre_ok = False
                        start_check = max(0, t0 - win_pre)
                        if start_check < t0:
                            # vector radial y velocidad media en ventana previa
                            r_prev = np.vstack((bx[start_check:t0] - hx[start_check:t0],
                                                by[start_check:t0] - hy[start_check:t0])).T
                            v_prev = np.vstack((vel[b]['vx'][start_check:t0],
                                                vel[b]['vy'][start_check:t0])).T
                            # producto punto medio (sumado)
                            if len(r_prev)>0:
                                dot = np.mean((r_prev * v_prev).sum(axis=1))
                                pre_ok = (dot < 0)
                        else:
                            pre_ok = True  # no hay suficiente histórico -> permitir

                        # comprobar post-condición (alejamiento) opcional
                        post_ok = False
                        end_check = min(n, te + 1 + win_post)
                        r_post = np.vstack((bx[te+1:end_check] - hx[te+1:end_check],
                                            by[te+1:end_check] - hy[te+1:end_check])).T if te+1<end_check else np.empty((0,2))
                        v_post = np.vstack((vel[b]['vx'][te+1:end_check],
                                            vel[b]['vy'][te+1:end_check])).T if te+1<end_check else np.empty((0,2))
                        if len(r_post)>0:
                            dot_post = np.mean((r_post * v_post).sum(axis=1))
                            post_ok = (dot_post > 0)
                        else:
                            post_ok = True  # no datos -> permitir

                        # decidir: preferir casos con pre_ok True; si no, aun así aceptar si dwell grande
                        if pre_ok or dwell_len >= max(4, min_dwell_frames):
                            catches.append({'frame': int((t0+te)//2), 'frame_start': t0, 'frame_end': te,
                                            'ball': b, 'hand': 'L' if hand=='lefthand' else 'R',
                                            'dwell': dwell_len})
                        t = te + 1
                    else:
                        t = te + 1
                else:
                    t += 1
    # eliminar solapamientos por mismo ball (mantener primero)
    catches_sorted = sorted(catches, key=lambda x: (x['ball'], x['frame_start']))
    filtered = []
    for c in catches_sorted:
        if not any((c['ball']==f['ball'] and not (c['frame_end'] < f['frame_start'] or c['frame_start'] > f['frame_end'])) for f in filtered):
            filtered.append(c)
    return filtered
