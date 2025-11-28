import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def load_raw(path):
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 6:
        raise ValueError("CSV insuficiente: se esperan manos + pelotas.")
    if (df.shape[1] - 4) % 2 != 0:
        raise ValueError("Número de columnas de pelotas inválido.")
    df.columns = [
        "x_righthand","y_righthand",
        "x_lefthand","y_lefthand",
        "x_ball1","y_ball1",
        "x_ball2","y_ball2",
        "x_ball3","y_ball3"
    ]
    return df

def reidentify_balls(df, max_cost=150):
    """
    Reidentifica pelotas para mantener IDs consistentes:
    - Para cada frame construye matriz de costos (distancias euclídeas) entre
      posiciones previas (orden actual de bolas) y nuevas.
    - Aplica Hungarian (linear_sum_assignment) para mínima suma total.
    - Reordena las posiciones del frame según el emparejamiento (intercambia columnas virtualmente).
    - Sustituye columnas x_ball{i}, y_ball{i} por el orden reidentificado.
    """
    ball_pairs = [(c, c.replace("x_", "y_")) for c in df.columns if c.startswith("x_ball")]
    n_balls = len(ball_pairs)
    if n_balls == 0:
        return df

    # Matriz [frames, balls, 2]
    positions = np.zeros((len(df), n_balls, 2), dtype=np.float32)
    for bi, (xc, yc) in enumerate(ball_pairs):
        xs = df[xc].astype(float).to_numpy()
        ys = df[yc].astype(float).to_numpy()
        # interpolar y rellenar NaN
        xs = pd.Series(xs).interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill').to_numpy()
        ys = pd.Series(ys).interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill').to_numpy()
        positions[:, bi, 0] = xs
        positions[:, bi, 1] = ys

    tracks = np.zeros_like(positions)
    tracks[0] = positions[0]
    prev = tracks[0]

    for t in range(1, len(df)):
        cur = positions[t]
        cost = np.linalg.norm(prev[:, None, :] - cur[None, :, :], axis=2)
        # Penalizar distancias grandes
        cost[cost > max_cost] = max_cost * 4
        row_ind, col_ind = linear_sum_assignment(cost)
        reordered = np.zeros_like(cur)
        for r, c in zip(row_ind, col_ind):
            reordered[r] = cur[c]
        # Fallback si queda fila vacía (ruido)
        for r in range(n_balls):
            if np.all(reordered[r] == 0):
                reordered[r] = prev[r]
        tracks[t] = reordered
        prev = reordered

    # Sustituir en df
    for bi, (xc, yc) in enumerate(ball_pairs):
        df[xc] = tracks[:, bi, 0]
        df[yc] = tracks[:, bi, 1]
    return df

def save_output(df, assigned, out_path):
    # Convertir a enteros (redondeo)
    assigned_int = np.rint(assigned).astype(int)
    out = pd.DataFrame()
    # manos (también forzar entero)
    for i in range(4):
        out[i] = df.iloc[:, i].astype(int)
    n_balls = assigned_int.shape[1]
    col_idx = 4
    for b in range(n_balls):
        out[col_idx] = assigned_int[:, b, 0]
        out[col_idx+1] = assigned_int[:, b, 1]
        col_idx += 2
    # Guardar sin header ni índices, valores puros enteros
    out.to_csv(out_path, header=False, index=False)
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Reidentificación (Hungarian) de pelotas en CSV (enteros para playpattern).")
    ap.add_argument("--csv", required=True, help="Ruta CSV original (sin cabecera).")
    ap.add_argument("--out", default=None, help="Ruta CSV salida (default *_hungarian.csv).")
    ap.add_argument("--max-cost", type=float, default=150.0)
    args = ap.parse_args()

    df = load_raw(args.csv)
    assigned = reidentify_balls(df, max_cost=args.max_cost)
    out_path = args.out or (args.csv.rsplit(".csv",1)[0] + "_hungarian.csv")
    save_output(df, assigned, out_path)
    print("Guardado:", out_path)

if __name__ == "__main__":
    main()