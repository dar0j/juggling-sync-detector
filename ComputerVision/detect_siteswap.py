#!/usr/bin/env python3
"""
Detecta siteswap síncrono a partir de posiciones (manos y pelotas) en CSV.
Incluye:
 - Reidentificación de pelotas con algoritmo Húngaro.
 - Detección de catches.
 - Agrupamiento en beats síncronos.
 - Cálculo del siteswap (ej. "(4x,2x)").
 - Visualización en OpenCV con trayectorias y beats detectados.

Autor: Adaptado para dataset de Daniel Ortuño (2025)
"""

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
import os

# ---------------- CONFIG ----------------
CSV_PATH = "/home/dar0j/Documentos/2025/intro trabajo titulo el E/old rasmus/patterns/3_(4,2x)(2x,4).csv"
FPS = 30
DIST_THRESH = 60
WINDOW = 7
POLY = 2
DEBUG = True

# ----------------------------------------
def load_positions(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df.columns = [
        "x_righthand","y_righthand",
        "x_lefthand","y_lefthand",
        "x_ball1","y_ball1",
        "x_ball2","y_ball2",
        "x_ball3","y_ball3"
    ]
    return df

def smooth_positions(df):
    for col in df.columns:
        df[col] = df[col].interpolate(limit_direction='both')
        if len(df[col]) >= WINDOW:
            df[col] = savgol_filter(df[col], WINDOW, POLY)
    return df

# ----------------------------------------
# Etapa de REIDENTIFICACIÓN con Hungarian
# ----------------------------------------
def reidentify_balls(df, max_cost=100):
    """Corrige IDs de pelotas frame a frame para mantener consistencia"""
    n_frames = len(df)
    ball_cols = [("x_ball1","y_ball1"),("x_ball2","y_ball2"),("x_ball3","y_ball3")]
    n_balls = len(ball_cols)
    
    # Crear lista con todas las posiciones (frames, ball_id)
    all_positions = []
    for t in range(n_frames):
        frame_pts = []
        for xb, yb in ball_cols:
            frame_pts.append([df.loc[t, xb], df.loc[t, yb]])
        all_positions.append(np.array(frame_pts, dtype=np.float32))

    # Lista con IDs reidentificados
    tracks = np.zeros((n_frames, n_balls, 2))
    tracks[0,:,:] = all_positions[0]

    prev_pts = all_positions[0]
    for t in range(1, n_frames):
        cur_pts = all_positions[t]
        # matriz de costos
        cost = np.linalg.norm(prev_pts[:,None,:] - cur_pts[None,:,:], axis=2)
        cost[cost>max_cost] = max_cost * 2
        row_ind, col_ind = linear_sum_assignment(cost)
        reordered = np.zeros_like(cur_pts)
        for r,c in zip(row_ind, col_ind):
            reordered[r] = cur_pts[c]
        tracks[t,:,:] = reordered
        prev_pts = reordered

    # Reemplazar en df
    for i in range(n_balls):
        df[f"x_ball{i+1}"] = tracks[:,i,0]
        df[f"y_ball{i+1}"] = tracks[:,i,1]
    return df

# ----------------------------------------
# Detectar CATCHES
# ----------------------------------------
def compute_velocities(df):
    vel = {}
    dt = 1.0/FPS
    for c in df.columns:
        if c.startswith('x_'):
            base = c[2:]
            y_col = f'y_{base}'
            if y_col in df.columns:
                x = df[c].values
                y = df[y_col].values
                vx = np.gradient(x, dt)
                vy = np.gradient(y, dt)
                vel[base] = np.vstack((vx,vy)).T
    return vel

def detect_catches(df, vel, dist_thresh=60):
    catches = []
    n_frames = len(df)
    balls = [f"ball{i}" for i in range(1,4)]
    for b in balls:
        bx = df[f"x_{b}"].values
        by = df[f"y_{b}"].values
        vx, vy = vel[b][:,0], vel[b][:,1]
        for hand in ["lefthand","righthand"]:
            hx = df[f"x_{hand}"].values
            hy = df[f"y_{hand}"].values
            dist = np.sqrt((bx-hx)**2 + (by-hy)**2)
            for t in range(1, n_frames-1):
                if dist[t] < dist_thresh and dist[t] <= dist[t-1] and dist[t] <= dist[t+1]:
                    # Cambio de velocidad (acercamiento -> alejamiento)
                    vdot_prev = (bx[t-1]-hx[t-1])*vx[t-1] + (by[t-1]-hy[t-1])*vy[t-1]
                    vdot_next = (bx[t+1]-hx[t+1])*vx[t+1] + (by[t+1]-hy[t+1])*vy[t+1]
                    if vdot_prev < 0 and vdot_next > 0:
                        catches.append({"frame":t,"ball":b,"hand":"L" if hand=="lefthand" else "R"})
    # eliminar duplicados
    catches = sorted(catches, key=lambda c:(c["ball"],c["frame"]))
    filtered=[]
    for c in catches:
        if not any((c["ball"]==f["ball"] and abs(c["frame"]-f["frame"])<3) for f in filtered):
            filtered.append(c)
    return filtered

# ----------------------------------------
# Agrupar catches en beats síncronos
# ----------------------------------------
def group_into_beats(catches, frame_window=4):
    catches = sorted(catches, key=lambda c:c["frame"])
    beats=[]
    i=0
    while i < len(catches):
        base=catches[i]
        group=[base]
        j=i+1
        while j<len(catches) and abs(catches[j]["frame"]-base["frame"])<=frame_window:
            group.append(catches[j])
            j+=1
        beats.append(group)
        i=j
    beats_out=[]
    for g in beats:
        f=int(np.mean([c["frame"] for c in g]))
        beats_out.append({"frame":f,"entries":g})
    return beats_out

# ----------------------------------------
# Calcular siteswap
# ----------------------------------------
def compute_siteswap(beats):
    appearances=[]
    for bi,b in enumerate(beats):
        left=next((e for e in b["entries"] if e["hand"]=="L"),None)
        right=next((e for e in b["entries"] if e["hand"]=="R"),None)
        appearances.append((bi,left["ball"] if left else None,"L"))
        appearances.append((bi,right["ball"] if right else None,"R"))
    N=len(appearances)
    ball_indices={}
    for i,(bi,ball,hand) in enumerate(appearances):
        if ball:
            ball_indices.setdefault(ball,[]).append(i)
    beat_pairs=[]
    for bi in range(len(beats)):
        pair=[]
        for side in [0,1]:
            idx=2*bi+side
            if idx>=N: continue
            ball=appearances[idx][1]
            hand=appearances[idx][2]
            if not ball:
                pair.append("-")
                continue
            next_idx=None
            for j in ball_indices[ball]:
                if j>idx:
                    next_idx=j
                    break
            if next_idx is None:
                next_idx=ball_indices[ball][0]
            delta=(appearances[next_idx][0]-appearances[idx][0])%len(beats)
            if delta==0: delta=len(beats)
            num=delta*2
            cross = (appearances[next_idx][2]!=hand)
            pair.append(f"{num}x" if cross else str(num))
        beat_pairs.append(tuple(pair))
    return beat_pairs

def minimal_period(pairs):
    seq=['(' + p[0]+','+p[1]+')' for p in pairs]
    n=len(seq)
    for k in range(1,n+1):
        if all(seq[i]==seq[i%k] for i in range(n)):
            return seq[:k]
    return seq

# ----------------------------------------
# Visualización
# ----------------------------------------
def visualize(df, beats, catches):
    H, W = 480, 640
    scale=2
    traj_colors=[(255,0,0),(0,255,0),(0,0,255)]
    bg=np.ones((H,W,3),dtype=np.uint8)*40
    n_frames=len(df)
    ball_cols=[("x_ball1","y_ball1"),("x_ball2","y_ball2"),("x_ball3","y_ball3")]
    # precompute trajectories
    trajs=[]
    for xb,yb in ball_cols:
        pts=np.vstack((df[xb].values,df[yb].values)).T.astype(int)
        trajs.append(pts)
    catch_frames=[c["frame"] for c in catches]
    for t in range(n_frames):
        frame=bg.copy()
        # dibujar manos
        lh=(int(df.loc[t,"x_lefthand"]),int(df.loc[t,"y_lefthand"]))
        rh=(int(df.loc[t,"x_righthand"]),int(df.loc[t,"y_righthand"]))
        cv2.circle(frame,lh,6,(255,255,255),-1)
        cv2.circle(frame,rh,6,(255,255,255),-1)
        # dibujar pelotas
        for i,pts in enumerate(trajs):
            x,y=int(pts[t,0]),int(pts[t,1])
            color=traj_colors[i]
            cv2.circle(frame,(x,y),5,color,-1)
            # dibujar trayectoria
            for k in range(max(0,t-20),t):
                x1,y1=pts[k]
                x2,y2=pts[k+1]
                cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),color,1)
        # marcar catches
        if t in catch_frames:
            cv2.putText(frame,f"CATCH {t}",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        cv2.imshow("Siteswap Detection",frame)
        key=cv2.waitKey(int(1000/FPS))
        if key==27:
            break
    cv2.destroyAllWindows()

# ----------------------------------------
# MAIN
# ----------------------------------------
if __name__ == "__main__":
    df=load_positions(CSV_PATH)
    df=smooth_positions(df)
    df=reidentify_balls(df)
    vel=compute_velocities(df)
    catches=detect_catches(df,vel,DIST_THRESH)
    beats=group_into_beats(catches)
    pairs=compute_siteswap(beats)
    period=minimal_period(pairs)
    print("Secuencia completa:", pairs)
    print("Periodo mínimo:", ''.join(period))
    if DEBUG:
        visualize(df, beats, catches)
