#!/usr/bin/env python3
"""
Detecta siteswap síncrono a partir de posiciones (manos y pelotas) en CSV.
Incluye:
 - Reidentificación de pelotas (Hungarian)
 - Filtro de Kalman (para suavizar y predecir trayectorias)
 - Detección de catches
 - Agrupamiento en beats síncronos
 - Cálculo del siteswap
 - Visualización en OpenCV
"""

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# --- CONFIG ---
CSV_PATH = "/home/dar0j/Documentos/2025/intro trabajo titulo el E/old rasmus/patterns/3_(4,2x)(2x,4).csv"
FPS = 30
DIST_THRESH = 60
DEBUG = True

# --------------------- KALMAN FILTER ---------------------
class Kalman2D:
    """Filtro de Kalman simple (x, y, vx, vy)"""
    def __init__(self, dt=1/30, process_var=1e-2, meas_var=1.0):
        self.dt = dt
        self.A = np.array([[1,0,dt,0],
                           [0,1,0,dt],
                           [0,0,1,0],
                           [0,0,0,1]])
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]])
        self.Q = np.eye(4)*process_var
        self.R = np.eye(2)*meas_var
        self.P = np.eye(4)
        self.x = np.zeros((4,1))
        self.initialized = False

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2].flatten()

    def update(self, z):
        z = np.reshape(z,(2,1))
        if not self.initialized:
            self.x[:2,0] = z.flatten()
            self.initialized = True
            return self.x[:2].flatten()
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        return self.x[:2].flatten()

# --------------------- FUNCIONES ---------------------
def load_positions(path):
    df = pd.read_csv(path, header=None)
    df.columns = [
        "x_righthand","y_righthand",
        "x_lefthand","y_lefthand",
        "x_ball1","y_ball1",
        "x_ball2","y_ball2",
        "x_ball3","y_ball3"
    ]
    return df

def reidentify_balls(df, max_cost=100):
    n_frames=len(df)
    ball_cols=[("x_ball1","y_ball1"),("x_ball2","y_ball2"),("x_ball3","y_ball3")]
    n_balls=len(ball_cols)
    tracks=np.zeros((n_frames,n_balls,2))
    tracks[0,:,:]=np.array([[df.loc[0,xb],df.loc[0,yb]] for xb,yb in ball_cols])
    prev_pts=tracks[0]
    for t in range(1,n_frames):
        cur_pts=np.array([[df.loc[t,xb],df.loc[t,yb]] for xb,yb in ball_cols])
        cost=np.linalg.norm(prev_pts[:,None,:]-cur_pts[None,:,:],axis=2)
        cost[cost>max_cost]=max_cost*2
        row,col=linear_sum_assignment(cost)
        reordered=np.zeros_like(cur_pts)
        for r,c in zip(row,col):
            reordered[r]=cur_pts[c]
        tracks[t,:,:]=reordered
        prev_pts=reordered
    for i in range(n_balls):
        df[f"x_ball{i+1}"]=tracks[:,i,0]
        df[f"y_ball{i+1}"]=tracks[:,i,1]
    return df

def apply_kalman(df, fps=30):
    """Aplica filtro de Kalman 2D a cada pelota"""
    n_frames=len(df)
    ball_cols=[("x_ball1","y_ball1"),("x_ball2","y_ball2"),("x_ball3","y_ball3")]
    for xb,yb in ball_cols:
        kf=Kalman2D(dt=1/fps)
        smoothed=[]
        for t in range(n_frames):
            meas=np.array([df.loc[t,xb],df.loc[t,yb]])
            pred=kf.predict()
            est=kf.update(meas)
            smoothed.append(est)
        smoothed=np.array(smoothed)
        df[xb]=smoothed[:,0]
        df[yb]=smoothed[:,1]
    return df

def detect_catches(df, dist_thresh=60):
    catches=[]
    n=len(df)
    balls=[f"ball{i}" for i in range(1,4)]
    for b in balls:
        bx=df[f"x_{b}"].values
        by=df[f"y_{b}"].values
        for hand in ["lefthand","righthand"]:
            hx=df[f"x_{hand}"].values
            hy=df[f"y_{hand}"].values
            dist=np.sqrt((bx-hx)**2+(by-hy)**2)
            for t in range(1,n-1):
                if dist[t]<dist_thresh and dist[t]<=dist[t-1] and dist[t]<=dist[t+1]:
                    catches.append({"frame":t,"ball":b,"hand":"L" if hand=="lefthand" else "R"})
    catches=sorted(catches,key=lambda x:(x["ball"],x["frame"]))
    filtered=[]
    for c in catches:
        if not any((c["ball"]==f["ball"] and abs(c["frame"]-f["frame"])<3) for f in filtered):
            filtered.append(c)
    return filtered

def group_into_beats(catches, frame_window=4):
    catches=sorted(catches,key=lambda c:c["frame"])
    beats=[]
    i=0
    while i<len(catches):
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
            lst=ball_indices[ball]
            next_idx=None
            for j in lst:
                if j>idx:
                    next_idx=j
                    break
            if next_idx is None:
                next_idx=lst[0]
            delta=(appearances[next_idx][0]-appearances[idx][0])%len(beats)
            if delta==0: delta=len(beats)
            num=delta*2
            cross=appearances[next_idx][2]!=hand
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

# --- Visualización simple ---
def visualize(df, catches):
    H,W=480,640
    colors=[(255,0,0),(0,255,0),(0,0,255)]
    ball_cols=[("x_ball1","y_ball1"),("x_ball2","y_ball2"),("x_ball3","y_ball3")]
    n=len(df)
    trajs=[np.vstack((df[x].values,df[y].values)).T.astype(int) for x,y in ball_cols]
    catch_frames=[c["frame"] for c in catches]
    for t in range(n):
        frame=np.ones((H,W,3),np.uint8)*30
        lh=(int(df.loc[t,"x_lefthand"]),int(df.loc[t,"y_lefthand"]))
        rh=(int(df.loc[t,"x_righthand"]),int(df.loc[t,"y_righthand"]))
        cv2.circle(frame,lh,6,(255,255,255),-1)
        cv2.circle(frame,rh,6,(255,255,255),-1)
        for i,tr in enumerate(trajs):
            x,y=tr[t]
            cv2.circle(frame,(x,y),5,colors[i],-1)
            if t>1:
                cv2.line(frame,tuple(tr[t-1]),tuple(tr[t]),colors[i],1)
        if t in catch_frames:
            cv2.putText(frame,f"CATCH {t}",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        cv2.imshow("Kalman Siteswap",frame)
        key=cv2.waitKey(int(1000/FPS))
        if key==27:
            break
    cv2.destroyAllWindows()

# --- MAIN ---
if __name__ == "__main__":
    df=load_positions(CSV_PATH)
    df=reidentify_balls(df)
    df=apply_kalman(df,FPS)
    catches=detect_catches(df,DIST_THRESH)
    beats=group_into_beats(catches)
    pairs=compute_siteswap(beats)
    period=minimal_period(pairs)
    print("Secuencia completa:", pairs)
    print("Periodo mínimo:", ''.join(period))
    if DEBUG:
        visualize(df, catches)
