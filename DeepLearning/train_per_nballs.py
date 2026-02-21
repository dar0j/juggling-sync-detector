#!/usr/bin/env python3
"""
train_per_nballs.py
Entrena un modelo TCN separado por cada número de pelotas (3, 4, 5, 6).
Lee CSVs generados por pipeline_track_to_csv.py (solo coordenadas de pelotas, sin manos).

Uso:
  python train_per_nballs.py --data_root runs/track_csvs --n_balls 3
  python train_per_nballs.py --data_root runs/track_csvs  # todos
"""
import argparse
import os
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, balanced_accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASK_VALUE = -1.0
TARGET_FPS = 60
SEQ_LEN_1SEC = 60  # 1 segundo a 60fps
AUG_SUFFIXES = ["_flip", "_noise", "_crop", "_shuffle"]


# ── Velocity features ───────────────────────────────────────────────────────────
def compute_velocity_features(pos_seq: np.ndarray) -> np.ndarray:
    """
    Calcula vx, vy por pelota usando np.gradient sobre posiciones ya normalizadas.
    Los frames con MASK_VALUE (-1) no se tocan: sus velocidades también quedan en -1.

    pos_seq: (n_frames, n_balls*2)  columnas: x0,y0, x1,y1, ...
    returns: (n_frames, n_balls*2)  columnas: vx0,vy0, vx1,vy1, ...
    """
    vel = np.full_like(pos_seq, MASK_VALUE)
    for col in range(pos_seq.shape[1]):
        col_data = pos_seq[:, col]
        valid_idx = np.where(col_data != MASK_VALUE)[0]
        if valid_idx.size < 2:
            continue
        grads = np.gradient(col_data[valid_idx])   # derivada central
        vel[valid_idx, col] = grads.astype(np.float32)
    return vel


def build_feature_sequence(data: np.ndarray) -> np.ndarray:
    """
    Dado un array de posiciones (n_frames, n_balls*2):
      1. normaliza z-score por columna (solo frames válidos)
      2. calcula velocidades sobre las posiciones normalizadas
      3. concatena → (n_frames, n_balls*4)  [x,y,vx,vy] por pelota
    """
    n_balls_2 = data.shape[1]

    # ── 1. z-score posiciones ──────────────────────────────────────────────────
    pos = data.copy()
    for col in range(n_balls_2):
        valid = pos[:, col] != MASK_VALUE
        if valid.sum() < 2:
            continue
        v = pos[valid, col]
        pos[valid, col] = (v - v.mean()) / (v.std() + 1e-8)

    # ── 2. velocidades sobre posiciones normalizadas ───────────────────────────
    vel = compute_velocity_features(pos)

    # ── 3. interleave: x0,y0,vx0,vy0, x1,y1,vx1,vy1, ... ────────────────────
    n_frames = pos.shape[0]
    n_balls = n_balls_2 // 2
    out = np.full((n_frames, n_balls * 4), MASK_VALUE, dtype=np.float32)
    for b in range(n_balls):
        out[:, b*4 + 0] = pos[:, b*2 + 0]   # x
        out[:, b*4 + 1] = pos[:, b*2 + 1]   # y
        out[:, b*4 + 2] = vel[:, b*2 + 0]   # vx
        out[:, b*4 + 3] = vel[:, b*2 + 1]   # vy

    return out   # (n_frames, n_balls*4)


# ── Resampling ────────────────────────────────────────────────────────────────
def resample_to_target_fps(data: np.ndarray, current_fps: float, target_fps: int = 60) -> np.ndarray:
    """Interpolar datos para normalizar la base temporal."""
    if abs(current_fps - target_fps) < 0.1:
        return data

    n_frames = data.shape[0]
    duration = n_frames / current_fps
    new_n_frames = int(duration * target_fps)
    
    old_idx = np.arange(n_frames)
    new_idx = np.linspace(0, n_frames - 1, new_n_frames)
    
    resampled = np.full((new_n_frames, data.shape[1]), MASK_VALUE, dtype=np.float32)
    
    for col in range(data.shape[1]):
        col_data = data[:, col]
        valid_mask = col_data != MASK_VALUE
        if valid_mask.sum() < 2:
            continue
        
        # Interpolar linealmente entre puntos válidos
        resampled[:, col] = np.interp(
            new_idx, 
            old_idx[valid_mask], 
            col_data[valid_mask],
            left=MASK_VALUE,
            right=MASK_VALUE
        )
    return resampled


# ── Data loading ────────────────────────────────────────────────────────────────
def parse_filename(fname: str):
    """
    '3_(0,6)_2'        -> (3, '(0,6)', '2')
    '3_(0,6)_2_flip'   -> (3, '(0,6)', '2')   ← sufijo de aug ignorado
    """
    # 1. Quitar sufijos de augmentación antes de parsear
    clean = fname
    for suf in AUG_SUFFIXES:
        if clean.endswith(suf):
            clean = clean[:-len(suf)]
            break

    tokens = clean.split("_")
    if len(tokens) < 2:
        return None, None, None
    try:
        nb = int(tokens[0])
    except ValueError:
        return None, None, None

    # El último token es el ID de sesión si es dígito
    if tokens[-1].isdigit() and len(tokens) > 2:
        trick = "_".join(tokens[1:-1])
        sid = tokens[-1]
    else:
        trick = "_".join(tokens[1:])
        sid = "0"

    return nb, trick, sid


def load_dataset_for_nballs(data_root: str, n_balls: int,
                             seq_len: int = SEQ_LEN_1SEC,
                             use_sliding_window: bool = True,
                             cache_dir: str = "/content/drive/MyDrive/runs/dets_cache_all"):
    """
    Carga CSVs de runs/track_csvs/{n_balls}b/*.csv
    
    Cada CSV: (n_frames, n_balls*2) sin header, coordenadas de pelotas.
    
    Si use_sliding_window=True: genera ventanas deslizantes de seq_len frames
      (como en rasmus/patterndataloader.py) para más muestras de entrenamiento.
    Si use_sliding_window=False: usa secuencia completa con padding temporal.
    
    Returns: X, y, num_classes, label_map, class_names
    """
    folder = os.path.join(data_root, f"{n_balls}b")
    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    if not csv_files:
        print(f"  No se encontraron CSVs en {folder}")
        return None, None, 0, {}, []

    n_pos_features = n_balls * 2        # x,y
    n_features     = n_balls * 4        # x,y,vx,vy  ← nuevo

    label_map = {}
    next_label = 0
    X_list, y_list = [], []

    for path in csv_files:
        fname = os.path.basename(path)[:-4]
        nb, trick, sid = parse_filename(fname)   # ya viene limpio
        if nb is None or nb != n_balls:
            continue

        trick_clean = trick.lower()   # sin _0, _1, sin sufijos aug
        
        if trick_clean not in label_map:
            label_map[trick_clean] = next_label
            next_label += 1
        y = label_map[trick_clean]

        # 1. Obtener FPS reales del cache NDJSON
        current_fps = 30.0
        
        # Para augmentados, buscar el ndjson del original
        base_fname = fname
        for suffix in ["_flip", "_noise", "_crop"]:
            if base_fname.endswith(suffix):
                base_fname = base_fname[:-len(suffix)]
                break
        
        ndjson_path = Path(cache_dir) / f"{base_fname}.ndjson"
        if ndjson_path.exists():
            try:
                with open(ndjson_path, 'r') as f:
                    meta = json.loads(f.readline())["meta"]
                    current_fps = meta.get("real_fps", meta.get("fps", 30.0))
            except:
                pass
        
        raw = pd.read_csv(path, header=None).values.astype(np.float32)

        # 2. Resamplear a 60fps antes de procesar
        raw = resample_to_target_fps(raw, current_fps, TARGET_FPS)

        # 3. Construir [x, y, vx, vy] (ahora basado en 60fps)
        sequence = build_feature_sequence(raw)   # (n_frames, n_balls*4)

        n_frames = sequence.shape[0]   # ← era data.shape[0], variable inexistente aquí
        if use_sliding_window:
            if n_frames < seq_len:
                # Pad temporal
                padded = np.full((seq_len, n_features), MASK_VALUE, dtype=np.float32)
                padded[:n_frames] = sequence
                X_list.append(padded)
                y_list.append(y)
            else:
                for start in range(n_frames - seq_len + 1):
                    X_list.append(sequence[start:start + seq_len].copy())
                    y_list.append(y)
        else:
            X_list.append(sequence)
            y_list.append(y)

    if not X_list:
        return None, None, 0, {}, []

    num_classes  = next_label
    class_names  = {v: k for k, v in label_map.items()}

    if use_sliding_window:
        # Todas las ventanas tienen el mismo largo
        X = np.array(X_list, dtype=np.float32)
    else:
        # Pad temporal al máximo
        max_len = max(s.shape[0] for s in X_list)
        X = np.full((len(X_list), max_len, n_features), MASK_VALUE, dtype=np.float32)
        for i, seq in enumerate(X_list):
            X[i, :seq.shape[0]] = seq

    y_arr = np.array(y_list, dtype=np.int32)
    return X, y_arr, num_classes, label_map, class_names


# ── Modelo ──────────────────────────────────────────────────────────────────────
def build_model(n_features: int, num_classes: int,
                filters_base=64, kernel_size=7, dilation=4,
                dense_units=256, dropout=0.5, lr=0.001):
    inp = layers.Input(shape=(None, n_features), name="coords")
    x = layers.Masking(mask_value=MASK_VALUE)(inp)
    x = layers.Conv1D(filters_base, kernel_size, activation="relu")(x)
    x = layers.Conv1D(filters_base * 2, 5, padding="same",
                       activation="relu", dilation_rate=dilation)(x)
    x = layers.Conv1D(filters_base * 2, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def load_videos_raw(data_root, n_balls, cache_dir):
    """
    Carga secuencias completas SIN sliding window.
    Returns: list de (sequence, label, video_id)
    """
    folder = os.path.join(data_root, f"{n_balls}b")
    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    label_map = {}
    next_label = 0
    videos = []

    for path in csv_files:
        fname = os.path.basename(path)[:-4]
        nb, trick, sid = parse_filename(fname)
        if nb is None or nb != n_balls:
            continue

        trick_clean = trick.lower()
        if trick_clean not in label_map:
            label_map[trick_clean] = next_label
            next_label += 1

        current_fps = 30.0
        base_fname = fname
        for suffix in AUG_SUFFIXES:
            if base_fname.endswith(suffix):
                base_fname = base_fname[:-len(suffix)]
                break

        ndjson_path = Path(cache_dir) / f"{base_fname}.ndjson"
        if ndjson_path.exists():
            try:
                with open(ndjson_path, 'r') as f:
                    meta = json.loads(f.readline())["meta"]
                    current_fps = meta.get("real_fps", meta.get("fps", 30.0))
            except:
                pass

        raw = pd.read_csv(path, header=None).values.astype(np.float32)
        raw = resample_to_target_fps(raw, current_fps, TARGET_FPS)
        sequence = build_feature_sequence(raw)

        videos.append((sequence, label_map[trick_clean], fname))

    class_names = {v: k for k, v in label_map.items()}
    return videos, label_map, class_names


def videos_to_windows(video_list, seq_len, n_features):
    """Aplica sliding window a una lista de videos."""
    X_list, y_list = [], []
    for seq, label, _ in video_list:
        n_frames = seq.shape[0]
        if n_frames < seq_len:
            padded = np.full((seq_len, n_features), MASK_VALUE, dtype=np.float32)
            padded[:n_frames] = seq
            X_list.append(padded)
            y_list.append(label)
        else:
            for start in range(n_frames - seq_len + 1):
                X_list.append(seq[start:start + seq_len].copy())
                y_list.append(label)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

def predict_video(model, sequence: np.ndarray, seq_len: int,
                  num_classes: int, aggregation: str = "prob_sum") -> dict:
    """
    Clasifica un video completo usando ventanas deslizantes.

    aggregation:
      'prob_sum'   → suma probabilidades de todas las ventanas (recomendado)
      'majority'   → voto mayoritario por conteo de predicciones
      'max_conf'   → retorna la predicción de la ventana con mayor confianza

    Returns dict con:
      'pred_class': clase predicha (int)
      'confidence': confianza [0-1]
      'class_probs': array (num_classes,) con probabilidades agregadas
      'window_preds': lista de predicciones por ventana
      'n_windows': número de ventanas procesadas
    """
    n_frames = sequence.shape[0]
    n_features = sequence.shape[1]

    # Generar ventanas
    windows = []
    if n_frames < seq_len:
        # Pad si el video es muy corto
        padded = np.full((seq_len, n_features), MASK_VALUE, dtype=np.float32)
        padded[:n_frames] = sequence
        windows.append(padded)
    else:
        for start in range(n_frames - seq_len + 1):
            windows.append(sequence[start:start + seq_len])

    if not windows:
        return None

    windows_arr = np.array(windows, dtype=np.float32)   # (n_windows, seq_len, n_features)
    all_probs = model.predict(windows_arr, verbose=0)    # (n_windows, num_classes)

    window_preds = np.argmax(all_probs, axis=1)          # clase por ventana

    if aggregation == "prob_sum":
        # Sumar distribuciones de probabilidad → más informativo que votar
        agg_probs = all_probs.sum(axis=0)
        agg_probs /= agg_probs.sum()
        pred_class = int(np.argmax(agg_probs))
        confidence = float(agg_probs[pred_class])

    elif aggregation == "majority":
        counts = np.bincount(window_preds, minlength=num_classes)
        pred_class = int(np.argmax(counts))
        confidence = float(counts[pred_class] / len(window_preds))
        agg_probs = counts / counts.sum()

    elif aggregation == "max_conf":
        max_conf_idx = np.max(all_probs, axis=1).argmax()
        pred_class = int(window_preds[max_conf_idx])
        agg_probs = all_probs[max_conf_idx]
        confidence = float(agg_probs[pred_class])

    else:
        raise ValueError(f"aggregation debe ser 'prob_sum', 'majority' o 'max_conf'")

    return {
        "pred_class": pred_class,
        "confidence": confidence,
        "class_probs": agg_probs,
        "window_preds": window_preds.tolist(),
        "n_windows": len(windows),
    }

def evaluate_video_level(model, test_videos, seq_len, n_features,
                          num_classes, class_names, aggregation="prob_sum"):
    """
    Evalúa a nivel de VIDEO COMPLETO usando agregación de ventanas.
    Esta es la métrica real de producción.
    """
    y_true, y_pred = [], []

    for seq, true_label, vid_id in test_videos:
        result = predict_video(model, seq, seq_len, num_classes, aggregation)
        if result is None:
            continue
        y_true.append(true_label)
        y_pred.append(result["pred_class"])

        status = "✓" if result["pred_class"] == true_label else "✗"
        print(f"    {status} {vid_id}: "
              f"true={class_names[true_label]} "
              f"pred={class_names[result['pred_class']]} "
              f"conf={result['confidence']:.2f} "
              f"({result['n_windows']} ventanas)")

    if not y_true:
        return 0.0, 0.0, 0.0

    acc = (np.array(y_true) == np.array(y_pred)).mean()
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ba  = balanced_accuracy_score(y_true, y_pred)
    print(f"    Video-level ({aggregation}): acc={acc:.3f} F1={f1:.3f} BA={ba:.3f}")
    return acc, f1, ba

# ── Entrenamiento con CV ────────────────────────────────────────────────────────
def train_one_nballs(n_balls, data_root, out_dir,
                     k_folds=5, epochs=120, batch=32,
                     seq_len=60, use_sliding_window=True,
                     cache_dir="runs/dets_cache_all"):

    n_features = n_balls * 4
    nb_out_dir = os.path.join(out_dir, f"{n_balls}b")
    os.makedirs(nb_out_dir, exist_ok=True)

    videos, label_map, class_names = load_videos_raw(data_root, n_balls, cache_dir)
    num_classes = len(label_map)

    if num_classes < 2:
        print(f"  Insuficientes clases para {n_balls}b")
        return None

    print(f"\n{'='*60}")
    print(f"  TRAINING MODEL FOR {n_balls} BALLS")
    print(f"{'='*60}")
    print(f"  Videos: {len(videos)}, Clases: {num_classes}")
    for cls_idx, cls_name in class_names.items():
        count = sum(1 for _, l, _ in videos if l == cls_idx)
        print(f"    {cls_name}: {count} videos")

    video_labels = np.array([v[1] for v in videos])
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_window_acc, fold_video_acc = [], []
    fold_window_f1,  fold_video_f1  = [], []
    fold_window_ba,  fold_video_ba  = [], []

    # Acumular predicciones de todos los folds para CM global
    all_y_true_video, all_y_pred_video = [], []
    all_y_true_window, all_y_pred_window = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(videos, video_labels)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")

        train_videos = [videos[i] for i in train_idx]
        test_videos  = [videos[i] for i in test_idx]

        # Split train → tr + val por video
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42+fold)
        train_labels = np.array([v[1] for v in train_videos])
        try:
            tr_idx, va_idx = next(sss.split(range(len(train_videos)), train_labels))
        except ValueError:
            tr_idx = list(range(len(train_videos)))
            va_idx = tr_idx[:max(1, len(tr_idx)//5)]

        tr_videos  = [train_videos[i] for i in tr_idx]
        val_videos = [train_videos[i] for i in va_idx]

        X_tr,  y_tr  = videos_to_windows(tr_videos,  seq_len, n_features)
        X_val, y_val = videos_to_windows(val_videos, seq_len, n_features)

        print(f"  Train: {len(tr_videos)} videos → {len(X_tr)} ventanas")
        print(f"  Val:   {len(val_videos)} videos → {len(X_val)} ventanas")
        print(f"  Test:  {len(test_videos)} videos")

        model = build_model(n_features, num_classes)
        early = EarlyStopping(monitor="val_loss", patience=10,
                              restore_best_weights=True, verbose=1)
        ckpt  = ModelCheckpoint(
            os.path.join(nb_out_dir, f"fold_{fold+1}_best.h5"),
            monitor="val_accuracy", save_best_only=True, verbose=0)

        model.fit(X_tr, y_tr,
                  validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=batch,
                  callbacks=[early, ckpt], verbose=1)

        # ── Evaluación por ventana ──────────────────────────────────────────
        X_test_w, y_test_w = videos_to_windows(test_videos, seq_len, n_features)
        pred_w = np.argmax(model.predict(X_test_w, verbose=0), axis=1)
        w_acc = (pred_w == y_test_w).mean()
        w_f1  = f1_score(y_test_w, pred_w, average="macro", zero_division=0)
        w_ba  = balanced_accuracy_score(y_test_w, pred_w)
        print(f"\n  [Window-level] acc={w_acc:.3f} F1={w_f1:.3f} BA={w_ba:.3f}")

        names_sorted = [class_names[i] for i in range(num_classes)]
        print(classification_report(y_test_w, pred_w,
                                    target_names=names_sorted, zero_division=0))

        all_y_true_window.extend(y_test_w.tolist())
        all_y_pred_window.extend(pred_w.tolist())

        fold_window_acc.append(w_acc)
        fold_window_f1.append(w_f1)
        fold_window_ba.append(w_ba)

        # ── Evaluación por video ────────────────────────────────────────────
        print(f"  [Video-level (prob_sum)]")
        y_true_v, y_pred_v = [], []
        for seq, true_label, vid_id in test_videos:
            result = predict_video(model, seq, seq_len, num_classes, "prob_sum")
            if result is None:
                continue
            y_true_v.append(true_label)
            y_pred_v.append(result["pred_class"])
            status = "✓" if result["pred_class"] == true_label else "✗"
            print(f"    {status} {vid_id}: "
                  f"true={class_names[true_label]} "
                  f"pred={class_names[result['pred_class']]} "
                  f"conf={result['confidence']:.2f} "
                  f"({result['n_windows']} ventanas)")

        if y_true_v:
            v_acc = (np.array(y_true_v) == np.array(y_pred_v)).mean()
            v_f1  = f1_score(y_true_v, y_pred_v, average="macro", zero_division=0)
            v_ba  = balanced_accuracy_score(y_true_v, y_pred_v)
            print(f"\n  [Video-level] acc={v_acc:.3f} F1={v_f1:.3f} BA={v_ba:.3f}")
            print(classification_report(y_true_v, y_pred_v,
                                        target_names=names_sorted, zero_division=0))
            all_y_true_video.extend(y_true_v)
            all_y_pred_video.extend(y_pred_v)
            fold_video_acc.append(v_acc)
            fold_video_f1.append(v_f1)
            fold_video_ba.append(v_ba)

    # ── Confusion Matrix global (todos los folds) ───────────────────────────
    names_sorted = [class_names[i] for i in range(num_classes)]

    for level, y_true_all, y_pred_all in [
        ("window", all_y_true_window, all_y_pred_window),
        ("video",  all_y_true_video,  all_y_pred_video),
    ]:
        if not y_true_all:
            continue
        cm = confusion_matrix(y_true_all, y_pred_all)
        fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes-1)))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set(xticks=range(num_classes), yticks=range(num_classes),
               xticklabels=names_sorted, yticklabels=names_sorted,
               xlabel="Predicho", ylabel="Real",
               title=f"{n_balls}b — Confusion Matrix ({level}-level, all folds)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        plt.tight_layout()
        plt.savefig(os.path.join(nb_out_dir, f"cm_{level}_level.png"), dpi=150)
        plt.close()
        print(f"  Confusion matrix ({level}) guardada.")

    # ── Resumen final ────────────────────────────────────────────────────────
    mean_w_acc = float(np.mean(fold_window_acc))
    mean_v_acc = float(np.mean(fold_video_acc)) if fold_video_acc else 0.0

    print(f"\n{'='*40}")
    print(f"  {n_balls}b RESULTS:")
    print(f"  Window-level: acc={mean_w_acc:.3f}±{np.std(fold_window_acc):.3f} "
          f"F1={np.mean(fold_window_f1):.3f}")
    if fold_video_acc:
        print(f"  Video-level:  acc={mean_v_acc:.3f}±{np.std(fold_video_acc):.3f} "
              f"F1={np.mean(fold_video_f1):.3f}  ← métrica real")

    results = {
        "n_balls": n_balls,
        "num_classes": num_classes,
        "n_videos": len(videos),
        "label_map": label_map,
        # window-level
        "fold_window_acc": fold_window_acc,
        "mean_window_acc": mean_w_acc,
        "std_window_acc": float(np.std(fold_window_acc)),
        "mean_window_f1": float(np.mean(fold_window_f1)),
        "mean_window_ba": float(np.mean(fold_window_ba)),
        # video-level
        "fold_video_acc": fold_video_acc,
        "mean_video_acc": mean_v_acc,
        "std_video_acc": float(np.std(fold_video_acc)) if fold_video_acc else 0.0,
        "mean_video_f1": float(np.mean(fold_video_f1)) if fold_video_f1 else 0.0,
        "mean_video_ba": float(np.mean(fold_video_ba)) if fold_video_ba else 0.0,
    }

    with open(os.path.join(nb_out_dir, "cv_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(nb_out_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    return results


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/content/drive/MyDrive/runs/track_csvs")
    ap.add_argument("--cache_dir", default="/content/drive/MyDrive/runs/dets_cache_all")  # ← añadido
    ap.add_argument("--out_dir",   default="models")
    ap.add_argument("--n_balls",   type=int, default=None)
    ap.add_argument("--k_folds",   type=int, default=5)
    ap.add_argument("--epochs",    type=int, default=120)
    ap.add_argument("--batch",     type=int, default=32)
    ap.add_argument("--seq_len",   type=int, default=60)
    ap.add_argument("--no_sliding_window", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    balls_to_train = [args.n_balls] if args.n_balls else [3, 4, 5, 6]
    all_results = {}

    for nb in balls_to_train:
        res = train_one_nballs(
            n_balls=nb,
            data_root=args.data_root,
            out_dir=args.out_dir,
            k_folds=args.k_folds,
            epochs=args.epochs,
            batch=args.batch,
            seq_len=args.seq_len,
            use_sliding_window=not args.no_sliding_window,
            cache_dir=args.cache_dir,  # ← pasado correctamente
        )
        if res:
            all_results[f"{nb}b"] = res

    print(f"\n{'='*60}")
    print("  RESUMEN GLOBAL")
    print(f"{'='*60}")
    for key, res in all_results.items():
        print(f"  {key}: "
              f"window acc={res['mean_window_acc']:.3f}±{res['std_window_acc']:.3f} "
              f"video acc={res['mean_video_acc']:.3f}±{res['std_video_acc']:.3f} "
              f"({res['num_classes']} clases, {res['n_videos']} videos)")

    with open(os.path.join(args.out_dir, "global_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()