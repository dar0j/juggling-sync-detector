#!/usr/bin/env python3
"""
pipeline_dl.py
Pipeline end-to-end para la app Flask:
  YOLO NANO (detección) → OC-SORT (tracking) → TCN per-nballs (clasificación)

No usa GridModel ni predict_trick.py.
"""
import numpy as np
import cv2
import json
import os
import sys
import urllib.request
from pathlib import Path
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras.models import load_model


# ── Constantes (deben coincidir con train_per_nballs.py) ────────────────────
MASK_VALUE = -1.0
TARGET_FPS = 60
SEQ_LEN = 60  # ventana de 1 segundo a 60fps


# ── OC-SORT bootstrap (mismo que pipeline_track_to_csv.py) ─────────────────
OCSORT_URLS = {
    "ocsort.py": "https://raw.githubusercontent.com/noahcao/OC_SORT/master/trackers/ocsort_tracker/ocsort.py",
    "kalmanfilter.py": "https://raw.githubusercontent.com/noahcao/OC_SORT/master/trackers/ocsort_tracker/kalmanfilter.py",
    "association.py": "https://raw.githubusercontent.com/noahcao/OC_SORT/master/trackers/ocsort_tracker/association.py",
}

def ensure_ocsort(vendor_dir: Path = Path("ocsort")):
    """Descarga OC-SORT si no existe."""
    vendor_dir.mkdir(parents=True, exist_ok=True)
    for name, url in OCSORT_URLS.items():
        dst = vendor_dir / name
        if not dst.exists():
            print(f"Descargando OC-SORT: {dst}")
            urllib.request.urlretrieve(url, dst)
    init = vendor_dir / "__init__.py"
    if not init.exists():
        init.write_text("from ocsort.ocsort import OCSort\n", encoding="utf-8")


# ── Feature engineering (idéntico a train_per_nballs.py) ────────────────────
def compute_velocity_features(pos_seq: np.ndarray) -> np.ndarray:
    """Calcula vx, vy por pelota usando np.gradient."""
    vel = np.full_like(pos_seq, MASK_VALUE)
    for col in range(pos_seq.shape[1]):
        col_data = pos_seq[:, col]
        valid_idx = np.where(col_data != MASK_VALUE)[0]
        if valid_idx.size < 2:
            continue
        grads = np.gradient(col_data[valid_idx])
        vel[valid_idx, col] = grads.astype(np.float32)
    return vel


def build_feature_sequence(data: np.ndarray) -> np.ndarray:
    """
    Dado array de posiciones (n_frames, n_balls*2):
      1. z-score por columna
      2. velocidades
      3. concatena → (n_frames, n_balls*4)
    """
    n_balls_2 = data.shape[1]
    pos = data.copy()
    for col in range(n_balls_2):
        valid = pos[:, col] != MASK_VALUE
        if valid.sum() < 2:
            continue
        v = pos[valid, col]
        pos[valid, col] = (v - v.mean()) / (v.std() + 1e-8)

    vel = compute_velocity_features(pos)

    n_frames = pos.shape[0]
    n_balls = n_balls_2 // 2
    out = np.full((n_frames, n_balls * 4), MASK_VALUE, dtype=np.float32)
    for b in range(n_balls):
        out[:, b * 4 + 0] = pos[:, b * 2 + 0]
        out[:, b * 4 + 1] = pos[:, b * 2 + 1]
        out[:, b * 4 + 2] = vel[:, b * 2 + 0]
        out[:, b * 4 + 3] = vel[:, b * 2 + 1]
    return out


def resample_to_target_fps(data: np.ndarray, current_fps: float,
                            target_fps: int = 60) -> np.ndarray:
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
        resampled[:, col] = np.interp(
            new_idx, old_idx[valid_mask], col_data[valid_mask],
            left=MASK_VALUE, right=MASK_VALUE
        )
    return resampled


# ── Track repair (mismo que pipeline_track_to_csv.py) ──────────────────────
def repair_fragmented_tracks(track_data: dict, n_balls: int,
                              max_gap: int = 45, max_merge_dist: float = 120.0):
    """Fusiona tracks fragmentados que pertenecen probablemente a la misma pelota."""
    if len(track_data) <= n_balls:
        return track_data

    track_info = {}
    for tid, points in track_data.items():
        frames = [p[0] for p in points]
        track_info[tid] = {
            "start": min(frames), "end": max(frames), "len": len(points),
            "start_pos": next(p for p in points if p[0] == min(frames)),
            "end_pos": next(p for p in points if p[0] == max(frames)),
        }

    sorted_ids = sorted(track_info.keys(), key=lambda t: track_info[t]["start"])
    merged = {}
    active_tracks = []

    for tid in sorted_ids:
        info = track_info[tid]
        best_merge = None
        best_score = float("inf")

        for atid in active_tracks:
            if atid in merged:
                continue
            a_info = track_info[atid]
            gap = info["start"] - a_info["end"]
            if gap < 0 or gap > max_gap:
                continue
            ex, ey = a_info["end_pos"][1], a_info["end_pos"][2]
            sx, sy = info["start_pos"][1], info["start_pos"][2]
            dist = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
            if dist < max_merge_dist:
                score = dist + gap * 2
                if score < best_score:
                    best_score = score
                    best_merge = atid

        if best_merge is not None:
            merged[tid] = best_merge
            track_info[best_merge]["end"] = info["end"]
            track_info[best_merge]["end_pos"] = info["end_pos"]
            track_info[best_merge]["len"] += info["len"]
        else:
            active_tracks.append(tid)

    result = defaultdict(list)
    for tid, points in track_data.items():
        target_id = merged.get(tid, tid)
        while target_id in merged:
            target_id = merged[target_id]
        result[target_id].extend(points)

    for tid in result:
        result[tid].sort(key=lambda p: p[0])

    return dict(result)


def select_top_tracks(track_data: dict, n_balls: int):
    """Selecciona los N tracks más largos."""
    if len(track_data) <= n_balls:
        return track_data
    ranked = sorted(track_data.items(), key=lambda kv: len(kv[1]), reverse=True)
    return dict(ranked[:n_balls])


def tracks_to_array(track_data: dict, n_balls: int, total_frames: int):
    """Convierte tracks a array (total_frames, n_balls*2) con -1 para gaps."""
    sorted_tids = sorted(
        track_data.keys(),
        key=lambda t: min(p[0] for p in track_data[t])
    )[:n_balls]

    arr = np.full((total_frames, n_balls * 2), -1.0, dtype=np.float32)
    for col_idx, tid in enumerate(sorted_tids):
        for frame, cx, cy in track_data[tid]:
            if 0 <= frame < total_frames:
                arr[frame, col_idx * 2] = cx
                arr[frame, col_idx * 2 + 1] = cy
    return arr


def interpolate_gaps(arr: np.ndarray, max_gap: int = 30):
    """Interpola linealmente gaps cortos en cada columna."""
    result = arr.copy()
    for col in range(result.shape[1]):
        series = result[:, col]
        valid = series != -1.0
        if valid.sum() < 2:
            continue
        valid_indices = np.where(valid)[0]
        for i in range(len(valid_indices) - 1):
            start = valid_indices[i]
            end = valid_indices[i + 1]
            gap_len = end - start - 1
            if 0 < gap_len <= max_gap:
                for j in range(1, gap_len + 1):
                    alpha = j / (gap_len + 1)
                    result[start + j, col] = (
                        series[start] * (1 - alpha) + series[end] * alpha
                    )
    return result


# ── Clasificación con prob_sum (idéntico a train_per_nballs.py) ─────────────
def classify_sequence(model, sequence: np.ndarray, seq_len: int,
                       num_classes: int) -> dict:
    """
    Clasifica secuencia completa usando ventanas deslizantes + prob_sum.
    
    Args:
        model: modelo TCN cargado
        sequence: (n_frames, n_features) features procesadas
        seq_len: largo de ventana (60)
        num_classes: número de clases del modelo
    
    Returns:
        dict con pred_class, confidence, class_probs, n_windows
    """
    n_frames = sequence.shape[0]
    n_features = sequence.shape[1]

    windows = []
    if n_frames < seq_len:
        padded = np.full((seq_len, n_features), MASK_VALUE, dtype=np.float32)
        padded[:n_frames] = sequence
        windows.append(padded)
    else:
        for start in range(n_frames - seq_len + 1):
            windows.append(sequence[start:start + seq_len])

    if not windows:
        return None

    windows_arr = np.array(windows, dtype=np.float32)
    all_probs = model.predict(windows_arr, verbose=0)

    # prob_sum aggregation
    agg_probs = all_probs.sum(axis=0)
    agg_probs /= agg_probs.sum()
    pred_class = int(np.argmax(agg_probs))
    confidence = float(agg_probs[pred_class])

    return {
        "pred_class": pred_class,
        "confidence": confidence,
        "class_probs": agg_probs.tolist(),
        "n_windows": len(windows),
    }


# ── Pipeline principal ──────────────────────────────────────────────────────
class DLPipeline:
    """
    Pipeline completo: YOLO NANO → OC-SORT → TCN per-nballs.
    Se instancia una vez al iniciar la app.
    """

    def __init__(self, yolo_model_path: str = "MODELS/NANO.pt",
                 models_dir: str = "MODELS/VIDEO",
                 ocsort_vendor_dir: str = "ocsort"):
        """
        Args:
            yolo_model_path: ruta al modelo YOLO nano para detección de pelotas
            models_dir: directorio con subcarpetas 3b/, 4b/, 5b/, 6b/ 
                        cada una con fold_*_best.h5 y label_map.json
            ocsort_vendor_dir: directorio donde se descarga OC-SORT
        """
        self.models_dir = Path(models_dir)
        self.yolo_model_path = yolo_model_path
        
        # Cargar YOLO
        print(f"Cargando YOLO NANO desde {yolo_model_path}...")
        from ultralytics import YOLO
        self.yolo = YOLO(yolo_model_path)
        print("✓ YOLO NANO cargado")

        # Asegurar OC-SORT
        ensure_ocsort(Path(ocsort_vendor_dir))
        parent_dir = str(Path(ocsort_vendor_dir).resolve().parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from ocsort import OCSort
        self.OCSort = OCSort
        print("✓ OC-SORT cargado")

        # Cache de modelos TCN y label_maps por nballs
        self._tcn_cache = {}  # {nballs: (model, label_map, class_names)}

        # OC-SORT config (mismos valores optimizados de pipeline_track_to_csv.py)
        self.ocsort_cfg = {
            "det_thresh": 0.57,
            "iou_threshold": -0.57,
            "asso_func": "giou",
            "max_age": 90,
            "min_hits": 3,
            "delta_t": 1,
            "inertia": 0.5,
            "use_byte": False,
        }

    def _load_tcn(self, nballs: int):
        """Carga modelo TCN y label_map para N pelotas (con cache)."""
        if nballs in self._tcn_cache:
            return self._tcn_cache[nballs]

        nb_dir = self.models_dir / f"{nballs}b"
        if not nb_dir.exists():
            raise FileNotFoundError(
                f"No se encontró directorio de modelo: {nb_dir}"
            )

        # Buscar mejor modelo (fold_*_best.h5)
        model_files = sorted(nb_dir.glob("fold_*_best.h5"))
        if not model_files:
            raise FileNotFoundError(
                f"No se encontró modelo .h5 en {nb_dir}"
            )
        # Usar el primero (o el mejor según cv_results.json si quieres)
        model_path = model_files[0]

        # Cargar label_map
        lm_path = nb_dir / "label_map.json"
        if not lm_path.exists():
            raise FileNotFoundError(f"No se encontró {lm_path}")

        with open(lm_path, "r") as f:
            lm = json.load(f)

        class_names = {v: k for k, v in lm.items()}
        num_classes = len(lm)

        print(f"Cargando TCN para {nballs}b desde {model_path} "
              f"({num_classes} clases)...")
        model = load_model(str(model_path))
        print(f"✓ TCN {nballs}b cargado")

        self._tcn_cache[nballs] = (model, lm, class_names)
        return model, lm, class_names

    def detect_and_track(self, video_path: str, nballs: int,
                          conf: float = 0.25, iou: float = 0.7):
        """
        Paso 1-2: YOLO detección + OC-SORT tracking.
        
        Returns:
            np.ndarray: (n_frames, nballs*2) coordenadas trackeadas
            float: fps real del video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"  Video: {total_frames} frames, {w}x{h}, {fps:.1f} fps")

        # Detectar con YOLO (stream mode)
        print("  Detectando pelotas con YOLO NANO...")
        results = self.yolo.predict(
            source=video_path,
            conf=conf,
            iou=iou,
            stream=True,
            verbose=False,
        )

        # Crear tracker OC-SORT
        tracker = self.OCSort(
            det_thresh=float(self.ocsort_cfg["det_thresh"]),
            max_age=int(self.ocsort_cfg["max_age"]),
            min_hits=int(self.ocsort_cfg["min_hits"]),
            iou_threshold=float(self.ocsort_cfg["iou_threshold"]),
            delta_t=int(self.ocsort_cfg["delta_t"]),
            asso_func=str(self.ocsort_cfg["asso_func"]),
            inertia=float(self.ocsort_cfg["inertia"]),
            use_byte=bool(self.ocsort_cfg["use_byte"]),
        )

        all_track_points = defaultdict(list)
        frame_i = 0

        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                # Formato [x1, y1, x2, y2, score]
                dets5 = np.column_stack([xyxy, confs]).astype(np.float32)
            else:
                dets5 = np.zeros((0, 5), dtype=np.float32)

            tracks = tracker.update(dets5, (h, w), (h, w))
            if tracks is not None and len(tracks) > 0:
                for row in np.asarray(tracks):
                    x1, y1, x2, y2 = row[:4]
                    tid = int(row[4])
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    all_track_points[tid].append((frame_i, cx, cy))

            frame_i += 1

        print(f"  Raw tracks: {len(all_track_points)}")

        # Reparar fragmentación
        repaired = repair_fragmented_tracks(
            all_track_points, nballs,
            max_gap=45, max_merge_dist=120.0
        )
        print(f"  Repaired tracks: {len(repaired)}")

        # Seleccionar top N
        selected = select_top_tracks(repaired, nballs)
        print(f"  Selected tracks: {len(selected)}")

        # Convertir a array
        arr = tracks_to_array(selected, nballs, frame_i)
        arr = interpolate_gaps(arr, max_gap=30)

        return arr, fps

    def classify(self, arr: np.ndarray, nballs: int,
                  video_fps: float, top_k: int = 5):
        """
        Paso 3: Preprocesar features + clasificar con TCN.
        
        Args:
            arr: (n_frames, nballs*2) coordenadas trackeadas
            nballs: número de pelotas
            video_fps: fps real del video (para resampling)
            top_k: número de predicciones top a retornar
        
        Returns:
            list de (trick_name, probability) ordenadas por probabilidad desc
        """
        model, label_map, class_names = self._load_tcn(nballs)
        num_classes = len(label_map)

        # Resamplear a 60fps
        arr_60 = resample_to_target_fps(arr, video_fps, TARGET_FPS)
        print(f"  Resampled: {arr.shape[0]} → {arr_60.shape[0]} frames (60fps)")

        # Construir features [x, y, vx, vy] por pelota
        sequence = build_feature_sequence(arr_60)
        print(f"  Features: {sequence.shape}")

        # Clasificar con prob_sum
        result = classify_sequence(model, sequence, SEQ_LEN, num_classes)

        if result is None:
            return [("unknown", 0.0)]

        # Construir top-k predicciones
        probs = np.array(result["class_probs"])
        top_indices = np.argsort(probs)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            trick_name = class_names.get(idx, f"class_{idx}")
            predictions.append((trick_name, float(probs[idx])))

        print(f"  Predicción: {predictions[0][0]} "
              f"({predictions[0][1]:.1%}) "
              f"[{result['n_windows']} ventanas]")

        return predictions

    def process_video(self, video_path: str, nballs: int,
                       top_k: int = 5, conf: float = 0.25, iou: float = 0.7):
        """
        Pipeline completo end-to-end.
        
        Args:
            video_path: ruta al video
            nballs: número de pelotas
            top_k: número de predicciones top
            conf: umbral de confianza YOLO
            iou: umbral IOU YOLO
        
        Returns:
            dict con predictions, n_frames, n_balls, etc.
        """
        print(f"\n{'='*50}")
        print(f"DL Pipeline: {os.path.basename(video_path)} ({nballs} pelotas)")
        print(f"{'='*50}")

        # Paso 1-2: Detectar y trackear
        arr, video_fps = self.detect_and_track(video_path, nballs, conf, iou)

        # Paso 3: Clasificar
        predictions = self.classify(arr, nballs, video_fps, top_k)

        return {
            "predictions": predictions,
            "n_frames": int(arr.shape[0]),
            "n_balls": nballs,
            "video_fps": video_fps,
        }