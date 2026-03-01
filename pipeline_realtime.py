#!/usr/bin/env python3
# filepath: /home/dar0j/Documentos/2025/intro trabajo titulo el E/old rasmus/juggling-vision-py/pipeline_realtime.py
"""
pipeline_realtime.py
Pipeline de tiempo real para clasificación desde webcam.
Soporta:
  - Deep Learning: YOLO NANO → mini OC-SORT → TCN WINDOW model (por ventana de 60 frames)
  - Computer Vision: HSV color tracking → nohandlebars siteswap detection
  - Hybrid: YOLO NANO → OC-SORT → nohandlebars
"""
import numpy as np
import cv2
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict, deque

from tensorflow.keras.models import load_model


# ── Constantes (idénticas a train_per_nballs.py) ────────────────────────────
MASK_VALUE = -1.0
SEQ_LEN = 60  # 1 segundo a 60fps

# ── Feature engineering (idéntico a pipeline_dl.py / train_per_nballs.py) ────
def compute_velocity_features(pos_seq: np.ndarray) -> np.ndarray:
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


# ── Lightweight tracker for real-time ────────────────────────────────────────
class SimpleTracker:
    """
    Tracker liviano para tiempo real.
    Usa Hungarian assignment simple sin Kalman para mantener baja latencia.
    """
    def __init__(self, n_balls, max_cost=200):
        self.n_balls = n_balls
        self.max_cost = max_cost
        self.positions = {}  # track_id -> (x, y)
        self.next_id = 0
        self.lost_count = {}  # track_id -> frames sin detección
        self.max_lost = 15  # frames antes de eliminar track

    def update(self, detections):
        """
        detections: list de (cx, cy) centroides
        Returns: dict {track_id: (cx, cy)}
        """
        if not self.positions:
            # Inicializar tracks
            for det in detections[:self.n_balls]:
                self.positions[self.next_id] = det
                self.lost_count[self.next_id] = 0
                self.next_id += 1
            return dict(self.positions)

        if not detections:
            # Incrementar lost count
            for tid in list(self.positions.keys()):
                self.lost_count[tid] += 1
                if self.lost_count[tid] > self.max_lost:
                    del self.positions[tid]
                    del self.lost_count[tid]
            return dict(self.positions)

        # Hungarian assignment
        from scipy.optimize import linear_sum_assignment

        track_ids = list(self.positions.keys())
        track_pos = np.array([self.positions[tid] for tid in track_ids])
        det_pos = np.array(detections)

        n_tracks = len(track_pos)
        n_dets = len(det_pos)

        cost = np.linalg.norm(
            track_pos[:, None, :] - det_pos[None, :, :], axis=2
        )

        rows, cols = linear_sum_assignment(cost[:min(n_tracks, n_dets), :min(n_tracks, n_dets)])

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(rows, cols):
            if cost[r, c] < self.max_cost:
                tid = track_ids[r]
                self.positions[tid] = tuple(det_pos[c])
                self.lost_count[tid] = 0
                matched_tracks.add(r)
                matched_dets.add(c)

        # Tracks no matcheados
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.lost_count[tid] += 1
                if self.lost_count[tid] > self.max_lost:
                    del self.positions[tid]
                    del self.lost_count[tid]

        # Detections no matcheadas → nuevos tracks si hay espacio
        if len(self.positions) < self.n_balls:
            for j in range(n_dets):
                if j not in matched_dets and len(self.positions) < self.n_balls:
                    self.positions[self.next_id] = tuple(det_pos[j])
                    self.lost_count[self.next_id] = 0
                    self.next_id += 1

        return dict(self.positions)

    def get_ordered_positions(self):
        """Retorna posiciones ordenadas por track_id (primeros N)."""
        sorted_items = sorted(self.positions.items())[:self.n_balls]
        result = np.full(self.n_balls * 2, -1.0, dtype=np.float32)
        for i, (tid, (cx, cy)) in enumerate(sorted_items):
            result[i * 2] = cx
            result[i * 2 + 1] = cy
        return result


# ── Real-time DL pipeline ────────────────────────────────────────────────────
class RealtimeDLPipeline:
    """
    Pipeline de tiempo real: YOLO NANO → SimpleTracker → TCN WINDOW model.
    Clasifica cada ventana de 60 frames individualmente.
    """

    def __init__(self, yolo_model_path="MODELS/NANO.pt",
                 models_dir="MODELS/WINDOW",
                 confidence_threshold=0.3):
        self.models_dir = Path(models_dir)
        self.confidence_threshold = confidence_threshold

        # Cargar YOLO
        from ultralytics import YOLO
        self.yolo = YOLO(yolo_model_path)

        # Cache de modelos WINDOW
        self._model_cache = {}

        # Estado por sesión
        self.tracker = None
        self.frame_buffer = None  # deque de arrays (n_balls*2,)
        self.nballs = None
        self.predictions_history = deque(maxlen=10)  # últimas N predicciones

    def _load_model(self, nballs):
        if nballs in self._model_cache:
            return self._model_cache[nballs]

        nb_dir = self.models_dir / f"{nballs}b"
        if not nb_dir.exists():
            raise FileNotFoundError(f"No se encontró: {nb_dir}")

        model_files = sorted(nb_dir.glob("fold_*_best.h5"))
        if not model_files:
            raise FileNotFoundError(f"No se encontró modelo en {nb_dir}")

        lm_path = nb_dir / "label_map.json"
        if not lm_path.exists():
            raise FileNotFoundError(f"No se encontró {lm_path}")

        with open(lm_path) as f:
            label_map = json.load(f)

        class_names = {v: k for k, v in label_map.items()}
        model = load_model(str(model_files[0]))

        self._model_cache[nballs] = (model, label_map, class_names)
        return model, label_map, class_names

    def initialize(self, nballs):
        """Inicializar para una sesión de tiempo real."""
        self.nballs = nballs
        self.tracker = SimpleTracker(nballs, max_cost=200)
        self.frame_buffer = deque(maxlen=SEQ_LEN * 3)  # buffer largo para contexto
        self.predictions_history.clear()
        self._load_model(nballs)

    def process_frame(self, frame):
        """
        Procesa un frame de la webcam.
        
        Returns:
            dict con:
            - detections: list de (cx, cy) centroides detectados
            - tracks: dict {track_id: (cx, cy)}
            - prediction: dict o None si no hay suficientes frames
            - buffer_size: frames acumulados
            - annotated_frame: frame con anotaciones dibujadas
        """
        if self.nballs is None:
            raise ValueError("Llamar initialize() primero")

        h, w = frame.shape[:2]

        # 1. Detectar con YOLO
        results = self.yolo.predict(
            source=frame,
            conf=0.25,
            iou=0.7,
            verbose=False,
        )

        detections = []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            for box in xyxy:
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                detections.append((float(cx), float(cy)))

        # 2. Tracker update
        tracks = self.tracker.update(detections)
        positions = self.tracker.get_ordered_positions()
        self.frame_buffer.append(positions.copy())

        # 3. Clasificar si tenemos suficientes frames
        prediction = None
        buffer_arr = np.array(list(self.frame_buffer))
        n_buffered = len(self.frame_buffer)

        if n_buffered >= SEQ_LEN:
            # Tomar última ventana de 60 frames
            window = buffer_arr[-SEQ_LEN:]
            prediction = self._classify_window(window)
            if prediction is not None:
                self.predictions_history.append(prediction)

        # 4. Anotar frame
        annotated = self._annotate_frame(
            frame.copy(), detections, tracks, prediction, n_buffered
        )

        return {
            "detections": detections,
            "tracks": tracks,
            "prediction": prediction,
            "buffer_size": n_buffered,
            "annotated_frame": annotated,
        }

    def _classify_window(self, window):
        """Clasifica una ventana de 60 frames."""
        model, label_map, class_names = self._model_cache[self.nballs]
        num_classes = len(label_map)

        # Build features
        features = build_feature_sequence(window)  # (60, n_balls*4)

        # Predict
        inp = np.expand_dims(features, axis=0)  # (1, 60, n_balls*4)
        probs = model.predict(inp, verbose=0)[0]

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        if confidence < self.confidence_threshold:
            return None

        return {
            "trick": class_names.get(pred_class, f"class_{pred_class}"),
            "confidence": confidence,
            "class_probs": probs.tolist(),
        }

    def get_smoothed_prediction(self):
        """
        Retorna predicción suavizada usando las últimas N ventanas.
        Usa prob_sum como en el entrenamiento.
        """
        if not self.predictions_history:
            return None

        _, label_map, class_names = self._model_cache[self.nballs]
        num_classes = len(label_map)

        # Sumar probabilidades de últimas predicciones
        agg = np.zeros(num_classes)
        for pred in self.predictions_history:
            agg += np.array(pred["class_probs"])

        agg /= agg.sum()
        pred_class = int(np.argmax(agg))
        confidence = float(agg[pred_class])

        return {
            "trick": class_names.get(pred_class, f"class_{pred_class}"),
            "confidence": confidence,
            "n_windows": len(self.predictions_history),
        }

    def _annotate_frame(self, frame, detections, tracks, prediction, buffer_size):
        """Dibuja detecciones, tracks y predicción en el frame."""
        COLORS = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
        ]

        # Dibujar detecciones (círculos pequeños)
        for cx, cy in detections:
            cv2.circle(frame, (int(cx), int(cy)), 4, (200, 200, 200), -1)

        # Dibujar tracks (círculos grandes con ID)
        for i, (tid, (cx, cy)) in enumerate(sorted(tracks.items())):
            color = COLORS[i % len(COLORS)]
            cv2.circle(frame, (int(cx), int(cy)), 12, color, 3)
            cv2.putText(frame, f"B{i+1}", (int(cx)+15, int(cy)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Barra de progreso del buffer
        h, w = frame.shape[:2]
        progress = min(buffer_size / SEQ_LEN, 1.0)
        bar_w = int(w * 0.6)
        bar_x = (w - bar_w) // 2
        bar_y = h - 40
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20),
                      (60, 60, 60), -1)
        fill_w = int(bar_w * progress)
        color_bar = (0, 255, 0) if progress >= 1.0 else (0, 165, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 20),
                      color_bar, -1)
        cv2.putText(frame, f"{buffer_size}/{SEQ_LEN} frames",
                   (bar_x + bar_w + 10, bar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Predicción actual
        if prediction:
            text = f"{prediction['trick']} ({prediction['confidence']:.0%})"
            cv2.putText(frame, text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Predicción suavizada
        smoothed = self.get_smoothed_prediction()
        if smoothed:
            text2 = f"[avg] {smoothed['trick']} ({smoothed['confidence']:.0%})"
            cv2.putText(frame, text2, (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return frame


# ── Real-time CV pipeline ────────────────────────────────────────────────────
class RealtimeCVPipeline:
    """
    Pipeline CV de tiempo real: HSV tracking → siteswap detection.
    Requiere pelotas de color uniforme y altamente contrastante con el fondo.
    """

    def __init__(self):
        self.tracker = None
        self.frame_buffer = deque(maxlen=300)  # 5 segundos a 60fps
        self.nballs = None
        self.hsv_range = None
        self.min_area = 100
        self.bg_subtractor = None
        self.kernel = None
        self.siteswap_result = None
        self.frames_since_last_analysis = 0
        self.analysis_interval = 30  # analizar cada 30 frames (0.5s)

    def initialize(self, nballs, hsv_range, min_area=100):
        self.nballs = nballs
        self.hsv_range = hsv_range
        self.min_area = min_area
        self.tracker = SimpleTracker(nballs, max_cost=200)
        self.frame_buffer.clear()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.siteswap_result = None
        self.frames_since_last_analysis = 0

    def process_frame(self, frame):
        if self.nballs is None:
            raise ValueError("Llamar initialize() primero")

        h_min, s_min, v_min, h_max, s_max, v_max = self.hsv_range

        # 1. Background subtraction + HSV mask
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.005)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv,
                                  np.array([h_min, s_min, v_min]),
                                  np.array([h_max, s_max, v_max]))
        combined = cv2.bitwise_and(fg_mask, color_mask)
        combined = cv2.dilate(combined, self.kernel, iterations=1)

        # 2. Detecciones
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > 5000:
                continue
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                detections.append((cx, cy))

        # Ordenar por área descendente, tomar top N
        detections = detections[:self.nballs * 2]

        # 3. Track
        tracks = self.tracker.update(detections)
        positions = self.tracker.get_ordered_positions()
        self.frame_buffer.append(positions.copy())

        # 4. Análisis de siteswap periódico
        self.frames_since_last_analysis += 1
        buffer_size = len(self.frame_buffer)

        if (buffer_size >= 120 and
                self.frames_since_last_analysis >= self.analysis_interval):
            self._analyze_siteswap()
            self.frames_since_last_analysis = 0

        # 5. Anotar
        annotated = self._annotate_frame(
            frame.copy(), detections, tracks, combined, buffer_size
        )

        return {
            "detections": detections,
            "tracks": tracks,
            "siteswap": self.siteswap_result,
            "buffer_size": buffer_size,
            "annotated_frame": annotated,
            "mask": combined,
        }

    def _analyze_siteswap(self):
        """Analiza el buffer actual para detectar siteswap."""
        import tempfile
        import pandas as pd
        sys.path.insert(0, "./ComputerVision")
        from nohandlebars import pipeline as siteswap_pipeline

        buffer_arr = np.array(list(self.frame_buffer))

        # Guardar CSV temporal
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                              delete=False) as f:
                tmp_path = f.name
                for row in buffer_arr:
                    f.write(','.join(str(int(v)) for v in row) + '\n')

            result = siteswap_pipeline(
                tmp_path,
                n_balls=self.nballs,
                smooth_window=9,
                prominence=6,
                distance=8,
                frame_window=7,
                use_median=True,
                interpolate=True,
                visualize=False
            )

            self.siteswap_result = {
                "siteswap": result.get("siteswap", ""),
                "canonical": result.get("siteswap_canonical", ""),
                "period_length": result.get("period_length", 0),
                "num_peaks": result.get("num_peaks", 0),
            }
        except Exception as e:
            self.siteswap_result = {"siteswap": "", "error": str(e)}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _annotate_frame(self, frame, detections, tracks, mask, buffer_size):
        COLORS = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
        ]

        for cx, cy in detections:
            cv2.circle(frame, (cx, cy), 4, (200, 200, 200), -1)

        for i, (tid, (cx, cy)) in enumerate(sorted(tracks.items())):
            color = COLORS[i % len(COLORS)]
            cv2.circle(frame, (int(cx), int(cy)), 12, color, 3)

        h, w = frame.shape[:2]

        # Buffer progress
        min_frames = 120
        progress = min(buffer_size / min_frames, 1.0)
        bar_w = int(w * 0.4)
        bar_x = w - bar_w - 20
        bar_y = h - 40
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20),
                      (60, 60, 60), -1)
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 20),
                      (0, 255, 0) if progress >= 1.0 else (0, 165, 255), -1)

        # Siteswap
        if self.siteswap_result and self.siteswap_result.get("canonical"):
            ss = self.siteswap_result["canonical"]
            cv2.putText(frame, f"Siteswap: {ss}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        if buffer_size < min_frames:
            cv2.putText(frame, f"Acumulando... {buffer_size}/{min_frames}",
                       (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 165, 255), 2)

        return frame