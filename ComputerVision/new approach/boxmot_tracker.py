import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Optional, Dict
import yaml
from tqdm import tqdm

try:
    from boxmot import DeepOcSort, OcSort, BotSort, ByteTrack
    BOXMOT_AVAILABLE = True
except ImportError:
    BOXMOT_AVAILABLE = False
    print("⚠️ BoxMOT no disponible. Instalar con: pip install boxmot")


class BoxMOTJugglingTracker:
    """
    Tracker de malabarismo usando BoxMOT con preprocesamiento robusto.
    Soporta OCSORT, DeepOCSORT, BoTSORT, ByteTrack.
    """

    TRACKER_TYPES = {
        'ocsort': OcSort,
        'deepocsort': DeepOcSort,
        'botsort': BotSort,
        'bytetrack': ByteTrack
    }

    def __init__(self, 
                 tracker_type='deepocsort',
                 reid_weights='osnet_x0_25_msmt17.pt',  # ReID model para DeepOCSORT
                 device='cpu',
                 fp16=False,
                 # Parámetros de tracking
                 track_high_thresh=0.5,
                 track_low_thresh=0.1,
                 new_track_thresh=0.6,
                 track_buffer=30,
                 match_thresh=0.8,
                 min_hits=3,
                 # Preprocesamiento
                 bg_method='MOG2',
                 min_contour_area=100,
                 max_contour_area=5000,
                 enclosing_area_diff=0.5,
                 use_blur=True,
                 blur_kernel=5,
                 morph_kernel_size=3,
                 morph_operations=['open'],
                 history=100,
                 var_threshold=25,
                 detect_shadows=False,
                 asso_func='iou',
                 delta_t=3,
                 inertia=0.2):
        
        if not BOXMOT_AVAILABLE:
            raise ImportError("BoxMOT no está instalado")
        
        self.tracker_type = tracker_type.lower()
        if self.tracker_type not in self.TRACKER_TYPES:
            raise ValueError(f"tracker_type debe ser uno de: {list(self.TRACKER_TYPES.keys())}")
        
        # Inicializar tracker
        tracker_class = self.TRACKER_TYPES[self.tracker_type]
        
        if self.tracker_type in ['deepocsort', 'botsort']:
            # Trackers que usan ReID
            self.tracker = tracker_class(
                model_weights=Path(reid_weights),
                device=device,
                fp16=fp16,
                track_high_thresh=track_high_thresh,
                track_low_thresh=track_low_thresh,
                new_track_thresh=new_track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh
            )
        else:
            # OCSORT, ByteTrack (sin ReID)
            self.tracker = tracker_class(
                det_thresh=track_high_thresh,
                max_age=track_buffer,
                min_hits=min_hits,
                iou_threshold=match_thresh,
                asso_func=asso_func,   # ✅
                delta_t=delta_t,       # ✅
                inertia=inertia,       # ✅
            )
        
        # Parámetros de preprocesamiento
        self.bg_method = bg_method
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.blur_kernel = blur_kernel
        self.morph_kernel_size = morph_kernel_size
        self.morph_operations = morph_operations
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.enclosing_area_diff = enclosing_area_diff
        self.use_blur = use_blur
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size)
        )
        
        # ✅ En vez de duplicar el código aquí también
        self._init_bg_subtractor()
    
    def _init_bg_subtractor(self):
        """Inicializa (o reinicia) el background subtractor."""
        if self.bg_method == 'MOG2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows
            )
        elif self.bg_method == 'KNN':
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.var_threshold * 10,
                detectShadows=self.detect_shadows
            )

    def reset(self):
        """Reinicia tracker y BG subtractor para un nuevo video."""
        self._init_bg_subtractor()
        self.tracker.reset()
    
    def detect_balls(self, frame: np.ndarray) -> np.ndarray:
        """
        Detecta pelotas usando background subtraction + filtrado morfológico.
        
        Returns:
            detections: np.array de shape (N, 6) con formato [x1, y1, x2, y2, conf, class]
        """
        # 1. Preprocesamiento
        if self.use_blur:
            frame_processed = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)
        else:
            frame_processed = frame
        
        # 2. Background subtraction
        fg_mask = self.bg_subtractor.apply(frame_processed, learningRate=0.001)
        
        # 3. Threshold
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # 4. Operaciones morfológicas
        for op in self.morph_operations:
            if op == 'open':
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
            elif op == 'close':
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
            elif op == 'dilate':
                fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=1)
            elif op == 'erode':
                fg_mask = cv2.erode(fg_mask, self.kernel, iterations=1)
        
        # 5. Detección de contornos
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 6. Filtrado por forma y área
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
            
            # Verificar forma circular
            _, radius = cv2.minEnclosingCircle(c)
            enclosing_area = np.pi * radius * radius
            approx = cv2.approxPolyDP(c, 0.1 * cv2.arcLength(c, True), True)
            
            is_circular = abs(area - enclosing_area) < self.enclosing_area_diff * enclosing_area
            is_convex = len(approx) > 3 and cv2.isContourConvex(approx)
            
            if is_circular or is_convex:
                x, y, w, h = cv2.boundingRect(c)
                # Formato BoxMOT: [x1, y1, x2, y2, confidence, class_id]
                detections.append([x, y, x + w, y + h, 0.95, 0])
        
        if len(detections) == 0:
            return np.empty((0, 6))
        
        return np.array(detections, dtype=np.float32)
    
    def track_video(self, video_path, output_csv=None, visualize=False, save_video=None):
        """
        Trackea un video completo.
        
        Returns:
            DataFrame con columnas: frame, track_id, x, y, w, h, conf
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Preparar escritura de video
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(save_video, fourcc, fps, (width, height))
        
        all_tracks = []
        
        # ✅ Usar reset() en vez de recrear manualmente el bg_subtractor aquí
        self.reset()

        # Resetear tracker (importante para cada video)
        self.tracker.reset()
        
        pbar = tqdm(total=total_frames, desc=f"Tracking {Path(video_path).name}")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Detectar pelotas
            detections = self.detect_balls(frame)
            
            # 2. Tracking con BoxMOT
            raw = self.tracker.update(detections, frame)
            
            # Normalizar: (N,8) con tracks, o (0,) vacío → siempre 2D
            tracks = np.array(raw, dtype=np.float64)
            if tracks.ndim != 2 or tracks.shape[1] < 5:
                tracks = np.empty((0, 8))
            
            # 3. Guardar resultados
            for track in tracks:
                x1, y1, x2, y2 = track[0], track[1], track[2], track[3]
                track_id = int(track[4])
                conf     = track[5]
                
                all_tracks.append({
                    'frame': frame_idx,
                    'track_id': track_id,
                    'x': int(x1), 'y': int(y1),
                    'w': int(x2 - x1), 'h': int(y2 - y1),
                    'conf': conf
                })
            
            # 4. Visualización
            if visualize or save_video:
                vis_frame = frame.copy()
                
                # Dibujar detecciones (azul)
                for det in detections:
                    x1, y1, x2, y2 = det[:4].astype(int)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                
                # Dibujar tracks (verde)
                for track in tracks:
                    x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
                    tid = int(track[4])
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"ID{tid}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Info
                info = f"Frame: {frame_idx}/{total_frames} | Tracks: {len(tracks)} | Dets: {len(detections)}"
                cv2.putText(vis_frame, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if visualize:
                    cv2.imshow('Tracking', vis_frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break
                
                if save_video:
                    out_video.write(vis_frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if save_video:
            out_video.release()
        if visualize:
            cv2.destroyAllWindows()
        
        # Convertir a DataFrame
        df_tracks = pd.DataFrame(all_tracks)
        
        # Guardar CSV
        if output_csv:
            df_tracks.to_csv(output_csv, index=False)
            print(f"✓ CSV guardado: {output_csv}")
        
        return df_tracks


def track_multiple_videos(video_paths: List[str],
                         output_dir: str,
                         tracker_config: Dict,
                         visualize: bool = False,
                         save_videos: bool = False):
    """
    Trackea múltiples videos con la misma configuración.
    
    Args:
        video_paths: lista de rutas a videos
        output_dir: directorio para guardar CSVs
        tracker_config: diccionario con parámetros del tracker
        visualize: mostrar tracking en tiempo real
        save_videos: guardar videos con tracking visualizado
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tracker = BoxMOTJugglingTracker(**tracker_config)
    
    results = {}
    
    for video_path in video_paths:
        video_path = Path(video_path)
        print(f"\n{'='*60}")
        print(f"Procesando: {video_path.name}")
        print(f"{'='*60}")
        
        output_csv = output_dir / f"{video_path.stem}_tracks.csv"
        output_video = output_dir / f"{video_path.stem}_tracked.mp4" if save_videos else None
        
        df_tracks = tracker.track_video(
            str(video_path),
            output_csv=str(output_csv),
            visualize=visualize,
            save_video=str(output_video) if save_videos else None
        )
        
        results[video_path.name] = df_tracks
        
        # Estadísticas
        unique_ids = df_tracks['track_id'].nunique()
        total_frames = df_tracks['frame'].max() + 1
        avg_tracks_per_frame = len(df_tracks) / total_frames
        
        print(f"\n📊 Estadísticas:")
        print(f"  - Frames procesados: {total_frames}")
        print(f"  - IDs únicos: {unique_ids}")
        print(f"  - Detecciones totales: {len(df_tracks)}")
        print(f"  - Promedio tracks/frame: {avg_tracks_per_frame:.2f}")
    
    return results