import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import re
from collections import deque

try:
    import tracking.data_saver_files.mot16_utils as mu
except:
    import data_saver_files.mot16_utils as mu


class KalmanBoxTracker:
    """
    Kalman Filter adaptado para OCSORT con velocidad observada.
    Estado: [x, y, vx, vy] donde (x,y) es el centro de la pelota
    """
    count = 0
    
    def __init__(self, detection):
        """
        Args:
            detection: tuple (x, y) centro de la pelota
        """
        self.kf = cv2.KalmanFilter(4, 2)  # 4 estados, 2 mediciones
        
        # Matriz de transición (modelo de velocidad constante)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x_new = x + vx
            [0, 1, 0, 1],  # y_new = y + vy
            [0, 0, 1, 0],  # vx_new = vx
            [0, 0, 0, 1]   # vy_new = vy
        ], dtype=np.float32)
        
        # Matriz de medición (solo observamos x, y)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Ruido del proceso (ajustar según dinámica de pelotas)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        
        # Ruido de medición (ajustar según calidad de detecciones)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        
        # Estado inicial: [x, y, 0, 0]
        self.kf.statePost = np.array([detection[0], detection[1], 0, 0], dtype=np.float32)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = deque(maxlen=30)  # Para velocity smoothing
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        
    def update(self, detection):
        """
        Actualiza el estado con una nueva detección.
        """
        self.time_since_update = 0
        self.history.clear()
        self.hits += 1
        self.hit_streak += 1
        
        # Medir velocidad observada
        if len(self.history) > 0:
            prev_x, prev_y = self.history[-1]
            vx_obs = detection[0] - prev_x
            vy_obs = detection[1] - prev_y
        else:
            vx_obs, vy_obs = 0, 0
            
        # Corrección con velocidad observada (OCSORT trick)
        measurement = np.array([[np.float32(detection[0])], 
                               [np.float32(detection[1])]])
        self.kf.correct(measurement)
        
        # Smooth velocity usando historia
        if len(self.history) >= 3:
            velocities = np.diff(np.array(list(self.history)), axis=0)
            vx_smooth = np.median(velocities[:, 0])
            vy_smooth = np.median(velocities[:, 1])
            self.kf.statePost[2] = vx_smooth
            self.kf.statePost[3] = vy_smooth
            
        self.history.append(detection)
        
    def predict(self):
        """
        Predice la siguiente posición.
        """
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        prediction = self.kf.predict()
        self.history.append((prediction[0], prediction[1]))
        
        return (int(prediction[0]), int(prediction[1]))
    
    def get_state(self):
        """
        Retorna el estado actual (x, y).
        """
        return (int(self.kf.statePost[0]), int(self.kf.statePost[1]))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3, max_distance=150):
    """
    Asigna detecciones a trackers usando Hungarian Algorithm.
    Combina distancia euclidiana + edad del tracker para priorizar tracks estables.
    
    Args:
        detections: lista de (x, y) centros detectados
        trackers: lista de estados predichos [(x, y), ...]
        max_distance: distancia máxima en píxeles para asociar
        
    Returns:
        matches: lista de pares (detection_idx, tracker_idx)
        unmatched_detections: índices de detecciones sin tracker
        unmatched_trackers: índices de trackers sin detección
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty(0, dtype=int)
    
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=int), np.arange(len(trackers))
    
    # Matriz de costos: distancia euclidiana
    cost_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            distance = np.linalg.norm(np.array(det) - np.array(trk))
            cost_matrix[d, t] = distance
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filtrar matches por distancia máxima
    matches = []
    unmatched_detections = []
    unmatched_trackers = list(range(len(trackers)))
    
    for d, t in zip(row_ind, col_ind):
        if cost_matrix[d, t] < max_distance:
            matches.append([d, t])
            if t in unmatched_trackers:
                unmatched_trackers.remove(t)
        else:
            unmatched_detections.append(d)
    
    # Detecciones sin match
    for d in range(len(detections)):
        if d not in row_ind:
            unmatched_detections.append(d)
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class OCSort:
    """
    Observation-Centric SORT para tracking de pelotas.
    Mejora sobre SORT usando velocidad observada y asociación robusta.
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, max_distance=150):
        """
        Args:
            max_age: frames máximos sin detección antes de eliminar tracker
            min_hits: detecciones mínimas para considerar track válido
            max_distance: distancia máxima en píxeles para asociar (ajustar según resolución)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.trackers = []
        self.frame_count = 0
        
    def update(self, detections):
        """
        Args:
            detections: np.array de shape (N, 2) con centros (x, y)
            
        Returns:
            np.array de shape (M, 3) con [x, y, id] de tracks activos
        """
        self.frame_count += 1
        
        # Predicción de trackers existentes
        trks = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks.append(pos)
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Eliminar trackers con predicciones inválidas
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = [trk for i, trk in enumerate(trks) if i not in to_del]
        
        # Asociación usando Hungarian
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, trks, self.iou_threshold, self.max_distance
        )
        
        # Actualizar trackers con matches
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]])
        
        # Crear nuevos trackers para detecciones sin match
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i])
            self.trackers.append(trk)
        
        # Eliminar trackers muertos
        ret = []
        for trk in reversed(self.trackers):
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)
                continue
            
            # Solo retornar tracks con suficientes hits
            if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                d = trk.get_state()
                ret.append([d[0], d[1], trk.id])
        
        if len(ret) > 0:
            return np.array(ret)
        return np.empty((0, 3))


def contour_center(c):
    """Calcula el centro de un contorno."""
    M = cv2.moments(c)
    try:
        center = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    except:
        center = None
    return center


def ocsort_bg_tracker(source_path, 
                      bg_method='MOG2',  # 'MOG2' o 'KNN'
                      min_contour_area=100, 
                      max_contour_area=5000,
                      enclosing_area_diff=0.5, 
                      arc_const=0.1,
                      use_blur=False,
                      blur_kernel=5,
                      morph_kernel_size=3,
                      morph_operations=['open', 'close'],
                      # OCSORT params
                      max_age=30,
                      min_hits=3,
                      max_distance=150,
                      # Background subtractor params
                      history=100,
                      var_threshold=25,
                      detect_shadows=False,
                      # Visualización y guardado
                      save_data=False,
                      visualize=False):
    """
    Tracker robusto usando Background Subtraction + OCSORT.
    
    Args:
        source_path: ruta al video
        bg_method: 'MOG2' o 'KNN' para background subtraction
        min_contour_area: área mínima de contorno en píxeles²
        max_contour_area: área máxima de contorno en píxeles²
        enclosing_area_diff: diferencia permitida entre área real y círculo mínimo
        arc_const: constante para aproximación poligonal
        use_blur: aplicar Gaussian blur antes de procesamiento
        blur_kernel: tamaño del kernel de blur (impar)
        morph_kernel_size: tamaño del kernel para operaciones morfológicas
        morph_operations: lista de operaciones ['open', 'close', 'dilate', 'erode']
        max_age: frames sin detección antes de eliminar tracker
        min_hits: detecciones mínimas para considerar track válido
        max_distance: distancia máxima para asociar detección a tracker
        history: frames para modelo de fondo
        var_threshold: umbral de varianza (menor = más sensible)
        detect_shadows: detectar sombras (ralentiza pero mejora en algunos casos)
        save_data: guardar resultados en formato MOT16
        visualize: mostrar tracking en tiempo real
    """
    # Extraer siteswap del nombre del archivo
    try:
        ss = re.search(r"ss(\d+)", source_path).group(1)
    except:
        ss = "Unknown"
    
    system = f"OCSort_{bg_method}"
    
    cap = cv2.VideoCapture(source_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Crear background subtractor
    if bg_method == 'MOG2':
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
    elif bg_method == 'KNN':
        bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=history,
            dist2Threshold=var_threshold * 10,  # KNN usa threshold diferente
            detectShadows=detect_shadows
        )
    else:
        raise ValueError(f"bg_method debe ser 'MOG2' o 'KNN', recibido: {bg_method}")
    
    # Crear kernel morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                      (morph_kernel_size, morph_kernel_size))
    
    # Inicializar OCSORT
    tracker = OCSort(max_age=max_age, min_hits=min_hits, max_distance=max_distance)
    
    # Inicializar guardado de datos
    if save_data:
        file = mu.file_initializer(system, ss, 'Tracking')
    
    if visualize:
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    
    ret, frame = cap.read()
    current_frame = 0
    
    print(f"Procesando {total_frames} frames con {system}...")
    
    while ret:
        # Preprocessing opcional
        if use_blur:
            frame_processed = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        else:
            frame_processed = frame
        
        # 1. BACKGROUND SUBTRACTION
        fg_mask = bg_subtractor.apply(frame_processed, learningRate=0.001)
        
        # 2. THRESHOLD (limpiar ruido)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # 3. MORPHOLOGY OPERATIONS
        for op in morph_operations:
            if op == 'open':
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            elif op == 'close':
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            elif op == 'dilate':
                fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
            elif op == 'erode':
                fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        
        # 4. CONTOUR DETECTION
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. FILTRADO DE CONTORNOS (forma circular + área)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            
            # Filtro por área
            if area < min_contour_area or area > max_contour_area:
                continue
            
            # Filtro por forma circular
            _, radius = cv2.minEnclosingCircle(c)
            enclosing_area = np.pi * radius * radius
            approx = cv2.approxPolyDP(c, arc_const * cv2.arcLength(c, True), True)
            
            # Verificar si es circular (por área) o convexo
            is_circular = abs(area - enclosing_area) < enclosing_area_diff * enclosing_area
            is_convex = len(approx) > 3 and cv2.isContourConvex(approx)
            
            if is_circular or is_convex:
                center = contour_center(c)
                if center is not None:
                    detections.append(center)
        
        # 6. OCSORT TRACKING
        if len(detections) > 0:
            detections_array = np.array(detections, dtype=np.float32)
            tracks = tracker.update(detections_array)
        else:
            tracks = tracker.update(np.empty((0, 2)))
        
        # 7. GUARDAR DATOS (formato MOT16)
        if save_data:
            for track in tracks:
                x, y, track_id = int(track[0]), int(track[1]), int(track[2])
                mu.file_writer(file, current_frame + 1, track_id + 1, (x, y))
        
        # 8. VISUALIZACIÓN
        if visualize:
            vis_frame = frame.copy()
            
            # Dibujar detecciones en rojo
            for det in detections:
                cv2.circle(vis_frame, det, 5, (0, 0, 255), -1)
            
            # Dibujar tracks en verde
            for track in tracks:
                x, y, track_id = int(track[0]), int(track[1]), int(track[2])
                cv2.circle(vis_frame, (x, y), 8, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"ID{track_id}", (x + 12, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Información del frame
            info_text = f"Frame: {current_frame}/{total_frames} | Tracks: {len(tracks)} | Dets: {len(detections)}"
            cv2.putText(vis_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar ventanas
            cv2.imshow('Tracking', vis_frame)
            cv2.imshow('Mask', fg_mask)
            
            k = cv2.waitKey(1)
            if k == 27:  # ESC para salir
                break
        
        # Progreso cada 100 frames
        if current_frame % 100 == 0:
            print(f"  Frame {current_frame}/{total_frames} | Active tracks: {len(tracker.trackers)}")
        
        ret, frame = cap.read()
        current_frame += 1
    
    # Cleanup
    if visualize:
        cap.release()
        cv2.destroyAllWindows()
    cap.release()
    
    if save_data:
        print(f'Tracking finalizado. Datos guardados.')
        mu.file_saver(file)
    
    print(f"\n✓ Procesamiento completo: {current_frame} frames")
    print(f"✓ Total de tracks creados: {KalmanBoxTracker.count}")
    
    return tracker


if __name__ == "__main__":
    # Ejemplo de uso
    source_path = '/home/dar0j/Documentos/2025/intro trabajo titulo el E/.mp4'
    
    tracker = ocsort_bg_tracker(
        source_path=source_path,
        bg_method='MOG2',  # Probar también 'KNN'
        min_contour_area=100,
        max_contour_area=5000,
        enclosing_area_diff=0.5,
        arc_const=0.1,
        use_blur=True,
        blur_kernel=5,
        morph_kernel_size=3,
        morph_operations=['open', 'close'],
        # OCSORT tuning
        max_age=30,        # ↑ = tracks sobreviven más sin detección
        min_hits=3,        # ↑ = más estricto para considerar track válido
        max_distance=150,  # ↑ = permite asociaciones más lejanas
        # Background subtractor tuning
        history=100,       # ↑ = modelo de fondo más estable
        var_threshold=25,  # ↓ = más sensible a cambios
        detect_shadows=False,
        save_data=True,
        visualize=True
    )