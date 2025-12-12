from glob import glob
import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def auto_extract_hsv_range(video_path, num_samples=50, h_margin=15, sv_margin=80):
    """
    Extrae automáticamente el rango HSV del color más frecuente.
    
    Args:
        video_path: ruta al video
        num_samples: número de frames a analizar
        h_margin: margen para el canal H (hue)
        sv_margin: margen para canales S y V
    
    Returns:
        tuple: (h_min, s_min, v_min, h_max, s_max, v_max)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Background subtractor para detectar objetos en movimiento
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
    
    # Acumular colores HSV de píxeles detectados
    hsv_pixels = []
    
    # Muestrear frames distribuidos uniformemente
    sample_indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
    
    print(f"Analizando {num_samples} frames para extraer color...")
    
    # CAMBIO CRÍTICO: usar detección directa sin bg_subtractor inicialmente
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Opción A: sin background subtraction (más robusto para auto-extracción)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detectar colores saturados (pelotas típicamente tienen S alto)
        high_sat_mask = (hsv[:,:,1] > 50) & (hsv[:,:,2] > 50)
        moving_pixels = hsv[high_sat_mask]
        
        if len(moving_pixels) > 0:
            hsv_pixels.append(moving_pixels)
    
    cap.release()
    
    if not hsv_pixels:
        print("No se detectaron objetos en movimiento. Usando rango por defecto.")
        return 117, 116, 28, 193, 255, 174#(0, 140, 87, 9, 196, 203) #(0, 50, 50, 180, 255, 255)
    
    # Concatenar todos los píxeles
    all_hsv = np.vstack(hsv_pixels)
    
    # Encontrar el color más frecuente usando histograma 3D simplificado
    # Discretizar en bins para crear "clusters" de colores
    h_bins = (all_hsv[:, 0] // 10).astype(np.int32)  # Dividir H en bins de 10
    s_bins = (all_hsv[:, 1] // 25).astype(np.int32)  # Dividir S en bins de 25
    v_bins = (all_hsv[:, 2] // 25).astype(np.int32)  # Dividir V en bins de 25
    
    # Crear identificador único por bin
    bin_ids = h_bins * 1000 + s_bins * 100 + v_bins
    
    # Contar frecuencias
    unique, counts = np.unique(bin_ids, return_counts=True)
    most_common_bin = unique[np.argmax(counts)]
    
    # Decodificar el bin más frecuente
    h_bin = (most_common_bin // 1000) * 10
    s_bin = ((most_common_bin % 1000) // 100) * 25
    v_bin = (most_common_bin % 100) * 25
    
    # Crear rango con márgenes
    h_min = max(0, h_bin - h_margin)
    h_max = min(180, h_bin + h_margin + 10)
    s_min = max(0, s_bin - sv_margin)
    s_max = min(255, s_bin + sv_margin + 25)
    v_min = max(0, v_bin - sv_margin)
    v_max = min(255, v_bin + sv_margin + 25)
    
    print(f"Color detectado: H={h_bin}, S={s_bin}, V={v_bin}")
    print(f"Rango HSV: ({h_min}, {s_min}, {v_min}, {h_max}, {s_max}, {v_max})")
    
    return (h_min, s_min, v_min, h_max, s_max, v_max)


# Versión mejorada que usa auto-extracción + background subtraction
def track_balls_auto(video_path, num_balls, output_csv, 
                     auto_hsv=True, hsv_range=None, 
                     min_area=100, max_cost=150, visualize=False):
    """
    Tracking automático: extrae color HSV y luego trackea con HSV + bg subtraction.
    
    Args:
        auto_hsv: si True, extrae HSV automáticamente; si False, usa hsv_range
        hsv_range: rango manual (si auto_hsv=False)
    """
    # 1. Extraer rango HSV automáticamente
    if auto_hsv:
        hsv_range = auto_extract_hsv_range(video_path)
    
    h_min, s_min, v_min, h_max, s_max, v_max = hsv_range
    
    # 2. Tracking con HSV
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
    trajectories = np.full((total_frames, num_balls, 2), np.nan, dtype=np.float32)
    prev_positions = None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    frame_idx = 0
    print(f"Trackeando {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Combinar background subtraction + HSV para robustez
        fg_mask = bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, 
                                 np.array([h_min, s_min, v_min]), 
                                 np.array([h_max, s_max, v_max]))
        
        # Combinar ambas máscaras (AND lógico)
        combined_mask = cv2.bitwise_and(fg_mask, color_mask)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append((cx, cy, area))
        
        centers = sorted(centers, key=lambda x: x[2], reverse=True)[:num_balls]
        centers = [(x, y) for x, y, _ in centers]
        
        # Hungarian assignment
        if prev_positions is None:
            current_positions = [None] * num_balls
            for i, center in enumerate(centers):
                current_positions[i] = center
        else:
            current_positions = hungarian_assignment(prev_positions, centers, max_cost)
        
        for ball_id in range(num_balls):
            pos = current_positions[ball_id]
            if pos is not None:
                trajectories[frame_idx, ball_id, 0] = pos[0]
                trajectories[frame_idx, ball_id, 1] = pos[1]
        
        prev_positions = current_positions
        
        if visualize:
            vis = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            for i, pos in enumerate(current_positions):
                if pos:
                    pos_s = (pos[0]//2, pos[1]//2)
                    cv2.circle(vis, pos_s, 5, (0, 255, 0), -1)
                    cv2.putText(vis, str(i+1), (pos_s[0]+8, pos_s[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            width = 360
            height = 640
            cv2.imshow('Tracking', cv2.resize(vis, (width, height)))
            cv2.imshow('Mask', cv2.resize(combined_mask, (width, height)))
            
            # Posicionar ventanas lado a lado arriba
            cv2.moveWindow('Tracking', 2000, 50)      # izquierda
            cv2.moveWindow('Mask', 2000 + width + 20, 50)  # derecha
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total_frames}...")
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    data = trajectories[:frame_idx].reshape(frame_idx, num_balls * 2)
    # Convertir a int, reemplazando NaN con vacío o 0
    data_int = np.where(np.isnan(data), -1, data).astype(int)  # -1 marca valores faltantes
    np.savetxt(output_csv, data_int, delimiter=',', fmt='%d')
    print(f"CSV guardado: {output_csv}")


def hungarian_assignment(prev_positions, new_centers, max_cost):
    """Hungarian assignment"""
    n_balls = len(prev_positions)
    n_detections = len(new_centers)
    
    if n_detections == 0:
        return [None] * n_balls
    
    cost_matrix = np.full((n_balls, n_detections), max_cost * 2, dtype=np.float32)
    
    for i, prev_pos in enumerate(prev_positions):
        if prev_pos is None:
            continue
        prev_arr = np.array(prev_pos, dtype=np.float32)
        centers_arr = np.array(new_centers, dtype=np.float32)
        dists = np.linalg.norm(centers_arr - prev_arr, axis=1)
        cost_matrix[i, :] = dists
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    result = [None] * n_balls
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < max_cost:
            result[i] = new_centers[j]
    
    return result


class KalmanBallTracker:
    """Filtro de Kalman para tracking de pelota individual"""
    def __init__(self, initial_pos):
        self.kf = cv2.KalmanFilter(4, 2)  # 4 estados (x,y,vx,vy), 2 mediciones (x,y)
        
        # Matriz de transición (movimiento uniforme)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Matriz de medición
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Covarianzas (ajustar según ruido del video)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.5#* 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        
        # Estado inicial
        self.kf.statePost = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
        self.frames_lost = 0
        self.max_lost = 30  # máximo de frames sin detección antes de eliminar
    
    def predict(self):
        """Predice siguiente posición"""
        prediction = self.kf.predict()
        return (int(prediction[0]), int(prediction[1]))
    
    def update(self, measurement):
        """Actualiza con nueva medición"""
        if measurement is None:
            self.frames_lost += 1
            return self.predict()
        else:
            self.frames_lost = 0
            meas = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
            self.kf.correct(meas)
            return measurement
    
    def is_lost(self):
        return self.frames_lost > self.max_lost


def track_balls_with_kalman(video_path, num_balls, output_csv, 
                            auto_hsv=True, hsv_range=None, 
                            min_area=100, max_cost=200, visualize=False):
    """Tracking con Kalman Filter para robustez"""
    
    if auto_hsv:
        hsv_range = auto_extract_hsv_range(video_path, num_samples=50, h_margin=15, sv_margin=100)
    
    h_min, s_min, v_min, h_max, s_max, v_max = hsv_range
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Background subtractor con parámetros relajados
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=200,          # más historia para escenas dinámicas
        varThreshold=25,      # umbral más bajo = más sensible
        detectShadows=False
    )
    
    trajectories = np.full((total_frames, num_balls, 2), np.nan, dtype=np.float32)
    kalman_trackers = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    frame_idx = 0
    print(f"Trackeando con Kalman Filter...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # === DETECCIÓN ===
        fg_mask = bg_subtractor.apply(frame, learningRate=0.001)  # learning rate bajo
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, 
                                 np.array([h_min, s_min, v_min]), 
                                 np.array([h_max, s_max, v_max]))
        
        # Máscara combinada
        combined_mask = cv2.bitwise_and(fg_mask, color_mask)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > 5000:  # filtrar áreas irreales
                continue
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                detections.append((cx, cy, area))
        
        detections = sorted(detections, key=lambda x: x[2], reverse=True)
        detections = [(x, y) for x, y, _ in detections]
        
        # === INICIALIZACIÓN DE TRACKERS ===
        if frame_idx == 0 and detections:
            for det in detections[:num_balls]:
                kalman_trackers.append(KalmanBallTracker(det))
        
        # === PREDICCIÓN + ASIGNACIÓN ===
        if kalman_trackers:
            predictions = [kf.predict() for kf in kalman_trackers]
            
            # Hungarian assignment: predictions <-> detections
            assignments = hungarian_assignment_kalman(predictions, detections, max_cost)
            
            # Actualizar cada tracker
            for i, (kf, assigned_det) in enumerate(zip(kalman_trackers, assignments)):
                updated_pos = kf.update(assigned_det)
                if updated_pos:
                    trajectories[frame_idx, i, 0] = updated_pos[0]
                    trajectories[frame_idx, i, 1] = updated_pos[1]
        
        # Eliminar trackers perdidos
        kalman_trackers = [kf for kf in kalman_trackers if not kf.is_lost()]
        
        # === VISUALIZACIÓN ===
        if visualize:
            vis = frame.copy()
            for i, kf in enumerate(kalman_trackers):
                pred = kf.predict()
                # Prediction en azul
                cv2.circle(vis, pred, 15, (255, 0, 0), 2)
                # Estado actual en verde
                state_pos = (int(kf.kf.statePost[0]), int(kf.kf.statePost[1]))
                cv2.circle(vis, state_pos, 8, (0, 255, 0), -1)
                cv2.putText(vis, f"B{i+1}", (state_pos[0]+12, state_pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Detecciones en rojo
            for det in detections:
                cv2.circle(vis, det, 5, (0, 0, 255), -1)
            
            vis = cv2.resize(vis, (0, 0), fx=0.5, fy=0.5)
            mask_vis = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            mask_vis = cv2.resize(mask_vis, (0, 0), fx=0.5, fy=0.5)
            
            width = vis.shape[1]
            height = vis.shape[0]
            
            cv2.imshow('Kalman Tracking', vis)
            cv2.imshow('Mask', mask_vis)
            cv2.moveWindow('Kalman Tracking', 2000, 50)
            cv2.moveWindow('Mask', 2000 + width + 20, 50)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total_frames} | Trackers: {len(kalman_trackers)}")
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Guardar CSV como enteros
    data = trajectories[:frame_idx].reshape(frame_idx, num_balls * 2)
    data_int = np.where(np.isnan(data), -1, data).astype(int)
    np.savetxt(output_csv, data_int, delimiter=',', fmt='%d')
    print(f"✓ CSV guardado: {output_csv}")
    
    # Estadísticas (usar datos originales para cálculo preciso)
    valid_frames = np.sum(~np.isnan(data), axis=0)
    for i in range(num_balls):
        pct = (valid_frames[i*2] / frame_idx) * 100
        print(f"  Pelota {i+1}: {pct:.1f}% frames con detección")


def hungarian_assignment_kalman(predictions, detections, max_cost):
    """Assignment optimizado para Kalman (predictions vs detections)"""
    n_pred = len(predictions)
    n_det = len(detections)
    
    if n_det == 0:
        return [None] * n_pred
    
    cost_matrix = np.full((n_pred, n_det), max_cost * 2, dtype=np.float32)
    
    for i, pred in enumerate(predictions):
        pred_arr = np.array(pred, dtype=np.float32)
        det_arr = np.array(detections, dtype=np.float32)
        dists = np.linalg.norm(det_arr - pred_arr, axis=1)
        cost_matrix[i, :] = dists
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    result = [None] * n_pred
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < max_cost:
            result[i] = detections[j]
    
    return result


if __name__ == "__main__":
    for nball in [3, 4, 5 ,6]:
        input_dir =  f"/home/dar0j/Documentos/2025/intro trabajo titulo el E/PROJECT/Datasets/{nball}b"
        video_files = glob(os.path.join(input_dir, "*.mp4")) # GIF O MP4
        patternscsvfolder = f"{nball}b TRACK 60"
        os.makedirs(patternscsvfolder, exist_ok=True)

        for VIDEO_PATH in video_files:            
            vid_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
            # Si Kalman falla, prueba con HSV manual más permisivo:
            track_balls_with_kalman(
                video_path=VIDEO_PATH,
                num_balls=nball,
                output_csv=os.path.join(patternscsvfolder, f"{vid_name}.csv"),
                auto_hsv=False,
                hsv_range=(19, 143, 107, 30, 255, 162),
                #nicomoradas(0, 0, 0, 255, 185, 114),
#stephmulti(35, 103, 84, 226, 214, 184),stephrojas:(156, 87, 74, 176, 223, 117),
#stephmorado(132, 98, 60, 170, 209, 239), #plaza(111, 121, 44, 207, 166, 255), 
#crisamarillo:(19, 143, 107, 30, 255, 162),#RojoYo:(0, 128, 90, 8, 227, 156), 
# #GIFS:(0, 255, 255, 0, 255, 255),#stephplaza:(62, 91, 27, 151, 187, 148),
                min_area=30,
                max_cost=271,
                visualize=True
            )
    # #Kalman Filter auto hsv (needs fix)
    # track_balls_with_kalman(
    #     video_path="/home/dar0j/Documentos/2025/intro trabajo titulo el E/PROJECT/Datasets/3b gifs/3_(4,2x)(2x,4).gif",
    #     num_balls=3,
    #     output_csv="tracking_auto_box.csv",
    #     auto_hsv=True,
    #     min_area=30,       # Ajustar según tamaño de pelotas en píxeles
    #     max_cost=361,      # Distancia máxima prediction-detection (píxeles)
    #     visualize=True
    # )