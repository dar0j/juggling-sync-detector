import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def analyze_video_params(video_path, hsv_range, num_samples=100, show_detections=True):
    """
    Analiza el video para calibrar parámetros de tracking.
    
    Returns:
        dict con:
        - ball_size_mean: tamaño promedio de pelotas (píxeles²)
        - ball_size_std: desviación estándar del tamaño
        - ball_diameter_mean: diámetro promedio (píxeles)
        - motion_noise: ruido de movimiento (desviación de velocidad)
        - detection_noise: ruido de detección (desviación de posición)
        - recommended_params: parámetros sugeridos
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    h_min, s_min, v_min, h_max, s_max, v_max = hsv_range
    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Almacenar datos de análisis
    ball_areas = []
    ball_positions = []  # [(frame_idx, ball_id, x, y), ...]
    velocities = []
    position_jitter = []  # para medir ruido de detección
    
    sample_indices = np.linspace(10, total_frames-10, num_samples, dtype=int)
    prev_detections = None
    
    print(f"Analizando {num_samples} frames del video...")
    
    for sample_idx, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detección
        fg_mask = bg_subtractor.apply(frame, learningRate=0.001)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, 
                                 np.array([h_min, s_min, v_min]), 
                                 np.array([h_max, s_max, v_max]))
        
        combined_mask = cv2.bitwise_and(fg_mask, color_mask)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30 or area > 10000:
                continue
            
            # Calcular diámetro aproximado
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Solo tomar objetos razonablemente circulares (pelotas)
            if circularity > 0.5:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Diámetro aproximado
                    diameter = np.sqrt(4 * area / np.pi)
                    
                    ball_areas.append(area)
                    detections.append({'pos': (cx, cy), 'area': area, 'diameter': diameter, 'circularity': circularity})
        
        # Calcular velocidades si tenemos frame previo
        if prev_detections and len(detections) > 0 and len(prev_detections) > 0:
            # Asociación simple con Hungarian
            prev_pos = np.array([d['pos'] for d in prev_detections], dtype=np.float32)
            curr_pos = np.array([d['pos'] for d in detections], dtype=np.float32)
            
            if len(prev_pos) > 0 and len(curr_pos) > 0:
                n = min(len(prev_pos), len(curr_pos))
                cost = np.linalg.norm(prev_pos[:, None, :] - curr_pos[None, :, :], axis=2)
                rows, cols = linear_sum_assignment(cost[:n, :n])
                
                for r, c in zip(rows, cols):
                    if cost[r, c] < 300:  # umbral razonable
                        dx = curr_pos[c][0] - prev_pos[r][0]
                        dy = curr_pos[c][1] - prev_pos[r][1]
                        velocity = np.sqrt(dx*dx + dy*dy)
                        velocities.append(velocity)
                        
                        # Jitter: medir desviación en detección (cambios pequeños no físicos)
                        if velocity < 5:  # pelota "quieta" (catch)
                            position_jitter.append(velocity)
        
        prev_detections = detections
        
        # Visualización opcional
        if show_detections and sample_idx < 5:
            vis = frame.copy()
            for det in detections:
                pos = det['pos']
                cv2.circle(vis, pos, int(det['diameter']/2), (0, 255, 0), 2)
                cv2.putText(vis, f"{det['area']:.0f}px", (pos[0]+10, pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imshow('Calibration Sample', cv2.resize(vis, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey(500)
        
        if sample_idx % 10 == 0:
            print(f"  {sample_idx}/{num_samples}...")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Análisis estadístico
    if not ball_areas:
        print("⚠ No se detectaron pelotas. Verifica hsv_range.")
        return None
    
    ball_areas = np.array(ball_areas)
    velocities = np.array(velocities) if velocities else np.array([0])
    position_jitter = np.array(position_jitter) if position_jitter else np.array([0])
    
    area_mean = np.mean(ball_areas)
    area_std = np.std(ball_areas)
    diameter_mean = np.sqrt(4 * area_mean / np.pi)
    diameter_std = np.sqrt(4 * area_std / np.pi)
    
    velocity_mean = np.mean(velocities)
    velocity_std = np.std(velocities)
    
    jitter_mean = np.mean(position_jitter)
    jitter_std = np.std(position_jitter)
    
    # Recomendar parámetros
    min_area = max(30, int(area_mean - 2*area_std))
    max_area = int(area_mean + 3*area_std)
    
    # Process noise: proporcional a velocidad (mayor si pelotas rápidas)
    # Regla empírica: process_noise ≈ (velocity_std / fps)^2
    process_noise = max(0.01, min(0.5, (velocity_std / fps) ** 2))
    
    # Measurement noise: proporcional a jitter de detección
    # Regla empírica: measurement_noise ≈ jitter_std^2
    measurement_noise = max(1.0, min(50.0, jitter_std ** 2))
    
    # Max cost para Hungarian: ~3x desplazamiento típico entre frames
    max_cost = int(velocity_mean * 3 + 50)
    
    results = {
        'ball_size_mean': area_mean,
        'ball_size_std': area_std,
        'ball_diameter_mean': diameter_mean,
        'ball_diameter_std': diameter_std,
        'motion_velocity_mean': velocity_mean,
        'motion_velocity_std': velocity_std,
        'detection_jitter_mean': jitter_mean,
        'detection_jitter_std': jitter_std,
        'num_detections': len(ball_areas),
        'recommended_params': {
            'min_area': min_area,
            'max_area': max_area,
            'max_cost': max_cost,
            'process_noise_cov': process_noise,
            'measurement_noise_cov': measurement_noise,
            'kalman_max_lost': int(fps * 0.5)  # 0.5 segundos de tolerancia
        }
    }
    
    # Imprimir resumen
    print("\n" + "="*60)
    print("CALIBRACIÓN DE PARÁMETROS")
    print("="*60)
    print(f"Tamaño de pelotas:")
    print(f"  Área promedio:    {area_mean:.1f} ± {area_std:.1f} px²")
    print(f"  Diámetro promedio: {diameter_mean:.1f} ± {diameter_std:.1f} px")
    print(f"\nMovimiento:")
    print(f"  Velocidad promedio: {velocity_mean:.1f} ± {velocity_std:.1f} px/frame")
    print(f"  Velocidad en px/s:  {velocity_mean * fps:.1f} px/s")
    print(f"\nRuido de detección:")
    print(f"  Jitter promedio:    {jitter_mean:.2f} ± {jitter_std:.2f} px")
    print(f"\n{'PARÁMETROS RECOMENDADOS':^60}")
    print("-"*60)
    print(f"  min_area = {min_area}")
    print(f"  max_area = {max_area}")
    print(f"  max_cost = {max_cost}")
    print(f"  processNoiseCov = {process_noise:.4f}")
    print(f"  measurementNoiseCov = {measurement_noise:.1f}")
    print(f"  max_lost_frames = {int(fps * 0.5)}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    VIDEO_PATH = "/home/dar0j/Documentos/2025/intro trabajo titulo el E/PROJECT/Datasets/5b/5_(6x,4x)_12.mp4"
    
    # Usa tu rango HSV conocido (o el detectado automáticamente)
    HSV_RANGE = (117, 116, 28, 193, 255, 174)  # del código anterior
    
    # Ejecutar calibración
    params = analyze_video_params(
        video_path=VIDEO_PATH,
        hsv_range=HSV_RANGE,
        num_samples=100,
        show_detections=True  # mostrar primeros 5 frames con detecciones
    )
    
    if params:
        print("\nCopia estos parámetros a tu código:")
        print(f"""
track_balls_with_kalman(
    video_path=VIDEO_PATH,
    num_balls=5,
    output_csv="tracking_calibrated.csv",
    auto_hsv=False,
    hsv_range={HSV_RANGE},
    min_area={params['recommended_params']['min_area']},
    max_cost={params['recommended_params']['max_cost']},
    visualize=True
)

# Y en KalmanBallTracker.__init__():
self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * {params['recommended_params']['process_noise_cov']:.4f}
self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * {params['recommended_params']['measurement_noise_cov']:.1f}
self.max_lost = {params['recommended_params']['kalman_max_lost']}
""")