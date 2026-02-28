"""
auto_color_detect.py
Detecta colores de pelotas automáticamente usando BG subtraction + k-means HSV.
Integrable con batch_track.py y detect_balls.py.
"""
import cv2
import numpy as np
from pathlib import Path
import yaml


def detect_ball_colors(video_path, config, num_balls, n_sample_frames=50,
                       min_saturation=40, min_value=40):
    """
    Detecta los num_balls colores dominantes de las pelotas en el video.
    
    Problema del enfoque anterior: k-means en espacio HSV global mezcla
    pelotas de colores similares y fondo residual. 
    Fix: filtrar muestras de baja saturación (fondo/sombras) antes del k-means.
    
    Returns:
        list of dicts con 'center_hsv', 'lower', 'upper', 'count'
        None si no hay suficientes muestras
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        print(f"⚠️ No se pudo abrir: {video_path}")
        return None

    history   = config.get('history', 100)
    var_thr   = config.get('var_threshold', 25)
    mk        = config.get('morph_kernel_size', 3)
    min_area  = config.get('min_contour_area', 50)
    max_area  = config.get('max_contour_area', 5000)
    use_blur  = config.get('use_blur', True)
    blur_k    = config.get('blur_kernel', 5)
    morph_ops = config.get('morph_operations', ['open'])

    bg_method = config.get('bg_method', 'MOG2')
    if bg_method == 'KNN':
        bg_sub = cv2.createBackgroundSubtractorKNN(
            history=history, dist2Threshold=var_thr * 10, detectShadows=False)
    else:
        bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_thr, detectShadows=False)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))

    # Warmup del BG subtractor
    start_frame = min(history + 20, total // 4)
    for _ in range(start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if use_blur:
            frame = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
        bg_sub.apply(frame, learningRate=0.01)

    sample_indices = np.linspace(start_frame, total - 1,
                                  n_sample_frames, dtype=int)
    all_hsv_samples = []

    for target_frame in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_frame))
        ret, frame = cap.read()
        if not ret:
            continue

        proc = cv2.GaussianBlur(frame, (blur_k, blur_k), 0) if use_blur else frame
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        fg = bg_sub.apply(proc, learningRate=0.001)
        _, fg = cv2.threshold(fg, 250, 255, cv2.THRESH_BINARY)
        for op in morph_ops:
            if op == 'open':
                fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
            elif op == 'close':
                fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if not (min_area <= area <= max_area):
                continue

            mask = np.zeros(fg.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            pixels = hsv_frame[mask == 255]

            if len(pixels) < 5:
                continue

            # ✅ Filtrar píxeles de baja saturación (fondo/sombra)
            saturated = pixels[(pixels[:, 1] >= min_saturation) &
                                (pixels[:, 2] >= min_value)]
            if len(saturated) < 3:
                continue

            median_hsv = np.median(saturated, axis=0)
            all_hsv_samples.append(median_hsv)

    cap.release()

    if len(all_hsv_samples) < num_balls * 3:
        print(f"  ⚠️ Solo {len(all_hsv_samples)} muestras, necesito ≥{num_balls*3}")
        return None

    # K-means en HSV solo con Hue+Saturation (ignorar V = iluminación)
    samples = np.array(all_hsv_samples, dtype=np.float32)
    # Usar solo H y S para clustering (V varía con iluminación)
    samples_hs = samples[:, :2]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers_hs = cv2.kmeans(
        samples_hs, num_balls, None, criteria, 20, cv2.KMEANS_PP_CENTERS)

    colors = []
    for i, center_hs in enumerate(centers_hs):
        h, s = center_hs
        # V: usar mediana de muestras de este cluster
        cluster_mask = (labels.flatten() == i)
        v_vals = samples[cluster_mask, 2]
        v = float(np.median(v_vals))

        lower = np.array([max(0,   h-15), max(0,   s-50), max(0,   v-60)], dtype=np.uint8)
        upper = np.array([min(180, h+15), min(255, s+50), min(255, v+60)], dtype=np.uint8)

        count = int(np.sum(cluster_mask))
        colors.append({
            'center_hsv': (int(h), int(s), int(v)),
            'lower': lower.tolist(),
            'upper': upper.tolist(),
            'count': count
        })
        print(f"  🎨 Color {i+1}: H={int(h):3d} S={int(s):3d} V={int(v):3d}  "
              f"({count} muestras)")

    # Ordenar por count descendente (el color más frecuente = pelota más visible)
    colors.sort(key=lambda x: x['count'], reverse=True)
    return colors


def colors_to_config_dict(colors):
    """Convierte lista de colores a formato compatible con autocolortrack."""
    return {
        f'ball_{i+1}': {
            'lower_hsv': c['lower'],
            'upper_hsv': c['upper'],
            'center_hsv': list(c['center_hsv'])
        }
        for i, c in enumerate(colors)
    }


def save_color_config(colors, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)  # ✅ fix
    color_dict = colors_to_config_dict(colors)
    with open(output_path, 'w') as f:
        yaml.dump(color_dict, f, default_flow_style=False)
    print(f"  💾 Colores guardados: {output_path}")
    return color_dict