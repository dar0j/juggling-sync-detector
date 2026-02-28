"""
Detección de pelotas con parámetros tuneados.
Salida: detecciones en formato MOT Challenge (det.txt)
Opción --visualize para verificar calidad antes de pasar al tracker.
"""
import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path


def detect_balls_in_frame(frame, bg_subtractor, config, kernel):
    """Detección pura, sin tracking."""
    if config.get('use_blur', True):
        bk = config.get('blur_kernel', 5)
        proc = cv2.GaussianBlur(frame, (bk, bk), 0)
    else:
        proc = frame

    fg_mask = bg_subtractor.apply(proc, learningRate=0.001)
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

    for op in config.get('morph_operations', ['open']):
        if op == 'open':
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        elif op == 'close':
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    min_area = config.get('min_contour_area', 50)
    max_area = config.get('max_contour_area', 5000)
    enc_diff = config.get('enclosing_area_diff', 0.5)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        _, radius = cv2.minEnclosingCircle(c)
        enclosing_area = np.pi * radius * radius
        circularity = area / enclosing_area if enclosing_area > 0 else 0

        is_circular = abs(area - enclosing_area) < enc_diff * enclosing_area
        approx = cv2.approxPolyDP(c, 0.1 * cv2.arcLength(c, True), True)
        is_convex = len(approx) > 3 and cv2.isContourConvex(approx)

        if is_circular or is_convex:
            x, y, w, h = cv2.boundingRect(c)
            conf = float(np.clip(circularity, 0.3, 0.99))
            detections.append([x, y, w, h, conf])

    return detections, fg_mask


def process_video(video_path, config, output_path, visualize=False):
    """Procesa un video y guarda detecciones en formato MOT."""
    cap = cv2.VideoCapture(video_path)

    var_thresh = config.get('var_threshold', 25)
    history = config.get('history', 100)
    bg_method = config.get('bg_method', 'MOG2')

    if bg_method == 'KNN':
        bg_sub = cv2.createBackgroundSubtractorKNN(
            history=history, dist2Threshold=var_thresh * 10, detectShadows=False)
    else:
        bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_thresh, detectShadows=False)

    mk = config.get('morph_kernel_size', 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))

    all_dets = []
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets, fg_mask = detect_balls_in_frame(frame, bg_sub, config, kernel)

        for d in dets:
            x, y, w, h, conf = d
            # MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
            all_dets.append(f"{frame_idx},-1,{x},{y},{w},{h},{conf:.4f},-1,-1,-1")

        if visualize:
            vis = frame.copy()
            for d in dets:
                x, y, w, h, conf = d
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(vis, f"{conf:.2f}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imshow('Detections', vis)
            cv2.imshow('FG Mask', fg_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    if visualize:
        cv2.destroyAllWindows()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(all_dets))

    print(f"  💾 {frame_idx-1} frames, {len(all_dets)} detecciones → {output_path}")


def batch_detect(config_path, video_dir, output_dir, visualize=False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Remover parámetros de tracking (solo usar detección)
    config.pop('best_mota', None)
    config.pop('tracker_type', None)
    config.pop('det_thresh', None)
    config.pop('track_high_thresh', None)
    config.pop('max_age', None)
    config.pop('track_buffer', None)
    config.pop('match_thresh', None)
    config.pop('min_hits', None)
    config.pop('asso_func', None)
    config.pop('delta_t', None)
    config.pop('inertia', None)
    config.pop('use_byte', None)
    config.pop('min_conf', None)
    config.pop('iou_threshold', None)

    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    video_files = sorted(
        list(video_dir.glob('**/*.mp4')) +
        list(video_dir.glob('**/*.avi')) +
        list(video_dir.glob('**/*.gif'))
    )

    print(f"🔍 Detectando pelotas en {len(video_files)} videos")
    print(f"📄 Config: {config_path}")

    for i, vf in enumerate(video_files):
        print(f"\n[{i+1}/{len(video_files)}] {vf.name}")

        # Crear estructura MOT: output_dir/seq_name/det/det.txt
        seq_name = vf.stem
        det_path = output_dir / seq_name / "det" / "det.txt"

        process_video(str(vf), config, str(det_path), visualize)

    print(f"\n✅ Detecciones guardadas en: {output_dir}")
    print(f"   Formato: MOT Challenge (compatibles con BiTrack)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detección de pelotas → formato MOT')
    parser.add_argument('--config', required=True, help='YAML con parámetros de detección')
    parser.add_argument('--video_dir', required=True, help='Carpeta con videos')
    parser.add_argument('--output_dir', default='datasets/detections_mot')
    parser.add_argument('--visualize', action='store_true',
                        help='Mostrar ventana en vivo con detecciones')
    args = parser.parse_args()

    batch_detect(args.config, args.video_dir, args.output_dir, args.visualize)