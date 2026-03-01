# app.py - YOLO NANO + OC-SORT + TCN per-nballs + Computer Vision Pipeline
from flask import Flask, request, jsonify, render_template, send_file, Response
import os
import cv2
import numpy as np
import sys
sys.path.append("./rasmus")
sys.path.append("./ComputerVision")
import json
import base64
import traceback

# Importar módulos de Computer Vision
from calibration import analyze_video_params
from autocolortrack import track_balls_with_kalman, auto_extract_hsv_range
from nohandlebars import pipeline as siteswap_pipeline

# Importar nuevo pipeline DL
from pipeline_dl import DLPipeline

# Importar pipelines de tiempo real
from pipeline_realtime import RealtimeDLPipeline, RealtimeCVPipeline, SEQ_LEN

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============ Inicializar pipeline DL (YOLO + OC-SORT + TCN) ============
print("Inicializando pipeline Deep Learning...")
dl_pipeline = DLPipeline(
    yolo_model_path="MODELS/NANO.pt",
    models_dir="MODELS/VIDEO",
    ocsort_vendor_dir="ocsort"
)
print("✓ Pipeline DL listo")
# =========================================================================

# ============ Pipelines de tiempo real ============
print("Inicializando pipelines de tiempo real...")
realtime_dl = RealtimeDLPipeline(
    yolo_model_path="MODELS/NANO.pt",
    models_dir="MODELS/WINDOW",
    confidence_threshold=0.3
)
realtime_cv = RealtimeCVPipeline()
print("✓ Pipelines de tiempo real listos")

# Estado global de streaming (simplificado para un usuario)
_stream_state = {
    "active": False,
    "mode": None,     # 'deeplearning', 'computervision'
    "nballs": 4,
    "hsv_range": None,
    "min_area": 100,
    "camera_fps": 30,   # FPS real de la cámara
    "calibrating": False,  # True durante calibración HSV en vivo
}
# ==================================================

def frame_to_base64(frame):
    """Convierte frame de OpenCV a base64 para enviar al frontend"""
    # Redimensionar para web (max 640px de ancho)
    height, width = frame.shape[:2]
    if width > 640:
        scale = 640 / width
        new_width = 640
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/extract_frames', methods=['POST'])
def extract_frames():
    """Extrae 5 frames del video para calibración HSV"""
    video = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extraer 5 frames distribuidos uniformemente
        frame_indices = np.linspace(10, total_frames - 10, 5, dtype=int)
        frames_base64 = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames_base64.append(frame_to_base64(frame))
        
        cap.release()
        
        return jsonify({
            'success': True,
            'frames': frames_base64,
            'video_path': video.filename  # Guardar para siguiente paso
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/preview_mask', methods=['POST'])
def preview_mask():
    """Previsualiza máscara HSV con parámetros ajustables"""
    data = request.get_json()
    video_filename = data['video_path']
    hsv_range = tuple(data['hsv_range'])
    frame_index = data.get('frame_index', 0)
    
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(10, total_frames - 10, 5, dtype=int)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[frame_index])
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'success': False, 'error': 'No se pudo leer frame'})
        
        # Aplicar máscara HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_min, s_min, v_min, h_max, s_max, v_max = hsv_range
        mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]),
                                 np.array([h_max, s_max, v_max]))
        
        # Aplicar máscara al frame original
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convertir máscara a 3 canales para visualización
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        return jsonify({
            'success': True,
            'original': frame_to_base64(frame),
            'mask': frame_to_base64(mask_color),
            'result': frame_to_base64(result)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/calibrate_tracking', methods=['POST'])
def calibrate_tracking():
    """Calibra parámetros de tracking basándose en HSV y video"""
    data = request.get_json()
    video_filename = data['video_path']
    hsv_range = tuple(data['hsv_range'])
    
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    
    try:
        print(f"Calibrando parámetros para {video_filename}...")
        params = analyze_video_params(
            video_path=video_path,
            hsv_range=hsv_range,
            num_samples=50,
            show_detections=False
        )
        
        if params is None:
            return jsonify({
                'success': False,
                'error': 'No se detectaron pelotas. Ajusta el rango HSV.'
            })
        
        return jsonify({
            'success': True,
            'params': {
                'min_area': int(params['recommended_params']['min_area']),
                'max_cost': int(params['recommended_params']['max_cost']),
                'process_noise_cov': float(params['recommended_params']['process_noise_cov']),
                'measurement_noise_cov': float(params['recommended_params']['measurement_noise_cov'])
            },
            'stats': {
                'ball_diameter_mean': float(params['ball_diameter_mean']),
                'motion_velocity_mean': float(params['motion_velocity_mean']),
                'detection_jitter_mean': float(params['detection_jitter_mean'])
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/process_computervision', methods=['POST'])
def process_computervision():
    """Pipeline Computer Vision: HSV tracking + Siteswap Detection"""
    data = request.get_json()
    video_filename = data['video_path']
    nballs = int(data['nballs'])
    hsv_range = tuple(data['hsv_range'])
    tracking_params = data['tracking_params']
    
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    csv_path = os.path.join(UPLOAD_FOLDER,
                            f"tracking_{video_filename.replace('.mp4', '.csv')}")
    
    try:
        # PASO 1: Tracking con Kalman Filter
        print(f"Iniciando tracking para {nballs} pelotas...")
        track_balls_with_kalman(
            video_path=video_path,
            num_balls=nballs,
            output_csv=csv_path,
            auto_hsv=False,
            hsv_range=hsv_range,
            min_area=tracking_params['min_area'],
            max_cost=tracking_params['max_cost'],
            visualize=False
        )
        print(f"✓ Tracking completado: {csv_path}")
        
        # PASO 2: Detección de siteswap
        print("Detectando siteswap...")
        siteswap_result = siteswap_pipeline(
            csv_path,
            n_balls=nballs,
            smooth_window=9,
            prominence=6,
            distance=8,
            frame_window=7,
            use_median=True,
            interpolate=True,
            visualize=False
        )
        
        # Limpiar archivos temporales
        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        return jsonify({
            'success': True,
            'siteswap': siteswap_result['siteswap'],
            'siteswap_canonical': siteswap_result['siteswap_canonical'],
            'period_length': siteswap_result['period_length'],
            'num_peaks': siteswap_result['num_peaks'],
            'x_center': float(siteswap_result['x_center'])
        })
        
    except Exception as e:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    """Pipeline Deep Learning: YOLO NANO → OC-SORT → TCN per-nballs"""
    video = request.files['video']
    nballs = int(request.form.get('nballs', 4))
    
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    
    try:
        print(f"Procesando {video.filename} con {nballs} pelotas (DL pipeline)...")
        
        result = dl_pipeline.process_video(
            video_path=video_path,
            nballs=nballs,
            top_k=5
        )
        
        # Limpiar video
        if os.path.exists(video_path):
            os.remove(video_path)
        
        return jsonify({
            'success': True,
            'predictions': result['predictions'],
            'n_balls': result['n_balls'],
            'n_frames': result['n_frames'],
        })
        
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/predict_hybrid', methods=['POST'])
def predict_hybrid():
    """Pipeline Híbrido: YOLO NANO → OC-SORT → nohandlebars (siteswap)"""
    video = request.files['video']
    nballs = int(request.form.get('nballs', 4))
    
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    
    try:
        print(f"Procesando {video.filename} con {nballs} pelotas (Hybrid pipeline)...")
        
        # Paso 1-2: YOLO + OC-SORT (reutilizar del DL pipeline)
        arr, video_fps = dl_pipeline.detect_and_track(
            video_path=video_path,
            nballs=nballs
        )
        
        # Paso 3: Guardar tracking como CSV temporal para nohandlebars
        csv_path = os.path.join(UPLOAD_FOLDER,
                                f"hybrid_{video.filename.replace('.mp4', '.csv')}")
        _tracking_array_to_csv(arr, nballs, csv_path)
        print(f"  Tracking CSV guardado: {csv_path}")
        
        # Paso 4: Detección de siteswap con nohandlebars
        print("  Detectando siteswap con nohandlebars...")
        siteswap_result = siteswap_pipeline(
            csv_path,
            n_balls=nballs,
            smooth_window=9,
            prominence=6,
            distance=8,
            frame_window=7,
            use_median=True,
            interpolate=True,
            visualize=False
        )
        
        # Limpiar
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        return jsonify({
            'success': True,
            'siteswap': siteswap_result['siteswap'],
            'siteswap_canonical': siteswap_result['siteswap_canonical'],
            'period_length': siteswap_result['period_length'],
            'num_peaks': siteswap_result['num_peaks'],
            'x_center': float(siteswap_result['x_center']),
            'n_frames': int(arr.shape[0]),
            'n_balls': nballs,
        })
        
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        csv_path_check = os.path.join(UPLOAD_FOLDER,
                                       f"hybrid_{video.filename.replace('.mp4', '.csv')}")
        if os.path.exists(csv_path_check):
            os.remove(csv_path_check)
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


def _tracking_array_to_csv(arr: np.ndarray, nballs: int, csv_path: str):
    """Convierte array de tracking a CSV compatible con nohandlebars (sin header, solo x,y pares)."""
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for frame_i in range(arr.shape[0]):
            row = []
            for b in range(nballs):
                x = arr[frame_i, b * 2]
                y = arr[frame_i, b * 2 + 1]
                row.append('' if x == -1.0 else f'{x:.2f}')
                row.append('' if y == -1.0 else f'{y:.2f}')
            writer.writerow(row)


@app.route('/download/<filename>')
def download(filename):
    """Descargar archivo"""
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)


@app.route('/realtime/calibrate_frame', methods=['POST'])
def realtime_calibrate_frame():
    """
    Recibe un frame de la cámara durante calibración HSV.
    Aplica la máscara y retorna el frame anotado con detecciones.
    """
    data = request.get_json()
    img_b64 = data['frame']
    hsv_range = tuple(data['hsv_range'])  # [h_min, s_min, v_min, h_max, s_max, v_max]
    min_area = int(data.get('min_area', 100))

    try:
        header, encoded = img_b64.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    if frame is None:
        return jsonify({"success": False, "error": "Frame inválido"})

    h, w = frame.shape[:2]
    h_min, s_min, v_min, h_max, s_max, v_max = hsv_range

    # Aplicar HSV mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]),
                             np.array([h_max, s_max, v_max]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Detectar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_detections = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > 5000:
            continue
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            r = int(np.sqrt(area / np.pi))
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            n_detections += 1

    # Overlay de máscara semitransparente
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_color[:, :, 0] = 0  # solo green channel
    mask_color[:, :, 2] = 0
    frame = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)

    # Texto de detecciones
    cv2.putText(frame, f"{n_detections} detecciones", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return jsonify({
        "success": True,
        "annotated_frame": frame_to_base64(frame),
        "n_detections": n_detections,
    })


@app.route('/realtime/start', methods=['POST'])
def realtime_start():
    """Inicializa sesión de tiempo real."""
    data = request.get_json()
    mode = data['mode']
    nballs = int(data['nballs'])
    camera_fps = float(data.get('camera_fps', 30))

    _stream_state["active"] = True
    _stream_state["mode"] = mode
    _stream_state["nballs"] = nballs
    _stream_state["camera_fps"] = camera_fps
    _stream_state["calibrating"] = False

    try:
        if mode == 'deeplearning':
            realtime_dl.initialize(nballs)
            return jsonify({"success": True,
                            "message": f"DL real-time iniciado ({nballs}b)",
                            "seq_len": SEQ_LEN,
                            "camera_fps": camera_fps})
        elif mode == 'computervision':
            hsv_range = tuple(data['hsv_range'])
            min_area = int(data.get('min_area', 100))
            _stream_state["hsv_range"] = hsv_range
            _stream_state["min_area"] = min_area
            realtime_cv.initialize(nballs, hsv_range, min_area)
            # Buffer mínimo depende de FPS de cámara: ~2 segundos
            min_frames = int(camera_fps * 2)
            return jsonify({"success": True,
                            "message": f"CV real-time iniciado ({nballs}b)",
                            "min_frames": min_frames,
                            "camera_fps": camera_fps})
        else:
            return jsonify({"success": False, "error": f"Modo no soportado: {mode}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route('/realtime/stop', methods=['POST'])
def realtime_stop():
    """Detiene sesión de tiempo real."""
    _stream_state["active"] = False
    _stream_state["mode"] = None
    return jsonify({"success": True})


@app.route('/realtime/frame', methods=['POST'])
def realtime_frame():
    """
    Recibe un frame del navegador (base64), lo procesa y retorna resultado.
    """
    if not _stream_state["active"]:
        return jsonify({"success": False, "error": "Sesión no activa"})

    data = request.get_json()
    img_b64 = data['frame']

    try:
        header, encoded = img_b64.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"success": False, "error": f"Error decodificando frame: {e}"})

    if frame is None:
        return jsonify({"success": False, "error": "Frame inválido"})

    mode = _stream_state["mode"]
    camera_fps = _stream_state.get("camera_fps", 30)

    try:
        if mode == 'deeplearning':
            result = realtime_dl.process_frame(frame)
            smoothed = realtime_dl.get_smoothed_prediction()

            response = {
                "success": True,
                "buffer_size": result["buffer_size"],
                "seq_len": SEQ_LEN,
                "n_detections": len(result["detections"]),
                "n_tracks": len(result["tracks"]),
                "annotated_frame": frame_to_base64(result["annotated_frame"]),
                "camera_fps": camera_fps,
            }

            if result["prediction"]:
                response["prediction"] = {
                    "trick": result["prediction"]["trick"],
                    "confidence": result["prediction"]["confidence"],
                }

            if smoothed:
                response["smoothed_prediction"] = {
                    "trick": smoothed["trick"],
                    "confidence": smoothed["confidence"],
                    "n_windows": smoothed["n_windows"],
                }

            return jsonify(response)

        elif mode == 'computervision':
            result = realtime_cv.process_frame(frame)
            min_frames = int(camera_fps * 2)

            response = {
                "success": True,
                "buffer_size": result["buffer_size"],
                "min_frames": min_frames,
                "n_detections": len(result["detections"]),
                "n_tracks": len(result["tracks"]),
                "annotated_frame": frame_to_base64(result["annotated_frame"]),
                "camera_fps": camera_fps,
            }

            if result["siteswap"] and result["siteswap"].get("canonical"):
                response["siteswap"] = result["siteswap"]

            return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    # Usar un solo thread (evita problemas de sesión múltiple)
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=False)