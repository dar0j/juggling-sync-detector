# app.py - CON SELECTOR DINÁMICO DE NÚMERO DE PELOTAS + COMPUTER VISION PIPELINE
from flask import Flask, request, jsonify, render_template, send_file
import os
import cv2
import numpy as np
from predict_trick import extract_coordinates_from_video, preprocess_sequence, predict_trick, create_overlay_video
import sys
sys.path.append("./rasmus")
sys.path.append("./ComputerVision")
from gridmodel import GridModel
from tensorflow.keras.models import load_model
import json
import tensorflow as tf
from tensorflow.python.keras import backend as K
import base64
from io import BytesIO
from PIL import Image

# Importar módulos de Computer Vision
from calibration import analyze_video_params
from autocolortrack import track_balls_with_kalman, auto_extract_hsv_range
from nohandlebars import pipeline as siteswap_pipeline

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============ Configurar sesión global de TF 1.x ============
session = tf.compat.v1.Session()
K.set_session(session)
graph = tf.compat.v1.get_default_graph()
# ============================================================

# Cache para GridModels (evitar recargar el mismo modelo)
grid_models_cache = {}

# Cargar TCN model y label_map al inicio (estos no cambian)
print("Cargando modelo TCN...")
with graph.as_default():
    K.set_session(session)
    tcn_model = load_model("fold_3_best.h5")
    with open("label_map.json") as f:
        label_map = json.load(f)
print("✓ Modelo TCN cargado")

def get_grid_model(nballs):
    """
    Obtiene GridModel para el número de pelotas especificado.
    Usa cache para evitar recargar modelos ya cargados.
    """
    if nballs not in grid_models_cache:
        print(f"Cargando GridModel para {nballs} pelotas...")
        with graph.as_default():
            K.set_session(session)
            grid_models_cache[nballs] = GridModel(
                "../grid_models/grid_model_submovavg_128x128.h5", 
                nBalls=nballs
            )
        print(f"✓ GridModel para {nballs} pelotas cargado")
    return grid_models_cache[nballs]

def frame_to_base64(frame):
    """Convierte frame de OpenCV a base64 para enviar al frontend"""
    # Redimensionar para web (max 640px de ancho)
    height, width = frame.shape[:2]
    if width > 640:
        scale = 640 / width
        new_width = 640
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    
    # Convertir a JPEG
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
        import traceback
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
        import traceback
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
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/process_computervision', methods=['POST'])
def process_computervision():
    """Pipeline completo: Tracking + Siteswap Detection"""
    data = request.get_json()
    video_filename = data['video_path']
    nballs = int(data['nballs'])
    hsv_range = tuple(data['hsv_range'])
    tracking_params = data['tracking_params']
    
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    csv_path = os.path.join(UPLOAD_FOLDER, f"tracking_{video_filename.replace('.mp4', '.csv')}")
    
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
        os.remove(video_path)
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
        # Limpiar en caso de error
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    """Pipeline Deep Learning (original)"""
    # Recibir video y parámetros
    video = request.files['video']
    nballs = int(request.form.get('nballs', 4))  # Número de pelotas
    fps = int(request.form.get('fps', 60))
    create_video = request.form.get('create_video', 'false') == 'true'
    
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    
    try:
        # Cargar GridModel apropiado para el número de pelotas
        grid_model = get_grid_model(nballs)
        
        # Procesar
        print(f"Procesando {video.filename} con {nballs} pelotas a {fps} fps...")
        
        # ============ FIX: Usar sesión dentro del contexto ============
        with graph.as_default():
            K.set_session(session)
            coords, n_balls = extract_coordinates_from_video(
                video_path, grid_model, target_fps=fps
            )
            preprocessed = preprocess_sequence(coords, n_balls)
            predictions = predict_trick(tcn_model, preprocessed, label_map, top_k=5)
        
        result = {
            'success': True,
            'predictions': predictions,
            'n_balls': int(n_balls),
            'n_frames': int(coords.shape[0])
        }
        
        # Crear video con overlay si se solicita
        if create_video:
            output_path = os.path.join(UPLOAD_FOLDER, f"result_{video.filename}")
            print(f"Generando video con overlay...")
            with graph.as_default():
                K.set_session(session)
                create_overlay_video(
                    video_path, 
                    output_path, 
                    grid_model,
                    predictions, 
                    fps=fps
                )
            result['video_url'] = f'/download/{os.path.basename(output_path)}'
            print(f"✓ Video guardado en {output_path}")
        
        # Limpiar video original
        os.remove(video_path)
        
        return jsonify(result)
        
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download/<filename>')
def download(filename):
    """Descargar video con overlay"""
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    # Usar un solo thread (evita problemas de sesión múltiple)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False)