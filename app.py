# app.py - CON SELECTOR DINÁMICO DE NÚMERO DE PELOTAS
from flask import Flask, request, jsonify, render_template, send_file
import os
import cv2
import numpy as np
from predict_trick import extract_coordinates_from_video, preprocess_sequence, predict_trick, create_overlay_video
import sys
sys.path.append("./rasmus")
from gridmodel import GridModel
from tensorflow.keras.models import load_model
import json
import tensorflow as tf
from tensorflow.python.keras import backend as K

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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