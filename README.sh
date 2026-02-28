# 0. Activar entorno
cd "/home/dar0j/Documentos/2025/intro trabajo titulo el E/PROJECT/Code"

# 1. DETECTAR pelotas con YOLO en todos los videos
python pipeline_detect.py \
  --model "100ep 1800imgs/best.pt" \
  --source "../Datasets/used to track" \
  --out_dir "runs/dets_cache_all" \
  --conf 0.25 --iou 0.7 \
  --skip_existing 2>&1 | grep -v "NNPACK"

# 2. TRACKEAR con OC-SORT + reparar fragmentación + generar CSVs
python pipeline_track_to_csv.py \
  --cache_dir runs/dets_cache_all \
  --visualize \
  --no_repair \
  --max_age 52 \
  --interp_gap 49 \
  --delta_t 1 \
  --min_hits 3 \
  --det_thresh    0.141 \
  --iou_threshold -0.263 \
  --asso_func     ciou

#Lee los videos originales y escribe real_fps en los .ndjson de deteccion
python fps_4_interp.py

# 3. AUGMENTAR datos
python data_augmentation_new.py \
  --data_root runs/track_csvs

# 4. (Opcional) TUNING de hiperparámetros por N pelotas
python hyperparameter_tuning_new.py \
  --data_root runs/track_csvs --n_balls 3

# 5. ENTRENAR modelos separados por N pelotas
python train_per_nballs.py \
  --no_sliding_window

# Para entrenar solo un modelo específico:
python train_per_nballs.py --data_root runs/track_csvs --n_balls 3





cd /home/dar0j/Documentos/2025/intro\ trabajo\ titulo\ el\ E/old\ rasmus/juggling-vision-py/ComputerVision/new\ approach
#1 preparar MOT dataset desde GTs
python prepare_gt_dataset.py \
    --input_dir datasets/gts+vids \
    --output_dir datasets/juggling-mot \
    --ball_radius 15

#2 tunear hiperparametros
python boxmot_tuner.py \
    --mot_dir datasets/juggling-mot \
    --tracker_type ocsort \
    --n_trials 50 \
    --output_config configs/best_config.yaml

#3  Tracking batch → formato autocolortrack
# Batch de múltiples videos:
python batch_track.py \
    --config configs/best_ocsort.yaml \
    --video_dir "../../../../PROJECT/Datasets/used to track" \
    --output_dir datasets/tracked \
    --visualize

#   O un solo video:
python batch_track.py \
    --config configs/best_config.yaml \
    --video "../../../../PROJECT/Datasets/used to track/already_something/video.mp4" \
    --output_dir datasets/tracked \
    --num_balls 3

#4  Siteswap detection con nohandlebars
# python siteswap_detector.py \
#     --input_dir datasets/tracked \
#     --output_dir datasets/siteswaps \
#     --num_balls 5
cd /home/dar0j/Documentos/2025/intro\ trabajo\ titulo\ el\ E/old\ rasmus/juggling-vision-py/ComputerVision

# Un CSV:
python nohandlebars.py --csv "new approach/datasets/tracked/video_name.csv" --nballs 5

# Batch (mover CSVs a carpetas "Nb TRACK 60"):
python nohandlebars.py --batch --base-dir "new approach/datasets/tracked"






# Demo interactiva (ver detecciones en vivo con sliders)
python preprocess_demo.py \
    --video_dir "../../../../PROJECT/Datasets/used to track" \
    --config configs/best_config_v5.yaml \
    --save configs/preprocess_tuned.yaml

# Después de ajustar los sliders y presionar 's':
python batch_track.py \
    --config configs/preprocess_tuned.yaml \
    --video_dir "../../../../PROJECT/Datasets/used to track" \
    --output_dir datasets/tracked \
    --auto_color \
    --color_config_dir datasets/colors

# Ver máscara + frame lado a lado: slider show_mask=2





┌──────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Detección   │ →   │ MOT format   │ →   │  BiTrack    │ →   │ nohandlebars │
│  (tuneada)   │     │  det.txt     │     │  (offline)  │     │  .py         │
└──────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
   BG subtract         frame,id=-1,        trayectorias        siteswap
   + contornos          x,y,w,h,conf       consistentes

cd "ComputerVision/new approach/bitrack"

# 1. Detectar pelotas (visualizable, verificable)
python detect_balls.py \
    --config configs/best_config_v5.yaml \
    --video_dir "../../../../PROJECT/Datasets/used to track" \
    --output_dir datasets/detections_mot \
    --visualize   # ← ventana en vivo para verificar

python track_from_dets.py \
    --det_dir datasets/detections_mot \
    --out_dir runs/track_csvs \
    --det_thresh 0.57 --iou_threshold -0.57 --asso_func giou \
    --max_age 90 --min_hits 3 --delta_t 1 --inertia 0.5
# 2. BiTrack (offline, toda la secuencia)
# cd al repo de BiTrack, correr con las detecciones
#python bitrack.py --input datasets/detections_mot --output datasets/bitrack_output

# 3. Convertir → CSV para nohandlebars
# python bitrack_to_csv.py \
#     --mot_dir datasets/bitrack_output \
#     --output_dir datasets/tracked

# 4. Siteswap
#python ../nohandlebars.py --csv datasets/tracked/5_cascade_001.csv --nballs 5
python ../nohandlebars.py --batch --base-dir runs/track_csvs





python realtime_siteswap.py \
    --source 0 \
    --nballs 3 \
    --config ../configs/best_config.yaml

webcam/video
    ↓ cada frame (~6ms)
BoxMOTJugglingTracker.detect_balls()  ← BG subtract + contornos
    ↓
tracker.tracker.update()              ← OcSort asigna IDs
    ↓
TrajectoryBuffer.update()             ← buffer circular N segundos
    ↓ cada 15 frames
analyze_buffer()                      ← picos Y + beats + siteswap
    ↓
find_largest_repeating_pattern()      ← patrón válido más repetido
    ↓
draw_overlay()                        ← muestra en pantalla