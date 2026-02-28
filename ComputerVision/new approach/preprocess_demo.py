#!/usr/bin/env python3
"""
preprocess_demo.py
Demo interactiva para tunear parámetros de BG subtraction y detección.

Controles:
  ← →    : cambiar video
  SPACE  : pausa/play
  r      : resetear BG subtractor
  s      : guardar config actual como YAML
  q      : salir

Sliders (ventana 'Params'):
  history, var_threshold, morph_kernel, min_area, max_area,
  blur_kernel, prominence, distance, frame_window

Uso:
  python preprocess_demo.py --video_dir "../../../../PROJECT/Datasets/used to track"
  python preprocess_demo.py --video_dir "." --config configs/best_config.yaml
"""
import cv2
import numpy as np
import argparse
import yaml
from pathlib import Path
import sys


# ── Estado global de sliders ─────────────────────────────────────────────────
STATE = {
    'history':        100,
    'var_threshold':  25,
    'morph_kernel':   3,
    'min_area':       50,
    'max_area':       5000,
    'blur_kernel':    5,    # 0 = sin blur
    'use_blur':       1,
    'prominence':     6,
    'distance':       8,
    'frame_window':   7,
    'morph_close':    0,    # 0=solo open, 1=open+close
    'show_mask':      0,    # 0=frame, 1=mask, 2=ambos
}

bg_sub = None
needs_reset = True


def make_bg_sub():
    global bg_sub, needs_reset
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=STATE['history'],
        varThreshold=STATE['var_threshold'],
        detectShadows=False
    )
    needs_reset = False


def process_frame(frame):
    global bg_sub, needs_reset

    if needs_reset or bg_sub is None:
        make_bg_sub()

    # Blur
    if STATE['use_blur'] and STATE['blur_kernel'] >= 3:
        bk = STATE['blur_kernel'] | 1  # asegurar impar
        proc = cv2.GaussianBlur(frame, (bk, bk), 0)
    else:
        proc = frame.copy()

    # BG subtraction
    fg = bg_sub.apply(proc, learningRate=0.001)
    _, fg = cv2.threshold(fg, 250, 255, cv2.THRESH_BINARY)

    # Morfología
    mk = max(3, STATE['morph_kernel'] | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    if STATE['morph_close']:
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    # Contornos
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    all_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        all_contours.append((c, area))
        if STATE['min_area'] <= area <= STATE['max_area']:
            _, radius = cv2.minEnclosingCircle(c)
            enc_area = np.pi * radius * radius
            circularity = area / enc_area if enc_area > 0 else 0
            valid_contours.append((c, area, circularity))

    return fg, valid_contours, all_contours


def draw_info(frame, valid_contours, all_contours, video_name, frame_idx,
              total_frames, video_idx, total_videos):
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Todos los contornos en gris tenue
    for c, area in all_contours:
        if area >= STATE['min_area'] and area <= STATE['max_area']:
            pass  # ya se dibuja abajo
        else:
            cv2.drawContours(vis, [c], -1, (60, 60, 60), 1)

    # Contornos válidos en verde con info
    for i, (c, area, circ) in enumerate(valid_contours):
        x, y, cw, ch = cv2.boundingRect(c)
        cv2.rectangle(vis, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
        cv2.putText(vis, f"a={int(area)} c={circ:.2f}",
                    (x, max(0, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 255, 100), 1)

    # Panel de info superior
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    cv2.putText(vis, f"[{video_idx+1}/{total_videos}] {video_name}",
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(vis, f"Frame {frame_idx}/{total_frames}  "
                     f"Dets: {len(valid_contours)} / {len(all_contours)}",
                (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Parámetros actuales
    param_str = (f"hist={STATE['history']} var={STATE['var_threshold']} "
                 f"mk={STATE['morph_kernel']} "
                 f"area=[{STATE['min_area']},{STATE['max_area']}] "
                 f"blur={'ON' if STATE['use_blur'] else 'OFF'}{STATE['blur_kernel']}")
    cv2.putText(vis, param_str, (5, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1)

    analysis_str = (f"prominence={STATE['prominence']} "
                    f"distance={STATE['distance']} "
                    f"frame_win={STATE['frame_window']} "
                    f"close={'ON' if STATE['morph_close'] else 'OFF'}")
    cv2.putText(vis, analysis_str, (5, 73),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 220, 180), 1)

    return vis


def create_trackbars(window_name):
    """Crea todos los sliders en la ventana de parámetros."""
    def cb_reset(_):
        global needs_reset
        needs_reset = True

    cv2.createTrackbar('history        ', window_name, STATE['history'],        500,  cb_reset)
    cv2.createTrackbar('var_threshold  ', window_name, STATE['var_threshold'],  100,  cb_reset)
    cv2.createTrackbar('morph_kernel   ', window_name, STATE['morph_kernel'],   15,   lambda _: None)
    cv2.createTrackbar('min_area       ', window_name, STATE['min_area'],       500,  lambda _: None)
    cv2.createTrackbar('max_area /10   ', window_name, STATE['max_area']//10,   2000, lambda _: None)
    cv2.createTrackbar('blur_kernel    ', window_name, STATE['blur_kernel'],    20,   lambda _: None)
    cv2.createTrackbar('use_blur  0/1  ', window_name, STATE['use_blur'],       1,    lambda _: None)
    cv2.createTrackbar('morph_close 0/1', window_name, STATE['morph_close'],    1,    lambda _: None)
    cv2.createTrackbar('prominence     ', window_name, int(STATE['prominence']), 20,  lambda _: None)
    cv2.createTrackbar('distance       ', window_name, STATE['distance'],       30,   lambda _: None)
    cv2.createTrackbar('frame_window   ', window_name, STATE['frame_window'],   20,   lambda _: None)
    cv2.createTrackbar('show 0=vid 1=mask 2=both', window_name, 0,             2,    lambda _: None)


def read_trackbars(window_name):
    global needs_reset
    old_hist = STATE['history']
    old_var  = STATE['var_threshold']

    STATE['history']       = max(10, cv2.getTrackbarPos('history        ',        window_name))
    STATE['var_threshold'] = max(1,  cv2.getTrackbarPos('var_threshold  ',        window_name))
    STATE['morph_kernel']  = max(1,  cv2.getTrackbarPos('morph_kernel   ',        window_name))
    STATE['min_area']      =         cv2.getTrackbarPos('min_area       ',        window_name)
    STATE['max_area']      =         cv2.getTrackbarPos('max_area /10   ',        window_name) * 10
    STATE['blur_kernel']   = max(3,  cv2.getTrackbarPos('blur_kernel    ',        window_name))
    STATE['use_blur']      =         cv2.getTrackbarPos('use_blur  0/1  ',        window_name)
    STATE['morph_close']   =         cv2.getTrackbarPos('morph_close 0/1',        window_name)
    STATE['prominence']    = max(1,  cv2.getTrackbarPos('prominence     ',        window_name))
    STATE['distance']      = max(1,  cv2.getTrackbarPos('distance       ',        window_name))
    STATE['frame_window']  = max(1,  cv2.getTrackbarPos('frame_window   ',        window_name))
    STATE['show_mask']     =         cv2.getTrackbarPos('show 0=vid 1=mask 2=both', window_name)

    if old_hist != STATE['history'] or old_var != STATE['var_threshold']:
        needs_reset = True


def state_to_config():
    return {
        'history':          STATE['history'],
        'var_threshold':    STATE['var_threshold'],
        'morph_kernel_size': max(3, STATE['morph_kernel'] | 1),
        'min_contour_area': STATE['min_area'],
        'max_contour_area': STATE['max_area'],
        'blur_kernel':      max(3, STATE['blur_kernel'] | 1),
        'use_blur':         bool(STATE['use_blur']),
        'morph_operations': ['open', 'close'] if STATE['morph_close'] else ['open'],
        'bg_method':        'MOG2',
        'detect_shadows':   False,
        # Parámetros de análisis
        'prominence':       float(STATE['prominence']),
        'distance':         STATE['distance'],
        'frame_window':     STATE['frame_window'],
    }


def run_demo(video_dir, initial_config=None, save_path='configs/preprocess_demo.yaml'):
    global needs_reset, bg_sub

    # Cargar videos
    video_dir = Path(video_dir)
    video_files = sorted(
        list(video_dir.glob('**/*.mp4')) +
        list(video_dir.glob('**/*.avi')) +
        list(video_dir.glob('**/*.gif'))
    )
    if not video_files:
        print(f"❌ No hay videos en {video_dir}")
        return

    print(f"🎬 {len(video_files)} videos encontrados")
    print("Controles: ← → cambiar video | SPACE pausa | r reset BG | s guardar | q salir")

    # Cargar config inicial
    if initial_config:
        with open(initial_config) as f:
            cfg = yaml.safe_load(f)
        STATE['history']       = cfg.get('history', 100)
        STATE['var_threshold'] = cfg.get('var_threshold', 25)
        STATE['morph_kernel']  = cfg.get('morph_kernel_size', 3)
        STATE['min_area']      = cfg.get('min_contour_area', 50)
        STATE['max_area']      = cfg.get('max_contour_area', 5000)
        STATE['blur_kernel']   = cfg.get('blur_kernel', 5)
        STATE['use_blur']      = int(cfg.get('use_blur', True))
        STATE['prominence']    = int(cfg.get('prominence', 6))
        STATE['distance']      = cfg.get('distance', 8)
        STATE['frame_window']  = cfg.get('frame_window', 7)
        ops = cfg.get('morph_operations', ['open'])
        STATE['morph_close']   = 1 if 'close' in ops else 0
        print(f"  Config cargada: {initial_config}")

    # Crear ventanas
    cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Preview', 960, 600)

    cv2.namedWindow('Params', cv2.WINDOW_NORMAL)
    # ✅ Fix: mostrar imagen negra como canvas para que los sliders sean visibles
    canvas = np.zeros((50, 800, 3), dtype=np.uint8)
    cv2.putText(canvas, 's=guardar  r=reset BG  SPACE=pausa  a/d=video anterior/siguiente  q=salir',
                (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)
    cv2.imshow('Params', canvas)
    cv2.resizeWindow('Params', 800, 500)  # ✅ más ancho para que quepan los nombres
    create_trackbars('Params')

    # Inicializar sliders con STATE actual
    cv2.setTrackbarPos('history',       'Params', STATE['history'])
    cv2.setTrackbarPos('var_threshold', 'Params', STATE['var_threshold'])
    cv2.setTrackbarPos('morph_kernel',  'Params', STATE['morph_kernel'])
    cv2.setTrackbarPos('min_area',      'Params', STATE['min_area'])
    cv2.setTrackbarPos('max_area',      'Params', STATE['max_area'] // 10)
    cv2.setTrackbarPos('blur_kernel',   'Params', STATE['blur_kernel'])
    cv2.setTrackbarPos('use_blur',      'Params', STATE['use_blur'])
    cv2.setTrackbarPos('morph_close',   'Params', STATE['morph_close'])
    cv2.setTrackbarPos('prominence',    'Params', int(STATE['prominence']))
    cv2.setTrackbarPos('distance',      'Params', STATE['distance'])
    cv2.setTrackbarPos('frame_window',  'Params', STATE['frame_window'])

    video_idx = 0
    paused = False

    while True:
        vf = video_files[video_idx]
        cap = cv2.VideoCapture(str(vf))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        make_bg_sub()
        frame_idx = 0

        print(f"\n▶ [{video_idx+1}/{len(video_files)}] {vf.name}")

        video_done = False
        while not video_done:
            read_trackbars('Params')

            # ✅ Redibujar canvas con valores actuales
            canvas = np.zeros((50, 800, 3), dtype=np.uint8)
            cv2.putText(canvas,
                        's=guardar  r=reset  SPACE=pausa  a/d=prev/next  q=salir',
                        (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)
            cv2.imshow('Params', canvas)

            if not paused:
                ret, frame = cap.read()
                if not ret:
                    video_done = True
                    break
                frame_idx += 1

            fg, valid_c, all_c = process_frame(frame)

            # Visualización según show_mask
            show_mode = STATE['show_mask']
            if show_mode == 0:
                vis = draw_info(frame, valid_c, all_c, vf.stem, frame_idx,
                                total_frames, video_idx, len(video_files))
            elif show_mode == 1:
                vis = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
                # Dibujar contornos válidos en verde sobre la máscara
                for c, area, _ in valid_c:
                    cv2.drawContours(vis, [c], -1, (0, 255, 0), 2)
            else:  # show_mode == 2: side by side
                vis_frame = draw_info(frame, valid_c, all_c, vf.stem, frame_idx,
                                      total_frames, video_idx, len(video_files))
                vis_mask = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
                for c, area, _ in valid_c:
                    cv2.drawContours(vis_mask, [c], -1, (0, 255, 0), 2)
                # Resize mask para que tenga el mismo alto
                h_f = vis_frame.shape[0]
                vis_mask_r = cv2.resize(vis_mask,
                                         (int(vis_mask.shape[1] * h_f / vis_mask.shape[0]), h_f))
                vis = np.hstack([vis_frame, vis_mask_r])

            cv2.imshow('Preview', vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

            elif key == ord('s'):
                cfg = state_to_config()
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as f:
                    yaml.dump(cfg, f, default_flow_style=False)
                print(f"\n💾 Config guardada: {save_path}")
                print(f"   {cfg}")

            elif key == ord('r'):
                needs_reset = True
                frame_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print("↺ BG subtractor reseteado")

            elif key == ord(' '):
                paused = not paused
                print(f"{'⏸ Pausado' if paused else '▶ Reanudado'}")

            elif key == 81 or key == ord('a'):  # ← anterior video
                video_idx = (video_idx - 1) % len(video_files)
                video_done = True

            elif key == 83 or key == ord('d'):  # → siguiente video
                video_idx = (video_idx + 1) % len(video_files)
                video_done = True

        cap.release()
        if not video_done:
            video_idx = (video_idx + 1) % len(video_files)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Demo interactiva de parámetros de preprocesamiento')
    ap.add_argument('--video_dir', required=True,
                    help='Carpeta con videos (subcarpetas OK)')
    ap.add_argument('--config', default=None,
                    help='YAML inicial (para partir de config tuneada)')
    ap.add_argument('--save', default='configs/preprocess_demo.yaml',
                    help='Donde guardar la config al presionar s')
    args = ap.parse_args()

    run_demo(args.video_dir, args.config, args.save)