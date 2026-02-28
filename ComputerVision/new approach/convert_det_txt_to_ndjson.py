#!/usr/bin/env python3
"""
convert_det_txt_to_ndjson.py
Convierte detecciones en formato MOT16 (det.txt) a formato NDJSON
compatible con ocsort_tuner.py y pipeline_track_to_csv.py

Formato MOT16 det.txt:
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, cls, visibility

Formato NDJSON de salida:
    Línea 0: {"meta": {"w": W, "h": H, "fps": FPS, "n_frames": N}}
    Línea i: {"frame": fi_0based, "dets": [[x1,y1,x2,y2,conf], ...]}

Uso:
    # Convertir todas las secuencias en juggling-mot:
    python convert_det_txt_to_ndjson.py --mot_dir datasets/juggling-mot

    # Convertir una secuencia concreta:
    python convert_det_txt_to_ndjson.py \
        --det_txt path/to/det.txt \
        --gt_txt  path/to/gt/gt.txt \
        --output  path/to/det/det.ndjson
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ── Utilidades ─────────────────────────────────────────────────────────────────

def load_mot_txt(path: Path) -> pd.DataFrame:
    """
    Carga un fichero MOT16 (gt.txt o det.txt) sin cabecera.
    Columnas: frame, id, bb_left, bb_top, bb_width, bb_height, conf, cls, visibility
    """
    df = pd.read_csv(
        path, header=None,
        names=["frame", "id", "bb_left", "bb_top",
               "bb_width", "bb_height", "conf", "cls", "visibility"]
    )
    return df


def get_frame_size_from_gt(gt_path: Path):
    """
    Infiere (w, h) a partir del GT:
    w = max(bb_left + bb_width),  h = max(bb_top + bb_height)
    """
    if not gt_path.exists():
        return None, None
    df = load_mot_txt(gt_path)
    w = int((df["bb_left"] + df["bb_width"]).max())   # ← ya estaba bien
    h = int((df["bb_top"]  + df["bb_height"]).max())  # ← ya estaba bien
    return w, h


def convert_det_txt_to_ndjson(det_path: Path, gt_path: Path,
                               output_path: Path,
                               fps: float = 30.0,
                               default_w: int = 1080,
                               default_h: int = 1920):
    """
    Convierte un det.txt MOT16 → det.ndjson compatible con ocsort_tuner.py
    
    Args:
        det_path:   ruta a det.txt con las detecciones
        gt_path:    ruta a gt.txt para inferir dimensiones del frame
        output_path: ruta de salida .ndjson
        fps:        frames por segundo del video
        default_w/h: dimensiones si no se puede inferir del GT
    """
    if not det_path.exists():
        print(f"  SKIP: no existe {det_path}")
        return False

    # Cargar detecciones
    df = load_mot_txt(det_path)

    # Dimensiones del frame
    w, h = get_frame_size_from_gt(gt_path)
    if w is None or w <= 0:
        # Inferir desde las propias detecciones como fallback
        w = int((df["bb_left"] + df["bb_width"]).max())
        h = int((df["bb_top"]  + df["bb_height"]).max())
    if w <= 0 or h <= 0:
        w, h = default_w, default_h

    # Forzar int nativo antes de serializar
    w, h = int(w), int(h)

    # Rango de frames
    frames_sorted = sorted(df["frame"].unique().astype(int))
    n_frames = int(max(frames_sorted)) if frames_sorted else 0  # ← int nativo

    # Agrupar detecciones por frame (0-based internamente)
    frame_dets: dict[int, list] = {}
    for _, row in df.iterrows():
        fi_1based = int(row["frame"])
        fi_0based = fi_1based - 1          # ocsort_tuner usa 0-based

        x1 = float(row["bb_left"])
        y1 = float(row["bb_top"])
        x2 = x1 + float(row["bb_width"])
        y2 = y1 + float(row["bb_height"])
        conf = float(row["conf"]) if row["conf"] != -1 else 1.0

        frame_dets.setdefault(fi_0based, []).append([x1, y1, x2, y2, conf])

    # Escribir NDJSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        # Línea 0: meta
        meta = {
            "meta": {
                "w": w,
                "h": h,
                "fps": fps,
                "n_frames": n_frames
            }
        }
        f.write(json.dumps(meta) + "\n")

        # Una línea por frame (solo frames con detecciones)
        for fi_0based in sorted(frame_dets.keys()):
            entry = {
                "frame": fi_0based,
                "dets": frame_dets[fi_0based]
            }
            f.write(json.dumps(entry) + "\n")

    print(f"  ✓ {det_path.parent.parent.name}  "
          f"frames={n_frames}  dets={sum(len(v) for v in frame_dets.values())}  "
          f"size={w}x{h}  → {output_path}")
    return True


def convert_all_sequences(mot_dir: Path, fps: float = 30.0,
                           splits: list[str] = ("train", "test"),
                           force: bool = False):
    """
    Recorre mot_dir/{split}/{seq}/det/det.txt y genera det.ndjson en el mismo dir.

    Estructura esperada:
        mot_dir/
          train/
            <seq>/
              gt/gt.txt
              det/det.txt   ← entrada
              det/det.ndjson ← salida
          test/
            ...
    """
    converted = skipped = missing = 0

    for split in splits:
        split_dir = mot_dir / split
        if not split_dir.exists():
            print(f"Split no encontrado: {split_dir}")
            continue

        seq_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        print(f"\n{split}: {len(seq_dirs)} secuencias")

        for seq_dir in seq_dirs:
            det_txt  = seq_dir / "det" / "det.txt"
            gt_txt   = seq_dir / "gt"  / "gt.txt"
            out_ndjson = seq_dir / "det" / "det.ndjson"

            if not det_txt.exists():
                print(f"  MISS: {seq_dir.name}/det/det.txt")
                missing += 1
                continue

            if out_ndjson.exists() and not force:
                print(f"  SKIP (ya existe): {seq_dir.name}/det/det.ndjson")
                skipped += 1
                continue

            ok = convert_det_txt_to_ndjson(det_txt, gt_txt, out_ndjson, fps=fps)
            converted += int(ok)

    print(f"\nResumen: {converted} convertidos | {skipped} ya existían | {missing} sin det.txt")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Convierte det.txt MOT16 → det.ndjson para ocsort_tuner.py"
    )
    # Modo batch (toda la carpeta juggling-mot)
    ap.add_argument("--mot_dir", default="datasets/juggling-mot",
                    help="Raíz del dataset MOT (contiene train/ y/o test/)")
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    help="Splits a procesar (default: train test)")
    ap.add_argument("--fps", type=float, default=60.0,
                    help="FPS de los videos (default: 60)")
    ap.add_argument("--force", action="store_true",
                    help="Sobreescribir .ndjson aunque ya existan")

    # Modo single
    ap.add_argument("--det_txt",  default=None,
                    help="[single] Ruta a det.txt concreto")
    ap.add_argument("--gt_txt",   default=None,
                    help="[single] Ruta a gt.txt para inferir dimensiones")
    ap.add_argument("--output",   default=None,
                    help="[single] Ruta de salida .ndjson")

    args = ap.parse_args()

    if args.det_txt:
        # Modo conversión individual
        det_path = Path(args.det_txt)
        gt_path  = Path(args.gt_txt) if args.gt_txt else det_path.parent.parent / "gt" / "gt.txt"
        out_path = Path(args.output) if args.output else det_path.with_suffix(".ndjson")

        print(f"Convirtiendo {det_path} → {out_path}")
        convert_det_txt_to_ndjson(det_path, gt_path, out_path, fps=args.fps)
    else:
        # Modo batch
        mot_dir = Path(args.mot_dir)
        if not mot_dir.exists():
            print(f"ERROR: no existe {mot_dir}")
            return
        print(f"Procesando {mot_dir} ...")
        convert_all_sequences(mot_dir, fps=args.fps,
                               splits=args.splits, force=args.force)

    print("\nDone. Ahora puedes ejecutar:")
    print("  python ocsort_tuner.py --mot_dir datasets/juggling-mot --n_trials 100")


if __name__ == "__main__":
    main()