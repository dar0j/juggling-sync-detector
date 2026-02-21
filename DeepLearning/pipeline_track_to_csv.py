#!/usr/bin/env python3
"""
pipeline_track_to_csv.py
Lee caches NDJSON de detecciones, ejecuta OC-SORT, repara tracks fragmentados,
y genera CSVs con coordenadas consistentes (x_ball1,y_ball1,...,x_ballN,y_ballN).

Uso:
  python pipeline_track_to_csv.py \
    --cache_dir runs/dets_cache_all \
    --video_root "../Datasets/used to track" \
    --out_dir runs/track_csvs \
    --det_thresh 0.57 --iou_threshold -0.57 --max_age 90 \
    --min_hits 3 --delta_t 1 --inertia 0.5 --asso_func giou \
    --visualize
"""
import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ── OC-SORT bootstrap ──────────────────────────────────────────────────────────
OCSORT_URLS = {
    "ocsort.py": "https://raw.githubusercontent.com/noahcao/OC_SORT/master/trackers/ocsort_tracker/ocsort.py",
    "kalmanfilter.py": "https://raw.githubusercontent.com/noahcao/OC_SORT/master/trackers/ocsort_tracker/kalmanfilter.py",
    "association.py": "https://raw.githubusercontent.com/noahcao/OC_SORT/master/trackers/ocsort_tracker/association.py",
}

def ensure_ocsort(vendor_dir: Path):
    vendor_dir.mkdir(parents=True, exist_ok=True)
    for name, url in OCSORT_URLS.items():
        dst = vendor_dir / name
        if not dst.exists():
            print(f"Descargando OC-SORT: {dst}")
            urllib.request.urlretrieve(url, dst)
    init = vendor_dir / "__init__.py"
    if not init.exists():
        init.write_text("", encoding="utf-8")


# ── Utilidades ──────────────────────────────────────────────────────────────────
def load_cache(path: Path):
    meta = None
    frames = {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            if i == 0 and "meta" in obj:
                meta = obj["meta"]
                continue
            frames[int(obj["frame"])] = obj["dets"]
    return meta, frames


def parse_filename(fname: str):
    """Extrae n_balls y trickname de '3_(0,6)_2'."""
    tokens = fname.split("_")
    if len(tokens) < 2:
        return None, None, None
    try:
        n_balls = int(tokens[0])
    except ValueError:
        return None, None, None
    if tokens[-1].isdigit() and len(tokens) > 2:
        trick = "_".join(tokens[1:-1])
        sample_id = tokens[-1]
    else:
        trick = "_".join(tokens[1:])
        sample_id = "0"
    return n_balls, trick, sample_id


def id_color(tid: int):
    rng = np.random.default_rng(int(tid) * 9973)
    return tuple(int(x) for x in rng.integers(60, 255, size=3))


# ── Reparación de tracks fragmentados ──────────────────────────────────────────
def repair_fragmented_tracks(track_data: dict, n_balls: int, max_gap: int = 45,
                              max_merge_dist: float = 120.0):
    """
    Fusiona tracks fragmentados que pertenecen probablemente a la misma pelota.
    
    track_data: {track_id: [(frame, cx, cy), ...], ...}
    n_balls:    número real de pelotas
    max_gap:    máximo gap temporal (frames) para considerar fusión
    max_merge_dist: distancia máxima en píxeles entre fin de un track e inicio del otro
    
    Returns: merged track_data con IDs consolidados
    """
    if len(track_data) <= n_balls:
        return track_data

    # Calcular inicio/fin y posición de cada track
    track_info = {}
    for tid, points in track_data.items():
        frames = [p[0] for p in points]
        track_info[tid] = {
            "start": min(frames),
            "end": max(frames),
            "len": len(points),
            "start_pos": next(p for p in points if p[0] == min(frames)),
            "end_pos": next(p for p in points if p[0] == max(frames)),
        }

    # Ordenar por frame de inicio
    sorted_ids = sorted(track_info.keys(), key=lambda t: track_info[t]["start"])

    # Greedy merge: para cada track corto que empieza después de otro que terminó
    merged = {}  # old_id -> new_id
    active_tracks = []  # lista de IDs de tracks "vivos" (no fusionados)

    for tid in sorted_ids:
        info = track_info[tid]
        best_merge = None
        best_score = float("inf")

        for atid in active_tracks:
            if atid in merged:
                continue
            a_info = track_info[atid]

            # ¿El track activo terminó antes de que empiece este?
            gap = info["start"] - a_info["end"]
            if gap < 0 or gap > max_gap:
                continue

            # Distancia entre última posición del activo y primera del nuevo
            ex, ey = a_info["end_pos"][1], a_info["end_pos"][2]
            sx, sy = info["start_pos"][1], info["start_pos"][2]
            dist = np.sqrt((ex - sx)**2 + (ey - sy)**2)

            if dist < max_merge_dist:
                score = dist + gap * 2  # penalizar gap también
                if score < best_score:
                    best_score = score
                    best_merge = atid

        if best_merge is not None:
            # Fusionar: este track se absorbe en best_merge
            merged[tid] = best_merge
            # Extender el rango del track activo
            track_info[best_merge]["end"] = info["end"]
            track_info[best_merge]["end_pos"] = info["end_pos"]
            track_info[best_merge]["len"] += info["len"]
        else:
            active_tracks.append(tid)

    # Construir resultado fusionado
    result = defaultdict(list)
    for tid, points in track_data.items():
        target_id = merged.get(tid, tid)
        # Resolver cadenas de merges
        while target_id in merged:
            target_id = merged[target_id]
        result[target_id].extend(points)

    # Ordenar puntos por frame
    for tid in result:
        result[tid].sort(key=lambda p: p[0])

    return dict(result)


def select_top_tracks(track_data: dict, n_balls: int):
    """Selecciona los N tracks más largos (más presentes)."""
    if len(track_data) <= n_balls:
        return track_data
    ranked = sorted(track_data.items(), key=lambda kv: len(kv[1]), reverse=True)
    return dict(ranked[:n_balls])


def tracks_to_csv_array(track_data: dict, n_balls: int, total_frames: int):
    """
    Convierte tracks a array (total_frames, n_balls*2) con -1 para frames sin detección.
    Columnas: x_ball1, y_ball1, x_ball2, y_ball2, ...
    """
    # Asignar IDs 0..n_balls-1 por orden de aparición (frame de inicio)
    sorted_tids = sorted(track_data.keys(),
                         key=lambda t: min(p[0] for p in track_data[t]))[:n_balls]

    arr = np.full((total_frames, n_balls * 2), -1.0, dtype=np.float32)

    for col_idx, tid in enumerate(sorted_tids):
        for frame, cx, cy in track_data[tid]:
            if 0 <= frame < total_frames:
                arr[frame, col_idx * 2] = cx
                arr[frame, col_idx * 2 + 1] = cy

    return arr


def interpolate_gaps(arr: np.ndarray, max_gap: int = 30):
    """Interpola linealmente gaps cortos (<=max_gap frames) en cada columna."""
    result = arr.copy()
    for col in range(result.shape[1]):
        series = result[:, col]
        valid = series != -1.0
        if valid.sum() < 2:
            continue

        # Encontrar gaps
        valid_indices = np.where(valid)[0]
        for i in range(len(valid_indices) - 1):
            start = valid_indices[i]
            end = valid_indices[i + 1]
            gap_len = end - start - 1
            if 0 < gap_len <= max_gap:
                # Interpolación lineal
                for j in range(1, gap_len + 1):
                    alpha = j / (gap_len + 1)
                    result[start + j, col] = (
                        series[start] * (1 - alpha) + series[end] * alpha
                    )

    return result


# ── Pipeline principal ──────────────────────────────────────────────────────────
def process_one_video(cache_path: Path, video_path: Path, n_balls: int,
                       ocsort_module, cfg: dict,
                       max_merge_gap: int = 45, max_merge_dist: float = 120.0,
                       interp_gap: int = 30,
                       repair: bool = True,                # ← nuevo
                       visualize: bool = False, vis_out_dir: Path = None):
    """
    Procesa un video: tracking + reparación + CSV.
    Returns: np.ndarray (n_frames, n_balls*2) o None si falla
    """
    meta, frame_dets = load_cache(cache_path)
    if meta is None:
        print(f"  WARNING: cache sin meta: {cache_path}")
        return None

    total_frames = max(frame_dets.keys()) + 1 if frame_dets else 0
    if total_frames == 0:
        return None

    h, w = meta["h"], meta["w"]

    # Crear tracker
    tracker = ocsort_module.OCSort(
        det_thresh=float(cfg["det_thresh"]),
        max_age=int(cfg["max_age"]),
        min_hits=int(cfg["min_hits"]),
        iou_threshold=float(cfg["iou_threshold"]),
        delta_t=int(cfg["delta_t"]),
        asso_func=str(cfg["asso_func"]),
        inertia=float(cfg["inertia"]),
        use_byte=bool(cfg.get("use_byte", False)),
    )

    # Ejecutar tracking
    all_track_points = defaultdict(list)  # tid -> [(frame, cx, cy), ...]

    for fi in range(total_frames):
        dets = frame_dets.get(fi, [])
        if len(dets):
            dets_np = np.asarray(dets, dtype=np.float32)
            dets5 = dets_np[:, :5]
        else:
            dets5 = np.zeros((0, 5), dtype=np.float32)

        tracks = tracker.update(dets5, (h, w), (h, w))
        if tracks is not None and len(tracks) > 0:
            for row in np.asarray(tracks):
                x1, y1, x2, y2 = row[:4]
                tid = int(row[4])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                all_track_points[tid].append((fi, cx, cy))

    n_raw_tracks = len(all_track_points)

    # Reparar fragmentación (opcional)
    if repair:
        repaired = repair_fragmented_tracks(
            all_track_points, n_balls,
            max_gap=max_merge_gap, max_merge_dist=max_merge_dist
        )
        n_repaired_tracks = len(repaired)
    else:
        repaired = all_track_points
        n_repaired_tracks = n_raw_tracks  # ← sin reparación, mismo número

    selected = select_top_tracks(repaired, n_balls)

    if repair:
        print(f"    Tracks: {n_raw_tracks} raw -> {n_repaired_tracks} repaired -> {len(selected)} selected")
    else:
        print(f"    Tracks: {n_raw_tracks} raw (repair OFF) -> {len(selected)} selected")

    # Convertir a array
    arr = tracks_to_csv_array(selected, n_balls, total_frames)

    # Interpolar gaps
    arr = interpolate_gaps(arr, max_gap=interp_gap)

    print(f"    Tracks: {n_raw_tracks} raw -> {n_repaired_tracks} repaired -> {len(selected)} selected")

    # Visualización
    if visualize and video_path.exists():
        _visualize_result(video_path, arr, n_balls, vis_out_dir, selected, total_frames)

    return arr


def _visualize_result(video_path, arr, n_balls, vis_out_dir, selected_tracks, total_frames):
    """Genera video con tracks superpuestos para verificar calidad."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if vis_out_dir:
        vis_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = vis_out_dir / f"{video_path.stem}_tracked.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    else:
        writer = None

    COLORS = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 255, 0), (255, 128, 0), (0, 128, 255),
    ]
    trails = [deque(maxlen=20) for _ in range(n_balls)]

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi >= total_frames:
            break

        for bi in range(n_balls):
            x = arr[fi, bi * 2]
            y = arr[fi, bi * 2 + 1]
            color = COLORS[bi % len(COLORS)]

            if x != -1.0 and y != -1.0:
                pos = (int(x), int(y))
                cv2.circle(frame, pos, 8, color, 2)
                cv2.putText(frame, f"B{bi+1}", (pos[0]+12, pos[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                trails[bi].append(pos)
            else:
                trails[bi].append(None)

            # Dibujar trail
            pts = [p for p in trails[bi] if p is not None]
            for j in range(1, len(pts)):
                cv2.line(frame, pts[j-1], pts[j], color, 2)

        info = f"Frame {fi}/{total_frames} | Balls: {n_balls}"
        cv2.putText(frame, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if writer:
            writer.write(frame)

        fi += 1

    cap.release()
    if writer:
        writer.release()
        print(f"    Visualización: {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="runs/dets_cache_all",
                    help="Carpeta con archivos .ndjson de detecciones")
    ap.add_argument("--video_root", default="../Datasets/used to track",
                    help="Raíz de carpetas 'already *' con videos")
    ap.add_argument("--out_dir", default="runs/track_csvs_fragment_repair",
                    help="Carpeta de salida para CSVs")
    # OC-SORT params (tus valores optimizados)
    ap.add_argument("--det_thresh", type=float, default=0.57)
    ap.add_argument("--iou_threshold", type=float, default=-0.57)
    ap.add_argument("--asso_func", default="giou")
    ap.add_argument("--max_age", type=int, default=90)
    ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--delta_t", type=int, default=1)
    ap.add_argument("--inertia", type=float, default=0.5)
    ap.add_argument("--use_byte", action="store_true")
    # Repair params
    ap.add_argument("--max_merge_gap", type=int, default=45,
                    help="Max frames gap para fusionar tracks fragmentados")
    ap.add_argument("--max_merge_dist", type=float, default=120.0,
                    help="Max distancia (px) para fusionar tracks fragmentados")
    ap.add_argument("--interp_gap", type=int, default=30,
                    help="Max gap (frames) para interpolar linealmente")
    # Visual/skip
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--no_repair", action="store_true",
                    help="Desactivar reparación de tracks fragmentados")
    args = ap.parse_args()

    # Asegurar código OC-SORT
    vendor_dir = Path("ocsort")
    ensure_ocsort(vendor_dir)
    from ocsort import ocsort as ocsort_module

    cache_dir = Path(args.cache_dir)
    video_root = Path(args.video_root)
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "visualizations" if args.visualize else None

    cfg = {
        "det_thresh": args.det_thresh,
        "iou_threshold": args.iou_threshold,
        "asso_func": args.asso_func,
        "max_age": args.max_age,
        "min_hits": args.min_hits,
        "delta_t": args.delta_t,
        "inertia": args.inertia,
        "use_byte": args.use_byte,
    }

    # Crear subcarpetas por n_balls
    for nb in [3, 4, 5, 6]:
        (out_dir / f"{nb}b").mkdir(parents=True, exist_ok=True)

    # Buscar todos los caches
    caches = sorted(cache_dir.glob("*.ndjson"))
    print(f"Encontrados {len(caches)} caches de detecciones")

    # Mapear nombres de video a rutas reales
    video_map = {}
    for folder in video_root.iterdir():
        if folder.is_dir() and folder.name.startswith("already"):
            for vf in folder.rglob("*"):
                if vf.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                    video_map[vf.stem] = vf

    stats = {"ok": 0, "skip": 0, "fail": 0}

    for ci, cache_path in enumerate(caches, 1):
        stem = cache_path.stem
        n_balls, trick, sample_id = parse_filename(stem)
        if n_balls is None:
            print(f"[{ci}/{len(caches)}] SKIP (nombre no parseable): {stem}")
            stats["skip"] += 1
            continue

        csv_out = out_dir / f"{n_balls}b" / f"{stem}.csv"
        if args.skip_existing and csv_out.exists():
            print(f"[{ci}/{len(caches)}] SKIP (existe): {stem}")
            stats["skip"] += 1
            continue

        video_path = video_map.get(stem)
        if video_path is None:
            # Buscar por nombre parcial
            for vname, vpath in video_map.items():
                if stem in vname or vname in stem:
                    video_path = vpath
                    break

        print(f"[{ci}/{len(caches)}] {stem} ({n_balls}b, trick={trick})")

        arr = process_one_video(
            cache_path=cache_path,
            video_path=video_path if video_path else Path(stem),
            n_balls=n_balls,
            ocsort_module=ocsort_module,
            cfg=cfg,
            max_merge_gap=args.max_merge_gap,
            max_merge_dist=args.max_merge_dist,
            interp_gap=args.interp_gap,
            repair=not args.no_repair,          # ← nuevo
            visualize=args.visualize and video_path is not None,
            vis_out_dir=vis_dir,
        )

        if arr is None:
            print(f"    FAIL: no se pudo procesar")
            stats["fail"] += 1
            continue

        # Guardar CSV (sin header, solo coordenadas)
        pd.DataFrame(arr).to_csv(csv_out, header=False, index=False)
        print(f"    OK: {csv_out} ({arr.shape[0]} frames, {arr.shape[1]//2} balls)")
        stats["ok"] += 1

    print(f"\n=== Resumen ===")
    print(f"OK: {stats['ok']} | Skip: {stats['skip']} | Fail: {stats['fail']}")
    print(f"CSVs en: {out_dir}")


if __name__ == "__main__":
    main()