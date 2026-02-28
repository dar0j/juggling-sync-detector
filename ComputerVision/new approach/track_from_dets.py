#!/usr/bin/env python3
"""
track_from_dets.py
Lee detecciones pre-computadas en formato MOT Challenge (det.txt),
las pasa por OC-SORT, repara tracks fragmentados e interpola gaps.
Genera CSVs en formato autocolortrack para nohandlebars.py.

NO necesita el video original — solo el det.txt.

Uso:
    python track_from_dets.py \
        --det_dir datasets/detections_mot \
        --out_dir track_csvs3 \
        --det_thresh 0.57 --iou_threshold -0.57 --max_age 90 \
        --min_hits 3 --delta_t 1 --inertia 0.5 --asso_func giou \
        --max_merge_gap 45 --max_merge_dist 120 --interp_gap 30
"""
import argparse
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ── OC-SORT bootstrap (igual que pipeline_track_to_csv.py) ────────────────────
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


# ── Parseo de nombre de secuencia ──────────────────────────────────────────────
def parse_seq_name(name: str):
    """Extrae n_balls del nombre '3_(0,6)_2'."""
    tokens = name.split("_")
    try:
        return int(tokens[0])
    except (ValueError, IndexError):
        return None


# ── Carga det.txt ──────────────────────────────────────────────────────────────
def load_det_txt(det_path: Path):
    """
    Carga det.txt MOT Challenge.
    Formato: frame,-1,x,y,w,h,conf,-1,-1,-1
    Returns: dict {frame_idx: [[x1,y1,x2,y2,conf], ...]}
    """
    rows = []
    with det_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6])
            rows.append((frame, x, y, w, h, conf))

    if not rows:
        return {}, 0, 0, 0

    frames_data = defaultdict(list)
    max_frame = 0
    max_x2 = 0
    max_y2 = 0
    for frame, x, y, w, h, conf in rows:
        x1, y1, x2, y2 = x, y, x + w, y + h
        frames_data[frame].append([x1, y1, x2, y2, conf])
        max_frame = max(max_frame, frame)
        max_x2 = max(max_x2, x2)
        max_y2 = max(max_y2, y2)

    return frames_data, max_frame, int(max_x2 + 50), int(max_y2 + 50)


# ── Reparación de tracks fragmentados (de pipeline_track_to_csv.py) ───────────
def repair_fragmented_tracks(track_data: dict, n_balls: int,
                              max_gap: int = 45, max_merge_dist: float = 120.0):
    if len(track_data) <= n_balls:
        return track_data

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

    sorted_ids = sorted(track_info.keys(), key=lambda t: track_info[t]["start"])
    merged = {}
    active_tracks = []

    for tid in sorted_ids:
        info = track_info[tid]
        best_merge = None
        best_score = float("inf")

        for atid in active_tracks:
            if atid in merged:
                continue
            a_info = track_info[atid]
            gap = info["start"] - a_info["end"]
            if gap < 0 or gap > max_gap:
                continue
            ex, ey = a_info["end_pos"][1], a_info["end_pos"][2]
            sx, sy = info["start_pos"][1], info["start_pos"][2]
            dist = np.sqrt((ex - sx)**2 + (ey - sy)**2)
            if dist < max_merge_dist:
                score = dist + gap * 2
                if score < best_score:
                    best_score = score
                    best_merge = atid

        if best_merge is not None:
            merged[tid] = best_merge
            track_info[best_merge]["end"] = info["end"]
            track_info[best_merge]["end_pos"] = info["end_pos"]
            track_info[best_merge]["len"] += info["len"]
        else:
            active_tracks.append(tid)

    result = defaultdict(list)
    for tid, points in track_data.items():
        target_id = merged.get(tid, tid)
        while target_id in merged:
            target_id = merged[target_id]
        result[target_id].extend(points)

    for tid in result:
        result[tid].sort(key=lambda p: p[0])

    return dict(result)


def select_top_tracks(track_data: dict, n_balls: int):
    if len(track_data) <= n_balls:
        return track_data
    ranked = sorted(track_data.items(), key=lambda kv: len(kv[1]), reverse=True)
    return dict(ranked[:n_balls])


def tracks_to_csv_array(track_data: dict, n_balls: int, total_frames: int):
    sorted_tids = sorted(track_data.keys(),
                         key=lambda t: min(p[0] for p in track_data[t]))[:n_balls]
    arr = np.full((total_frames, n_balls * 2), -1.0, dtype=np.float32)
    for col_idx, tid in enumerate(sorted_tids):
        for frame, cx, cy in track_data[tid]:
            fi = frame - 1  # MOT es 1-based, array es 0-based
            if 0 <= fi < total_frames:
                arr[fi, col_idx * 2] = cx
                arr[fi, col_idx * 2 + 1] = cy
    return arr


def interpolate_gaps(arr: np.ndarray, max_gap: int = 30):
    result = arr.copy()
    for col in range(result.shape[1]):
        series = result[:, col]
        valid = series != -1.0
        if valid.sum() < 2:
            continue
        valid_indices = np.where(valid)[0]
        for i in range(len(valid_indices) - 1):
            start = valid_indices[i]
            end = valid_indices[i + 1]
            gap_len = end - start - 1
            if 0 < gap_len <= max_gap:
                for j in range(1, gap_len + 1):
                    alpha = j / (gap_len + 1)
                    result[start + j, col] = (
                        series[start] * (1 - alpha) + series[end] * alpha
                    )
    return result


# ── Pipeline principal ────────────────────────────────────────────────────────
def process_sequence(det_path: Path, n_balls: int, ocsort_module,
                     cfg: dict, max_merge_gap: int, max_merge_dist: float,
                     interp_gap: int, repair: bool):
    """
    Procesa una secuencia:
      1. Carga det.txt
      2. Corre OC-SORT frame a frame (sin video, solo bboxes)
      3. Repara tracks fragmentados
      4. Interpola gaps
    Returns: np.ndarray (total_frames, n_balls*2) con -1 para frames vacíos
    """
    frames_data, max_frame, frame_w, frame_h = load_det_txt(det_path)
    if max_frame == 0:
        print(f"  SKIP: det.txt vacío")
        return None

    total_frames = max_frame  # 1-based → total_frames filas

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

    all_track_points = defaultdict(list)

    for fi in range(1, max_frame + 1):
        dets = frames_data.get(fi, [])
        if dets:
            dets_np = np.array(dets, dtype=np.float32)  # (N, 5): x1,y1,x2,y2,conf
        else:
            dets_np = np.zeros((0, 5), dtype=np.float32)

        # OcSort espera (h, w) del frame — usamos las dimensiones estimadas del det.txt
        tracks = tracker.update(dets_np, (frame_h, frame_w), (frame_h, frame_w))

        if tracks is not None and len(tracks) > 0:
            for row in np.asarray(tracks):
                x1, y1, x2, y2 = row[:4]
                tid = int(row[4])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                all_track_points[tid].append((fi, cx, cy))

    n_raw = len(all_track_points)

    if repair:
        repaired = repair_fragmented_tracks(
            all_track_points, n_balls,
            max_gap=max_merge_gap, max_merge_dist=max_merge_dist
        )
        n_repaired = len(repaired)
    else:
        repaired = all_track_points
        n_repaired = n_raw

    selected = select_top_tracks(repaired, n_balls)
    print(f"    Tracks: {n_raw} raw → {n_repaired} repaired → {len(selected)} selected  "
          f"(frames: {total_frames})")

    arr = tracks_to_csv_array(selected, n_balls, total_frames)
    arr = interpolate_gaps(arr, max_gap=interp_gap)
    return arr


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Track juggling balls from pre-computed MOT detections")
    ap.add_argument("--det_dir", default="datasets/detections_mot",
                    help="Carpeta raíz con subcarpetas <seq_name>/det/det.txt")
    ap.add_argument("--out_dir", default="runs/track_csvs",
                    help="Carpeta de salida para CSVs autocolortrack")
    # OC-SORT params
    ap.add_argument("--det_thresh",    type=float, default=0.57)
    ap.add_argument("--iou_threshold", type=float, default=-0.57)
    ap.add_argument("--asso_func",     default="giou")
    ap.add_argument("--max_age",       type=int,   default=90)
    ap.add_argument("--min_hits",      type=int,   default=3)
    ap.add_argument("--delta_t",       type=int,   default=1)
    ap.add_argument("--inertia",       type=float, default=0.5)
    ap.add_argument("--use_byte",      action="store_true")
    # Repair / interpolation
    ap.add_argument("--max_merge_gap",  type=int,   default=45)
    ap.add_argument("--max_merge_dist", type=float, default=120.0)
    ap.add_argument("--interp_gap",     type=int,   default=30)
    ap.add_argument("--no_repair",      action="store_true")
    ap.add_argument("--skip_existing",  action="store_true")
    args = ap.parse_args()

    # Asegurar OC-SORT
    vendor_dir = Path(__file__).parent / "ocsort"
    ensure_ocsort(vendor_dir)
    import sys
    sys.path.insert(0, str(vendor_dir.parent))
    from ocsort import ocsort as ocsort_module

    cfg = {
        "det_thresh":    args.det_thresh,
        "iou_threshold": args.iou_threshold,
        "asso_func":     args.asso_func,
        "max_age":       args.max_age,
        "min_hits":      args.min_hits,
        "delta_t":       args.delta_t,
        "inertia":       args.inertia,
        "use_byte":      args.use_byte,
    }

    det_dir = Path(args.det_dir)
    out_dir = Path(args.out_dir)

    # Crear subcarpetas por n_balls
    for nb in [3, 4, 5, 6]:
        (out_dir / f"{nb}b").mkdir(parents=True, exist_ok=True)

    sequences = sorted([d for d in det_dir.iterdir() if d.is_dir()])
    print(f"Secuencias encontradas: {len(sequences)}")

    stats = {"ok": 0, "skip": 0, "fail": 0}

    for ci, seq_dir in enumerate(sequences, 1):
        det_path = seq_dir / "det" / "det.txt"
        if not det_path.exists():
            print(f"[{ci}/{len(sequences)}] SKIP (sin det.txt): {seq_dir.name}")
            stats["skip"] += 1
            continue

        n_balls = parse_seq_name(seq_dir.name)
        if n_balls is None:
            print(f"[{ci}/{len(sequences)}] SKIP (nombre no parseable): {seq_dir.name}")
            stats["skip"] += 1
            continue

        csv_out = out_dir / f"{n_balls}b" / f"{seq_dir.name}.csv"
        if args.skip_existing and csv_out.exists():
            print(f"[{ci}/{len(sequences)}] SKIP (existe): {seq_dir.name}")
            stats["skip"] += 1
            continue

        print(f"[{ci}/{len(sequences)}] {seq_dir.name}  ({n_balls} pelotas)")

        arr = process_sequence(
            det_path=det_path,
            n_balls=n_balls,
            ocsort_module=ocsort_module,
            cfg=cfg,
            max_merge_gap=args.max_merge_gap,
            max_merge_dist=args.max_merge_dist,
            interp_gap=args.interp_gap,
            repair=not args.no_repair,
        )

        if arr is None:
            print(f"    FAIL")
            stats["fail"] += 1
            continue

        pd.DataFrame(arr).to_csv(csv_out, header=False, index=False)
        print(f"    → {csv_out}  ({arr.shape[0]} frames)")
        stats["ok"] += 1

    print(f"\n=== Resumen ===")
    print(f"OK: {stats['ok']} | Skip: {stats['skip']} | Fail: {stats['fail']}")
    print(f"CSVs en: {out_dir}/")


if __name__ == "__main__":
    main()
