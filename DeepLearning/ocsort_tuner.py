#!/usr/bin/env python3
"""
ocsort_tuner.py
Tunea parámetros de OC-SORT usando detecciones YOLO precomputadas (.ndjson)
y ground truth en formato MOT Challenge.

Uso:
    python ocsort_tuner.py \
        --mot_dir datasets/juggling-mot \
        --n_trials 100
"""
import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.optimize import linear_sum_assignment

# ── OC-SORT bootstrap ──────────────────────────────────────────────────────────
OCSORT_URLS = {
    "ocsort.py":       "https://raw.githubusercontent.com/noahcao/OC_SORT/master/trackers/ocsort_tracker/ocsort.py",
    "kalmanfilter.py": "https://raw.githubusercontent.com/noahcao/OC_SORT/master/trackers/ocsort_tracker/kalmanfilter.py",
    "association.py":  "https://raw.githubusercontent.com/noahcao/OC_SORT/master/trackers/ocsort_tracker/association.py",
}

def ensure_ocsort(vendor_dir: Path):
    vendor_dir.mkdir(parents=True, exist_ok=True)
    for name, url in OCSORT_URLS.items():
        dst = vendor_dir / name
        if not dst.exists():
            print(f"Descargando OC-SORT: {dst}")
            urllib.request.urlretrieve(url, dst)
        if name == "kalmanfilter.py":
            _patch_kalmanfilter(dst)
        elif name == "ocsort.py":
            _patch_ocsort(dst)
    init = vendor_dir / "__init__.py"
    if not init.exists():
        init.write_text("", encoding="utf-8")

def _patch_kalmanfilter(path: Path):
    text = path.read_text(encoding="utf-8")
    patched = text.replace("r = w / float(h)", "r = float(w) / float(h)")
    if patched != text:
        path.write_text(patched, encoding="utf-8")

def _patch_ocsort(path: Path):
    text = path.read_text(encoding="utf-8")
    patched = text.replace(
        r"observations \Delta t steps away",
        r"observations \\Delta t steps away"
    )
    if patched != text:
        path.write_text(patched, encoding="utf-8")


# ── Carga de datos ─────────────────────────────────────────────────────────────
def load_ndjson(path: Path):
    """Carga detecciones YOLO precomputadas."""
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


def load_mot_gt(gt_path: Path):
    """
    Carga GT MOT Challenge.
    Returns: {frame_1based: {ball_id: (cx, cy)}}
    """
    df = pd.read_csv(gt_path, header=None,
                     names=["frame","id","bb_left","bb_top",
                             "bb_width","bb_height","conf","cls","visibility"])
    result = {}
    for _, row in df.iterrows():
        frame = int(row["frame"])
        bid   = int(row["id"])
        cx = row["bb_left"] + row["bb_width"]  / 2
        cy = row["bb_top"]  + row["bb_height"] / 2
        result.setdefault(frame, {})[bid] = (cx, cy)
    return result


# ── MOTA ───────────────────────────────────────────────────────────────────────
def compute_mota(predictions, ground_truth, distance_threshold=30.0):
    """
    predictions:  {frame_1based: {track_id: (cx, cy)}}
    ground_truth: {frame_1based: {gt_id:    (cx, cy)}}
    """
    total_gt = total_fp = total_miss = total_switch = 0
    gt_to_track = {}

    for frame in sorted(set(list(ground_truth) + list(predictions))):
        gt_objs   = ground_truth.get(frame, {})
        pred_objs = predictions.get(frame, {})

        total_gt += len(gt_objs)

        if not gt_objs:
            total_fp += len(pred_objs)
            continue
        if not pred_objs:
            total_miss += len(gt_objs)
            continue

        gt_ids   = list(gt_objs.keys())
        pred_ids = list(pred_objs.keys())
        cost = np.full((len(gt_ids), len(pred_ids)), distance_threshold * 2)

        for i, gid in enumerate(gt_ids):
            for j, pid in enumerate(pred_ids):
                gx, gy = gt_objs[gid]
                px, py = pred_objs[pid]
                cost[i, j] = np.hypot(gx - px, gy - py)

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_gt = set(); matched_pred = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < distance_threshold:
                matched_gt.add(r); matched_pred.add(c)
                gid = gt_ids[r]; pid = pred_ids[c]
                if gid in gt_to_track and gt_to_track[gid] != pid:
                    total_switch += 1
                gt_to_track[gid] = pid

        total_miss += len(gt_ids)   - len(matched_gt)
        total_fp   += len(pred_ids) - len(matched_pred)

    mota = 1.0 - (total_miss + total_fp + total_switch) / max(total_gt, 1)
    return {"MOTA": mota, "misses": total_miss,
            "FP": total_fp, "switches": total_switch, "total_gt": total_gt}


# ── Post-procesamiento ─────────────────────────────────────────────────────────
def interpolate_tracks(track_points, max_gap=30):
    """Interpolación lineal de gaps en tracks."""
    result = {}
    for tid, points in track_points.items():
        points_sorted = sorted(points, key=lambda p: p[0])
        frames = [p[0] for p in points_sorted]
        xs     = [p[1] for p in points_sorted]
        ys     = [p[2] for p in points_sorted]

        new_points = list(points_sorted)
        for i in range(len(frames) - 1):
            gap = frames[i+1] - frames[i] - 1
            if 0 < gap <= max_gap:
                for j in range(1, gap + 1):
                    alpha = j / (gap + 1)
                    new_points.append((
                        frames[i] + j,
                        xs[i] * (1 - alpha) + xs[i+1] * alpha,
                        ys[i] * (1 - alpha) + ys[i+1] * alpha,
                    ))
        result[tid] = sorted(new_points, key=lambda p: p[0])
    return result


def run_ocsort_on_sequence(ndjson_path: Path, ocsort_module, cfg: dict,
                            interp_gap: int = 0):
    """
    Corre OC-SORT sobre detecciones precomputadas.
    ndjson_path: mot_dir/train|test/<seq>/det/det.ndjson
    Returns: {frame_1based: {track_id: (cx, cy)}}
    """
    meta, frame_dets = load_ndjson(ndjson_path)
    if meta is None:
        return {}

    total_frames = max(frame_dets.keys()) + 1 if frame_dets else 0
    h, w = meta["h"], meta["w"]

    tracker = ocsort_module.OCSort(
        det_thresh    = float(cfg["det_thresh"]),
        max_age       = int(cfg["max_age"]),
        min_hits      = int(cfg["min_hits"]),
        iou_threshold = float(cfg["iou_threshold"]),
        delta_t       = int(cfg["delta_t"]),
        asso_func     = str(cfg["asso_func"]),
        inertia       = float(cfg["inertia"]),
        use_byte      = bool(cfg.get("use_byte", False)),
    )

    track_points = defaultdict(list)   # tid → [(frame_0based, cx, cy)]

    for fi in range(total_frames):
        dets = frame_dets.get(fi, [])
        dets5 = (np.asarray(dets, dtype=np.float32)[:, :5]
                 if dets else np.zeros((0, 5), dtype=np.float32))
        tracks = tracker.update(dets5, (h, w), (h, w))

        if tracks is not None and len(tracks):
            for row in np.asarray(tracks):
                x1, y1, x2, y2 = row[:4]
                tid = int(row[4])
                track_points[tid].append((fi, (x1+x2)/2, (y1+y2)/2))

    if interp_gap > 0:
        track_points = interpolate_tracks(track_points, max_gap=interp_gap)

    predictions = {}
    for tid, points in track_points.items():
        for frame_0, cx, cy in points:
            frame_1 = frame_0 + 1
            predictions.setdefault(frame_1, {})[tid] = (cx, cy)

    return predictions


# ── Evaluar split ──────────────────────────────────────────────────────────────
def evaluate_split(split_dir: Path, ocsort_module, cfg: dict, interp_gap: int,
                   distance_threshold: float = 30.0, verbose: bool = False):
    """
    Evalúa todas las secuencias de un split.
    Estructura esperada:
        split_dir/
          <seq_name>/
            gt/gt.txt
            det/det.ndjson   ← detecciones YOLO precomputadas para esta secuencia
    """
    motas = []
    details = []

    for seq_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
        gt_path     = seq_dir / "gt"  / "gt.txt"
        ndjson_path = seq_dir / "det" / "det.ndjson"

        if not gt_path.exists():
            if verbose:
                print(f"  sin GT: {seq_dir.name}")
            continue
        if not ndjson_path.exists():
            if verbose:
                print(f"  sin det.ndjson: {seq_dir.name}")
            continue

        try:
            gt    = load_mot_gt(gt_path)
            preds = run_ocsort_on_sequence(ndjson_path, ocsort_module, cfg, interp_gap)
            m     = compute_mota(preds, gt, distance_threshold)
            motas.append(m["MOTA"])
            details.append({"seq": seq_dir.name, **m})
            if verbose:
                print(f"  {seq_dir.name}: MOTA={m['MOTA']:.4f} "
                      f"miss={m['misses']} FP={m['FP']} SW={m['switches']}")
        except Exception as e:
            print(f"  WARNING {seq_dir.name}: {e}")
            motas.append(-1.0)

    return motas, details


# ── Optuna objective ───────────────────────────────────────────────────────────
def objective(trial, mot_dir: Path, ocsort_module, distance_threshold: float):

    cfg = {
        "det_thresh":    trial.suggest_float("det_thresh",    0.2,  0.8),
        "iou_threshold": trial.suggest_float("iou_threshold", -0.8, -0.3),
        "asso_func":     trial.suggest_categorical("asso_func",
                             ["giou", "diou", "ciou", "ct_dist"]), #"iou",
        "max_age":       trial.suggest_int("max_age",   120, 120),
        "min_hits":      trial.suggest_int("min_hits",   1,   5),
        "delta_t":       trial.suggest_int("delta_t",    1,   4),
        "inertia":       trial.suggest_float("inertia",  0.55, 0.55),
        "use_byte":      trial.suggest_categorical("use_byte", [False]), #, True]),
    }
    interp_gap = trial.suggest_int("interp_gap", 0, 60)

    train_dir = mot_dir / "train"
    if not train_dir.exists():
        return -1.0

    motas, _ = evaluate_split(train_dir, ocsort_module, cfg, interp_gap,
                               distance_threshold)
    if not motas:
        return -1.0

    avg_mota = float(np.mean(motas))
    trial.set_user_attr("n_sequences", len(motas))
    return avg_mota


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot_dir",            default="datasets/juggling-mot")
    ap.add_argument("--n_trials",           type=int,   default=100)
    ap.add_argument("--output",             default="configs/best_ocsort.yaml")
    ap.add_argument("--distance_threshold", type=float, default=30.0)
    ap.add_argument("--study_name",         default="ocsort_juggling")
    ap.add_argument("--storage",            default=None,
                    help="SQLite para reanudar: sqlite:///ocsort_study.db")
    args = ap.parse_args()

    vendor_dir = Path("ocsort")
    ensure_ocsort(vendor_dir)
    sys.path.insert(0, str(vendor_dir.parent))
    from ocsort import ocsort as ocsort_module

    mot_dir   = Path(args.mot_dir)
    train_dir = mot_dir / "train"
    test_dir  = mot_dir / "test"

    # Contar secuencias disponibles
    if train_dir.exists():
        seqs  = [d for d in train_dir.iterdir() if d.is_dir()]
        avail = sum(1 for s in seqs if (s / "det" / "det.ndjson").exists())
        print(f"Secuencias train: {len(seqs)} | con det.ndjson: {avail}")
    else:
        print(f"ERROR: no existe {train_dir}")
        return

    # ── 1. Tuning sobre train ─────────────────────────────────────────────────
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial, mot_dir, ocsort_module, args.distance_threshold
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best_params     = dict(study.best_trial.params)
    best_interp     = best_params.pop("interp_gap")
    best_cfg        = best_params
    best_train_mota = study.best_value

    print(f"\n{'='*50}")
    print(f"Mejor MOTA (train): {best_train_mota:.4f}")

    # ── 2. Evaluación final sobre test ────────────────────────────────────────
    test_mota = None
    if test_dir.exists():
        print(f"\nEvaluando en TEST con mejores params...")
        test_motas, test_details = evaluate_split(
            test_dir, ocsort_module, best_cfg, best_interp,
            args.distance_threshold, verbose=True)

        if test_motas:
            test_mota = float(np.mean(test_motas))
            print(f"\nMOTA test por secuencia:")
            for d in test_details:
                print(f"  {d['seq']}: MOTA={d['MOTA']:.4f} "
                      f"miss={d['misses']} FP={d['FP']} SW={d['switches']}")
            print(f"\nMOTA (test): {test_mota:.4f}  ← métrica real")
        else:
            print("  No hay secuencias test con det.ndjson disponible")
    else:
        print(f"  No existe {test_dir}, omitiendo evaluación test")

    # ── 3. Guardar config ─────────────────────────────────────────────────────
    import yaml
    out_cfg = {
        **best_cfg,
        "interp_gap":      best_interp,
        "best_mota_train": best_train_mota,
        "best_mota_test":  test_mota,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(out_cfg, f, default_flow_style=False)

    print(f"\nConfig guardada: {args.output}")
    print("Parámetros óptimos:")
    for k, v in out_cfg.items():
        print(f"  {k}: {v}")

    # ── 4. Instrucciones para aplicar a pipeline_track_to_csv ─────────────────
    print(f"\n{'='*50}")
    print(f"Aplicar en pipeline_track_to_csv.py:")
    print(f"  python pipeline_track_to_csv.py \\")
    print(f"    --cache_dir runs/dets_cache_all \\")
    print(f"    --max_merge_gap {best_cfg.get('max_age', 45)} \\")
    print(f"    --interp_gap    {best_interp} \\")
    print(f"    --det_thresh    {best_cfg.get('det_thresh', 0.3):.3f} \\")
    print(f"    --iou_threshold {best_cfg.get('iou_threshold', 0.2):.3f} \\")
    print(f"    --asso_func     {best_cfg.get('asso_func', 'giou')}")


if __name__ == "__main__":
    main()