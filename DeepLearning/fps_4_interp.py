import cv2
import json
import subprocess
import os
from pathlib import Path

def get_real_fps(video_path):
    """
    Obtiene FPS reales usando ffprobe (más confiable que cv2 + time).
    Fallback a CAP_PROP_FPS si ffprobe no está disponible.
    """
    # Método 1: ffprobe (el más preciso, lee el timebase del contenedor)
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,avg_frame_rate",
            "-of", "json", str(video_path)
        ], capture_output=True, text=True, timeout=10)

        info = json.loads(result.stdout)
        stream = info["streams"][0]

        # avg_frame_rate es más confiable que r_frame_rate para VFR
        avg = stream.get("avg_frame_rate", "0/1")
        num, den = map(int, avg.split("/"))
        if den > 0 and num > 0:
            return round(num / den, 2), "ffprobe"
    except Exception:
        pass

    # Método 2: OpenCV metadata (sin medir tiempo)
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0:
            return round(fps, 2), "cv2_meta"

    return 30.0, "default"


def update_ndjson_metadata(ndjson_path, real_fps):
    if not ndjson_path.exists():
        return False

    lines = ndjson_path.read_text().splitlines()
    if not lines:
        return False

    try:
        data = json.loads(lines[0])
        if "meta" in data:
            data["meta"]["real_fps"] = real_fps
            lines[0] = json.dumps(data)
            ndjson_path.write_text("\n".join(lines) + "\n")
            return True
    except Exception as e:
        print(f"Error actualizando {ndjson_path.name}: {e}")
    return False


def main():
    video_root = Path("../Datasets/used to track")
    cache_dir = Path("runs/dets_cache_all")

    videos = []
    for folder in video_root.iterdir():
        if folder.is_dir() and folder.name.startswith("already"):
            videos.extend(list(folder.rglob("*.mp4")) + list(folder.rglob("*.avi")))

    print(f"Procesando {len(videos)} videos...")

    fps_groups = {}  # para ver distribución al final

    for v_path in videos:
        ndjson_path = cache_dir / f"{v_path.stem}.ndjson"
        real_fps, method = get_real_fps(v_path)

        fps_groups.setdefault(real_fps, []).append(v_path.name)
        status = "UPDATED" if update_ndjson_metadata(ndjson_path, real_fps) else "NO CACHE"
        print(f"[{status}] {v_path.name}: {real_fps} fps (via {method})")

    # Resumen de distribución
    print(f"\n{'='*50}")
    print("Distribución de FPS en el dataset:")
    for fps_val, names in sorted(fps_groups.items()):
        print(f"  {fps_val:6.2f} fps → {len(names):3d} videos")


if __name__ == "__main__":
    main()