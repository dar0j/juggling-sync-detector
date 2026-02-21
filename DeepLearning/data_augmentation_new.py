#!/usr/bin/env python3
"""
data_augmentation_new.py
Data augmentation para CSVs de tracking (solo pelotas, sin manos).
Adaptado para la nueva estructura de datos.

Uso:
  python data_augmentation_new.py --data_root runs/track_csvs
"""
import argparse
import os
import re
import glob

import numpy as np
import pandas as pd


# Trucos simétricos (no necesitan flip)
SYMMETRIC_TRICKS = {
    "(4,2x)(2x,4)", "(4,4)", "(4x,4x)", "(6,6)(2x,2x)",
    "(6x,2x)(2x,6x)", "(6x,0)(6,6)(6,6)(0,6x)(6,6)(6,6)",
    "(6x,2)(6,6)(2,6x)(6,6)", "(6x,4)(4,6x)",
    "(6x,6x)(6x,2)(6x,6x)(2,6x)", "(8x,2)(2,8x)",
    "(6,6)", "(6x,6x)",
}

# Trucos sin par real en el dataset: no generar su flip
NO_FLIP_TRICKS = {
    "(8x,2x)",  # su flip sería (2x,8x) que no existe en el dataset
}


def parse_filename(fname):
    tokens = fname.split("_")
    if len(tokens) < 2:
        return None, None, None
    try:
        nb = int(tokens[0])
    except ValueError:
        return None, None, None
    if tokens[-1].isdigit() and len(tokens) > 2:
        trick = "_".join(tokens[1:-1])
        sid = tokens[-1]
    else:
        trick = "_".join(tokens[1:])
        sid = "0"
    return nb, trick, sid


def generate_mirror_name(trick_name):
    """
    Genera nombre de truco espejado: (a,b) -> (b,a) para cada par.
    '(0,6)' -> '(6,0)', '(4,2x)(2x,4)' -> '(2x,4)(4,2x)'
    """
    coords = re.findall(r'\(([^,]+),([^)]+)\)', trick_name)
    if not coords:
        return None
    mirrored = "".join(f"({b},{a})" for a, b in coords)
    return mirrored


def flip_horizontal(data, image_width=None):
    """
    Flip horizontal: invierte coordenadas x.
    Para CSVs sin manos: columnas son x_b1, y_b1, x_b2, y_b2, ...
    """
    flipped = data.copy()
    # Columnas x son índices pares: 0, 2, 4, ...
    x_cols = list(range(0, data.shape[1], 2))

    if image_width is None:
        # Estimar ancho desde los datos
        all_x = data[:, x_cols]
        valid = all_x[all_x != -1.0]
        if valid.size == 0:
            return flipped
        image_width = valid.max() + valid.min()  # asume centrado

    for col in x_cols:
        mask = flipped[:, col] != -1.0
        flipped[mask, col] = image_width - data[mask, col]

    return flipped


def shuffle_ball_ids(data, n_balls, seed=42):
    """
    Permuta aleatoriamente las identidades de pelotas en cada frame.
    """
    rng = np.random.RandomState(seed)
    shuffled = data.copy()
    for t in range(shuffled.shape[0]):
        balls = shuffled[t].reshape(n_balls, 2)
        rng.shuffle(balls)
        shuffled[t] = balls.flatten()
    return shuffled


def add_gaussian_noise(data, std=2.0, seed=42):
    """Añade ruido gaussiano a coordenadas válidas."""
    rng = np.random.RandomState(seed)
    noisy = data.copy()
    valid = noisy != -1.0
    noise = rng.randn(*noisy.shape).astype(np.float32) * std
    noisy[valid] += noise[valid]
    return noisy


def temporal_crop(data, crop_ratio=0.8, seed=42):
    """Recorta un fragmento temporal aleatorio."""
    rng = np.random.RandomState(seed)
    n_frames = data.shape[0]
    crop_len = max(1, int(n_frames * crop_ratio))
    start = rng.randint(0, n_frames - crop_len + 1)
    return data[start:start + crop_len].copy()


def augment_dataset(data_root: str, augmentations: list = None, out_root: str = None):
    """
    augmentations válidas: flip, noise, crop
    shuffle eliminado: incompatible con cálculo de velocidades en training
    """
    if augmentations is None:
        augmentations = ["flip", "crop"]

    if "shuffle" in augmentations:
        print("WARNING: shuffle ignorado (destruye velocidades al calcular np.gradient)")
        augmentations = [a for a in augmentations if a != "shuffle"]

    # Si no se especifica out_root, guardar junto a los originales
    out_root = out_root or data_root

    for nb_folder in sorted(glob.glob(os.path.join(data_root, "*b"))):
        nb_str = os.path.basename(nb_folder).replace("b", "")
        try:
            n_balls = int(nb_str)
        except ValueError:
            continue

        csv_files = sorted(glob.glob(os.path.join(nb_folder, "*.csv")))
        augmented_count = 0
        skipped_no_pair = 0

        for path in csv_files:
            fname = os.path.basename(path)[:-4]
            if any(tag in fname for tag in ["_flip", "_noise", "_crop"]):
                continue

            nb, trick, sid = parse_filename(fname)
            if nb is None:
                continue

            data = pd.read_csv(path, header=None).values.astype(np.float32)

            # Definir out_nb_folder UNA SOLA VEZ por archivo, fuera de los bloques
            out_nb_folder = os.path.join(out_root, f"{n_balls}b")
            os.makedirs(out_nb_folder, exist_ok=True)

            # ── FLIP (solo trucos no simétricos y con par en el dataset) ──
            if "flip" in augmentations:
                if trick in SYMMETRIC_TRICKS:
                    pass  # flip no aporta
                elif trick in NO_FLIP_TRICKS:
                    skipped_no_pair += 1  # par no existe en dataset
                else:
                    mirror_name = generate_mirror_name(trick)
                    if mirror_name:
                        flipped = flip_horizontal(data)
                        out_fname = f"{nb}_{mirror_name}_{sid}_flip.csv"
                        out_path = os.path.join(out_nb_folder, out_fname)
                        pd.DataFrame(flipped).to_csv(out_path, header=False, index=False)
                        augmented_count += 1

            # ── NOISE ──
            if "noise" in augmentations:
                noisy = add_gaussian_noise(data, std=3.0, seed=42 + augmented_count)
                out_fname = f"{nb}_{trick}_{sid}_noise.csv"
                out_path = os.path.join(out_nb_folder, out_fname)
                pd.DataFrame(noisy).to_csv(out_path, header=False, index=False)
                augmented_count += 1

            # ── CROP ──
            if "crop" in augmentations:
                cropped = temporal_crop(data, crop_ratio=0.8, seed=42 + augmented_count)
                out_fname = f"{nb}_{trick}_{sid}_crop.csv"
                out_path = os.path.join(out_nb_folder, out_fname)
                pd.DataFrame(cropped).to_csv(out_path, header=False, index=False)
                augmented_count += 1

        print(f"{n_balls}b: {len(csv_files)} originales + {augmented_count} augmentados"
              + (f" ({skipped_no_pair} sin par, omitidos)" if skipped_no_pair else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="runs/track_csvs")
    ap.add_argument("--augmentations", nargs="+",
                    default=["flip", "crop"],
                    choices=["flip", "noise", "crop"],   # shuffle eliminado
                    help="Tipos de augmentación a aplicar")
    ap.add_argument("--out_root", default="runs/track_csvs_augmented",
                    help="Carpeta de salida, None para la misma ruta de entrada")
    args = ap.parse_args()
    augment_dataset(args.data_root, args.augmentations, args.out_root)

if __name__ == "__main__":
    main()