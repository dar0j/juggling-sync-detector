import os
import glob

def merge_csvs(gifs_folder, csv_folder):
    # Busca todos los archivos *_annotations.csv en la carpeta gifs
    for gifs_csv in sorted(glob.glob(os.path.join(gifs_folder, "*_annotations.csv"))):
        base = os.path.basename(gifs_csv)
        # Extrae el truco (segundo valor al hacer split por "_")
        parts = base.split("_")
        if len(parts) < 3:
            continue
        trick = parts[1]  # por ejemplo, "(0,6)"
        # Busca todos los archivos de ese truco en la carpeta csv_folder
        pattern = os.path.join(csv_folder, f"*_{trick}_*_annotations.csv")
        matching_csvs = sorted(glob.glob(pattern))
        # Lee y concatena el contenido de todos los matching_csvs
        merged_rows = []
        for csv_file in matching_csvs:
            with open(csv_file, "r") as f:
                merged_rows.extend(f.readlines())
        # Anexa al final del archivo de gifs
        with open(gifs_csv, "a") as f:
            f.writelines(merged_rows)
        print(f"Merged {len(matching_csvs)} files for trick {trick} into {base}")

# Procesa 3b, 4b, 5b, 6b
for nBalls in [3, 4, 5, 6]:
    gifs_folder = f"CSVs/60fps128/{nBalls}b gifs csv 128"
    csv_folder = f"CSVs/60fps128/{nBalls}b csv 60 128"
    if os.path.exists(gifs_folder) and os.path.exists(csv_folder):
        merge_csvs(gifs_folder, csv_folder)
    else:
        print(f"Skipping {nBalls}b: folders not found")