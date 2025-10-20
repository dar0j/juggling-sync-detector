import os
import glob
import numpy as np
import pandas as pd
from keras.models import load_model
import re

# Ajusta la ruta al modelo y a los CSV
pattern_model_path = "../../../old rasmus/pattern_models/3b_pattern_model.h5"
csv_dir = "../../../old rasmus/patterns"#"patterns3coincgif" #"../../../old rasmus/patterns3-6"
csv_files = glob.glob(os.path.join(csv_dir, "*.csv")) #"*_annotations.csv"))

# Carga el modelo
model = load_model(pattern_model_path)
# Nombres de patrones (ajusta si tienes otros)
pattern_names = [
    "441", "(4,2x)(2x,4)", "cascade", "(2,4)", "(2x,4x)", "mill's mess",
    "one up two up", "(4,2)", "reverse cascade", "(4x,2x)", "takeouts", "tennis"
]
SEQUENCE_LENGTH = 30

for csv_path in csv_files:
    base = os.path.basename(csv_path)
    # Extrae el patrón entre paréntesis
    match = re.search(r'\((.*)\)', base)
    true_label = match.group(0) if match else None

    data = pd.read_csv(csv_path, header=None).values
    print(f"\nArchivo: {base} (Etiqueta real: {true_label})")
    correct = 0
    total = 0
    for i in range(len(data) - SEQUENCE_LENGTH + 1):
        pattern = np.array(data[i:i+SEQUENCE_LENGTH], dtype=np.float32)
        # Normalización igual que en PatternDataLoader
        pattern[:,::2] = pattern[:,::2] - np.mean(pattern[:,::2])
        pattern[:,1::2] = pattern[:,1::2] - np.mean(pattern[:,1::2])
        pattern = pattern / (pattern.std() + 1e-8)
        pattern = np.expand_dims(pattern, axis=0)
        pred = model.predict(pattern)[0]
        pred_idx = np.argmax(pred)
        pred_name = pattern_names[pred_idx]
        print(f"Secuencia {i}-{i+SEQUENCE_LENGTH}: Predicción: {pred_name}", end="")
        if true_label:
            if pred_name == true_label:
                print("  ✔️")
                correct += 1
            else:
                print(f"  ✖️ (esperado: {true_label})")
            total += 1
        else:
            print()
    if total > 0:
        print(f"Accuracy para {base}: {correct}/{total} ({100*correct/total:.1f}%)")