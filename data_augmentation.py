import os
import glob
import pandas as pd
import numpy as np

# Trucos excluidos (ya simétricos o patrones especiales)
EXCLUDED_TRICKS = [
    "(4,2x)(2x,4)", "(4,4)", "(4x,4x)", "(6,6)(2x,2x)", "(6x,2x)(2x,6x)",
    "(6x,0)(6,6)(6,6)(0,6x)(6,6)(6,6)", "(6x,2)(6,6)(2,6x)(6,6)", "(6x,4)(4,6x)",
    "(6x,6x)(6x,2)(6x,6x)(2,6x)", "(8x,2)(2,8x)", "(8x,2x)", "(6,6)", "(6x,6x)"
]

# Ancho de imagen de referencia (ajustar según tus videos)
IMAGE_WIDTH = 256  # basado en carpeta "60fps128"

def flip_coordinates(csv_path, image_width=IMAGE_WIDTH):
    """
    Flipea coordenadas horizontalmente y swapea manos.
    Retorna array flipeado con mismo formato.
    """
    data = pd.read_csv(csv_path, header=None).values
    flipped = data.copy()
    
    # Columnas: x_right, y_right, x_left, y_left, x_ball1, y_ball1, ...
    # Flipear x (columnas pares: 0, 2, 4, ...)
    flipped[:, 0::2] = image_width - data[:, 0::2]
    
    # Swapear manos: right (cols 0,1) ↔ left (cols 2,3)
    flipped[:, [0,1,2,3]] = flipped[:, [2,3,0,1]]
    
    return flipped

def generate_symmetric_pair(trick_name):
    """
    Genera nombre de truco simétrico si aplica.
    Ej: (0,6) → (6,0), (4,2) → (2,4)
    """
    # if trick_name.startswith('(') and ',' in trick_name:
    #     parts = trick_name.strip('()').split(',')
    #     if len(parts) == 2:
    #         return f"({parts[1]},{parts[0]})"
    # return None
    import re
    coords = re.findall(r'\(([^,]+),([^)]+)\)', trick_name)
    puntos = [(x, y) for x, y in coords]
    
    # Reflejar cada punto respecto a la diagonal y=x (intercambiar x,y)
    reflejados = [(y, x) for x, y in puntos]
    # Reconstruir la cadena en formato "(x,y)(x,y)"
    resultado = ''.join([f'({x},{y})' for x, y in reflejados])
    return resultado

def augment_dataset(data_root="../CSVs/60fps128", output_root="../CSVs/60fps128/flip augmented"):
    """
    Genera versiones flipeadas de trucos no excluidos.
    Guarda CSVs aumentados con sufijo _flip.
    """
    os.makedirs(output_root, exist_ok=True)
    csv_files = glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True)
    
    augmented_count = 0
    for path in csv_files:
        fname = os.path.basename(path)[:-4]
        tokens = fname.split("_")
        if len(tokens) < 2:
            continue
        
        nballs_str = tokens[0]
        if len(tokens) > 2 and tokens[-1].isdigit():
            trick_tokens = tokens[1:-1]
            sample_id = tokens[-1]
        else:
            trick_tokens = tokens[1:]
            sample_id = "00"
        
        trickname = "_".join(trick_tokens)
        
        # Verificar si está excluido
        if trickname in EXCLUDED_TRICKS:
            continue
        
        # Generar par simétrico
        symmetric_trick = generate_symmetric_pair(trickname)
        if not symmetric_trick:
            continue
        
        # Flipear coordenadas
        flipped = flip_coordinates(path)
        
        # Guardar con nombre simétrico
        new_fname = f"{nballs_str}_{symmetric_trick}_{sample_id}.csv"
        output_path = os.path.join(output_root, new_fname)
        pd.DataFrame(flipped).to_csv(output_path, header=False, index=False)
        augmented_count += 1
        
        print(f"Flipped {fname} → {new_fname}")
    
    print(f"\nTotal augmented samples: {augmented_count}")

def augment_symmetric_patterns(data_root="../CSVs/60fps128", output_root="../CSVs/60fps128/shuffled augmented", num_versions=3):
    """
    Genera versiones shuffleadas de patrones simétricos.
    """
    os.makedirs(output_root, exist_ok=True)
    csv_files = glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True)
    
    augmented_count = 0
    for path in csv_files:
        fname = os.path.basename(path)[:-4]
        tokens = fname.split("_")
        if len(tokens) < 2:
            continue
        
        nballs_str = tokens[0]
        nballs = int(nballs_str)
        if len(tokens) > 2 and tokens[-1].isdigit():
            trick_tokens = tokens[1:-1]
            sample_id = tokens[-1]
        else:
            trick_tokens = tokens[1:]
            sample_id = "00"
        
        trickname = "_".join(trick_tokens)
        
        # Solo augmentar simétricos (SIN (8x,2x) que NO es simétrico)
        SYMMETRIC_TRICKS = [
            "(4,2x)(2x,4)", "(4,4)", "(4x,4x)", "(6,6)(2x,2x)", "(6x,2x)(2x,6x)",
            "(6x,0)(6,6)(6,6)(0,6x)(6,6)(6,6)", "(6x,2)(6,6)(2,6x)(6,6)", "(6x,4)(4,6x)",
            "(6x,6x)(6x,2)(6x,6x)(2,6x)", "(8x,2)(2,8x)", "(6,6)", "(6x,6x)"
        ]
        
        if trickname not in SYMMETRIC_TRICKS:
            continue
        
        data = pd.read_csv(path, header=None).values
        
        # Generar num_versions shuffleadas
        for version in range(num_versions):
            shuffled = data.copy()
            np.random.seed(42 + augmented_count)  # seed diferente por versión
            
            for t in range(shuffled.shape[0]):
                # Shufflear pelotas (columnas 4+)
                balls = shuffled[t, 4:4+2*nballs].reshape(nballs, 2)
                np.random.shuffle(balls)
                shuffled[t, 4:4+2*nballs] = balls.flatten()
            
            # Guardar con sufijo shuffle_{version}
            new_fname = f"{nballs_str}_{trickname}_{sample_id}_shuffle{version}.csv"
            output_path = os.path.join(output_root, new_fname)
            pd.DataFrame(shuffled).to_csv(output_path, header=False, index=False)
            augmented_count += 1
        
        print(f"Shuffled {fname} → {num_versions} versiones")
    
    print(f"\nTotal shuffled samples: {augmented_count}")

if __name__ == "__main__":
    augment_dataset()  # flip horizontal
    augment_symmetric_patterns(num_versions=3)  # shuffle simétricos