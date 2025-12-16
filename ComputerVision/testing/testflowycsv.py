import pandas as pd

# Leer archivos
df = pd.read_csv("3_(4,2x)(2x,4).csv", header=None, dtype=int)  # tiene manos en cols 0-3
df_pelotas = pd.read_csv("tracking_uncalibratedBOXgifs.csv", header=None, dtype=int)  # solo pelotas (sin manos)

# Tomar manos
df_manos = df.iloc[:, :4]

# Escalar pelotas a 256x256
# Obtener dimensiones originales del video (asumiendo que conoces las dimensiones)
# Si no las conoces, usa los valores máximos como referencia
original_width = 400# df_pelotas.iloc[:, ::2].max().max()  # max de columnas X (pares)
original_height = 450 #df_pelotas.iloc[:, 1::2].max().max()  # max de columnas Y (impares)

print(f"Dimensiones originales detectadas: {original_width}x{original_height}")

# Escalar columnas X (pares: 0, 2, 4, ...)
df_pelotas.iloc[:, ::2] = (df_pelotas.iloc[:, ::2] * 256 / original_width).round().astype(int)

# Escalar columnas Y (impares: 1, 3, 5, ...)
df_pelotas.iloc[:, 1::2] = (df_pelotas.iloc[:, 1::2] * 256 / original_height).round().astype(int)

# Combinar
df_combinado = pd.concat([df_manos, df_pelotas], axis=1)

# Guardar
df_combinado.to_csv("box3gif_hands.csv", index=False, header=False)

print("CSV combinado creado: manos (4 cols) + pelotas escaladas a 256x256 - enteros")
