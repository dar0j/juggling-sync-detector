import pandas as pd

# Leer el archivo CSV
df = pd.read_csv("3_(4,2x)(2x,4)_3.csv")

# Conservar solo las 4 primeras columnas
df_reducido = df.iloc[:, :4]

# Guardar el nuevo CSV
df_reducido.to_csv("box3_hands.csv", index=False)

print("Archivo reducido creado con éxito.")
