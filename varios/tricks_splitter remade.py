import csv

with open("detailed_tricks.txt", "r") as t:
    lineas = t.readlines()

with open("detailed_tricks.csv", "w", newline='') as archivo_csv:
    writer = csv.writer(archivo_csv)
    # Escribimos encabezados
    writer.writerow(["ss no alfabetica%", "balls", "trickname", "#vids available"])

    for linea in lineas:
        # Ejemplo: (2x,4x) [3b Sync Shower L] 4v
        ss, rest = linea.strip().split(" [")
        rest = rest.rstrip("]")  # Quita el corchete final
        balls_side_trick = rest.split(" ")
        balls = balls_side_trick[0].replace("b", "")
        trickname = " ".join(balls_side_trick[1:]).lower()  # Incluye side en trickname
        vidsno = linea.strip().split("] ")[1].replace("v", "")
        writer.writerow([ss, balls, trickname, vidsno])