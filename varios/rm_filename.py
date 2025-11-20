import os

string_a_remover = "_annotations"
#L=['(0,6)', '(2,4)', '(2x,4x)', '(4,4)(0,4)', '(0,8)', '(2x,6x)', '(4x,6x)']
#R=['(4,2)', '(4,4)(4,0)', '(4x,2x)', '(6,0)', '(6x,2x)', '(8,0)', '(6x,4x)', '(8x,2x)']

for fps in [30,60]:
    for model in [64,128]:
        for ball in range(3,7):
            #directorio = f"CSVs/{fps}fps{model}/{ball}b gifs csv"
            #if model==128: #al descomentar indentar la sgte linea
            directorio = f"CSVs/{fps}fps{model}/{ball}b csv {fps} 128" # gifs csv 128"
            for nombre_archivo in os.listdir(directorio):
                #if nombre_archivo.endswith(".csv"): #comentado porque todos son csv
                ruta_completa = os.path.join(directorio, nombre_archivo)
                # Verifica que sea archivo (no carpeta), comentado pq solo hay archivos
                #if os.path.isfile(ruta_completa):
                sinannotation = nombre_archivo.replace(string_a_remover, "")
                # splitext = os.path.splitext(sinannotation)[0]
                # trick = splitext.split("_")[1]
                # if trick in L:
                #     renamed_trick = "L"+trick
                # elif trick in R:
                #     renamed_trick = "R"+trick
                # else:
                #     renamed_trick = trick
                # nuevo_nombre = sinannotation.replace(trick, renamed_trick)
                nueva_ruta = os.path.join(directorio, sinannotation) #nuevo_nombre)
                os.rename(ruta_completa, nueva_ruta)
                print(f"Renombrado: {nombre_archivo} → {sinannotation}") #{nuevo_nombre}")
