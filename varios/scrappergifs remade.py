import requests
import os

# Lista de patrones
patterns = [
    "(0,6)", "(2,4)", "(2x,4x)", "(4,2)", "(4,2x)*", "(4,4)(0,4)", "(4,4)(4,0)",
    "(4x,2x)", "(6,0)", "(0,8)", "(2x,6x)", "(4,4)", "(4x,4x)", "(6,6)(2x,2x)",
    "(6x,2x)*", "(6x,2x)", "(8,0)", "(4x,6x)", "(6,6)(6,6)(6x,0)*", "(6,6)(6x,2)*",
    "(6x,4)*", "(6x,4x)", "(6x,6x)(6x,2)*", "(8x,2)*", "(8x,2x)", "(6,6)", "(6x,6x)"
]

output_dir = "gifs"
os.makedirs(output_dir, exist_ok=True)

# Descargar cada GIF
for pattern in patterns:
    url = f"https://jugglinglab.org/anim?pattern={pattern};redirect=true"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            filename = f"{pattern.replace('*','()').replace('(','').replace(')','').replace(',','_')}.gif"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"✅ Descargado: {filepath}")
        else:
            print(f"⚠️ Error al descargar {pattern}: {response.status_code}")
    except Exception as e:
        print(f"❌ Falló {pattern}: {e}")