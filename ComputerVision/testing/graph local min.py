import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Supongamos que ya tienes tus datos
x = np.linspace(-10, 10, 1000)
y = np.sin(x) + 0.5 * np.cos(2*x)  # Ejemplo de función con varios mínimos

# Encontrar índices de mínimos locales
min_indices = argrelextrema(y, np.less, order=1)[0] # order 1 defecto,5 robusto ante ruido y 100 sugerencia
# con datos ruidosos, aplicar un suavizado previo con scipy.ndimage.gaussian_filter1d.

# Graficar
plt.plot(x, y, label='Función')
plt.plot(x[min_indices], y[min_indices], 'ro', label='Mínimos locales')
plt.legend()
plt.grid(True)
plt.title('Mínimos locales de la función')
plt.show()
