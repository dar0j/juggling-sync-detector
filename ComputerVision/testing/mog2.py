import cv2

# Crear el objeto de sustracción de fondo con MOG2
fgbg = cv2.createBackgroundSubtractorMOG2()

# Abrir la cámara (0 = cámara por defecto)
cap = cv2.VideoCapture('/home/dar0j/Documentos/2025/intro trabajo titulo el E/PROJECT/Datasets/5b/5_(6x,4x)_12.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicar la máscara MOG2
    fgmask = fgbg.apply(frame)
    
    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar contornos sobre la imagen original
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    width = 360
    height = 640
    img_out_small = cv2.resize(frame, (width, height))
    mask_small = cv2.resize(fgmask, (width, height))
    # Mostrar la imagen original y la máscara
    cv2.imshow('Camara', img_out_small)
    cv2.imshow('Mascara MOG2', mask_small)

    # Salir con la tecla 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
