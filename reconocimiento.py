import cv2
import numpy as np
from keras.models import load_model

# ==============================
# CONFIGURACIÓN
# ==============================

UMBRAL_CONFIANZA = 0.85  # 85%

# ==============================
# CARGAR MODELO
# ==============================

model = load_model("keras_model.h5", compile=False)

# Cargar nombres de las clases
class_names = open("labels.txt", "r").readlines()

# ==============================
# ABRIR CÁMARA
# ==============================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Cámara iniciada.")
print("Presiona Q para salir.")

# ==============================
# RECONOCIMIENTO
# ==============================

while True:

    # Leer imagen de la cámara
    ret, frame = camera.read()

    if not ret:
        print("No se pudo obtener imagen de la cámara.")
        break

    # Preparar imagen para el modelo
    image = cv2.resize(frame, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalizar
    image = (image.astype(np.float32) / 127.5) - 1

    # Crear arreglo para el modelo
    data = np.ndarray(
        shape=(1, 224, 224, 3),
        dtype=np.float32
    )

    data[0] = image

    # ==============================
    # PREDICCIÓN
    # ==============================

    prediction = model.predict(data, verbose=0)

    index = np.argmax(prediction)
    confidence = prediction[0][index]

    # ==============================
    # DECIDIR SI RECONOCER
    # ==============================

    if confidence >= UMBRAL_CONFIANZA:

        class_name = class_names[index].strip()

        texto = f"{class_name} - {confidence * 100:.1f}%"

    else:

        texto = f"No reconocido - {confidence * 100:.1f}%"

    # ==============================
    # MOSTRAR RESULTADO
    # ==============================

    cv2.putText(
        frame,
        texto,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Mostrar cámara
    cv2.imshow("Reconocimiento", frame)

    # Presionar Q para salir
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# CERRAR
# ==============================

camera.release()
cv2.destroyAllWindows()