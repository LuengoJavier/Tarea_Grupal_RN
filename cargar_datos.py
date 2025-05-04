from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

def extraer_imagenes(directorio):
    imagenes_train, clases_train = [], []
    imagenes_test, clases_test = [], []

    for tipo_datos in ['train', 'test']:
        ruta_tipo = os.path.join(directorio, tipo_datos)
        clases = sorted(os.listdir(ruta_tipo))  # Asegura orden consistente de clases

        for idx, clase in enumerate(clases):
            ruta_clase = os.path.join(ruta_tipo, clase)
            for foto in os.listdir(ruta_clase):
                ruta_foto = os.path.join(ruta_clase, foto)
                try:
                    img = image.load_img(ruta_foto, target_size=(48, 48), color_mode='grayscale')
                    img_array = np.array(img)

                    if tipo_datos == 'train':
                        imagenes_train.append(img_array)
                        clases_train.append(idx)
                    else:
                        imagenes_test.append(img_array)
                        clases_test.append(idx)
                except Exception as e:
                    print(f"Error al cargar {ruta_foto}: {e}")

    # Convertir a arrays y normalizar
    X_train = np.expand_dims(np.array(imagenes_train, dtype=np.float32) / 255, axis=-1)
    X_test = np.expand_dims(np.array(imagenes_test, dtype=np.float32) / 255, axis=-1)
    y_train = to_categorical(clases_train)
    y_test = to_categorical(clases_test)

    return X_train, X_test, y_train, y_test

