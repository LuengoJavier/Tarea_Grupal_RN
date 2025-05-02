from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

# funcion para cargar imagenes
def extraer_imagenes_fer(ruta_directorio):
    imagenes_test = []
    imagenes_train = []
    clases_test = []
    clases_train = []

    data_divided = os.listdir(ruta_directorio)

    for division in data_divided:
        k = 0
        print(f"{division} :")
        for clase in os.listdir(os.path.join(ruta_directorio, division)):
            print(f"{clase} : {k}")
            for imagen in os.listdir(os.path.join(ruta_directorio, division, clase)):
                foto = image.load_img(os.path.join(ruta_directorio, division, clase, imagen), 
                                      target_size=(48, 48), 
                                      color_mode="grayscale")
                foto_array = image.img_to_array(foto)

                if division == "test":
                    imagenes_test.append(foto_array)
                    clases_test.append(k)
                elif division == "train":
                    imagenes_train.append(foto_array)
                    clases_train.append(k)
            k += 1

    X_test = np.array(imagenes_test, dtype=np.float32)
    X_train = np.array(imagenes_train, dtype=np.float32)

    image_size = X_train.shape[1]
    X_train = np.reshape(X_train, [-1, image_size, image_size, 1])
    X_test = np.reshape(X_test, [-1, image_size, image_size, 1])

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_test = to_categorical(np.array(clases_test))
    y_train = to_categorical(np.array(clases_train))

    return X_train, X_test, y_train, y_test