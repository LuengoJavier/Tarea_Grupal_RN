import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_USE_CUDNN_BATCHNORM'] = '0' 

batch_size = 64
epochs = 200
data_augmentation = True
n_clases = 7 #clases 
subtract_pixel_mean = True

n = 17  
depth = n * 6 + 2  # 104 capas

fer_path = "/home/cursos/ima543_2025_1/ima543_share/FER"  # Ruta Khipu

def extraer_imagenes(directorio):
    """Carga el dataset FER desde carpetas usando load_img"""
    print(f"Cargando imágenes desde: {directorio}")
    imagenes_train, clases_train = [], []
    imagenes_test, clases_test = [], []

    # Cargar imágenes de entrenamiento y prueba
    for tipo_datos in ['train', 'test']:
        ruta_tipo = os.path.join(directorio, tipo_datos)
        clases = sorted(os.listdir(ruta_tipo))  # Asegura orden consistente de clases
        print(f"Clases encontradas: {clases}")

        for idx, clase in enumerate(clases):
            ruta_clase = os.path.join(ruta_tipo, clase)
            print(f"Cargando clase: {clase} ({idx})")
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

    # Converto a arrays y normalizar
    X_train = np.expand_dims(np.array(imagenes_train, dtype=np.float32) / 255.0, axis=-1)
    X_test = np.expand_dims(np.array(imagenes_test, dtype=np.float32) / 255.0, axis=-1)
    y_train = to_categorical(clases_train)
    y_test = to_categorical(clases_test)
    
    print(f"Imágenes de entrenamiento: {X_train.shape[0]}")
    print(f"Imágenes de prueba: {X_test.shape[0]}")

    # Separar un conjunto de validación del conjunto de entrenamiento
    val_size = int(0.2 * X_train.shape[0])
    indices = np.random.permutation(X_train.shape[0])
    training_idx, val_idx = indices[val_size:], indices[:val_size]
    
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    X_train = X_train[training_idx]
    y_train = y_train[training_idx]
    
    print(f"Imágenes para validación: {X_val.shape[0]}")
    print(f"Imágenes para entrenamiento (después de separar validación): {X_train.shape[0]}")

    return X_train, X_val, X_test, y_train, y_val, y_test

# Learning rate scheduler
def lr_schedule(epoch):
    
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# Función de utilidad para construir capas de ResNet
def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True, conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder"""
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
                  padding='same', kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

# Implementación ResNet V1
def resnet_v1(input_shape, depth, n_clases=7):
    """ResNet Version 1 Model"""
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (e.g. 20, 32, 44, 56, 110, 164)')
    
    # Empezar definición del modelo
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    
    # Instanciar la pila de unidades residuales
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # primera capa pero no primer stack
                strides = 2  # downsampling
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            
            if stack > 0 and res_block == 0:
                # proyección lineal de la conexión residual
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1,
                                strides=strides, activation=None, batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Añadir clasificador
    x = AveragePooling2D(pool_size=4)(x)  # Ajustado para tamaño 48x48
    y = Flatten()(x)
    outputs = Dense(n_clases, activation='softmax', kernel_initializer='he_normal')(y)

    # Instanciar modelo
    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    """Función principal para cargar datos, crear modelo, entrenar y evaluar"""    
    # Cargar datos usando el método alternativo
    X_train, X_val, X_test, y_train, y_val, y_test = extraer_imagenes(fer_path)
    
    # Definir la forma de la entrada (48x48x1 para imágenes en escala de grises)
    input_shape = (48, 48, 1)
    
    # Crear modelo ResNet V1
    model = resnet_v1(input_shape=input_shape, depth=depth, n_clases=n_clases)
    model.summary()
    
    # Preparar directorio para guardar el modelo
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = f'fer2013_resnet{depth}v1.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    
    # Preparar callbacks
    checkpoint = ModelCheckpoint(filepath=filepath, 
                                monitor='val_accuracy',
                                verbose=1, 
                                save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                  cooldown=0,
                                  patience=5,
                                  min_lr=0.5e-6)
    
    callbacks = [checkpoint, lr_reducer, lr_scheduler]
    
    # Compilo
    model.compile(loss='categorical_crossentropy',
                 optimizer=Adam(learning_rate=lr_schedule(0)),
                 metrics=['accuracy'])
    
    #tiempo inicial
    start_time = time.time()
    
    # Configurar data augmentation si está habilitado
    if data_augmentation:
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
        
        # Ajustar el generador
        datagen.fit(X_train)
        
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            steps_per_epoch=math.ceil(X_train.shape[0] / batch_size),
            verbose=1
        )
    else:
        
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            shuffle=True,
            callbacks=callbacks,
            verbose=1
        )
    
    #tiempo total
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTiempo de entrenamiento: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Evaluar modelo
    scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    print(f'Test loss: {scores[0]:.4f}')
    print(f'Test accuracy: {scores[1]:.4f}')
    
  
    model.save(filepath)
    
    
    plt.figure(figsize=(12, 5))#curvas de entrenamiento
    
    # Grqfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precisión del modelo')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='lower right')
    
    # Grafica de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pérdida del modelo')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('resnet_v1_fer2013_train.png')
    
    performance = {
        'model': f'ResNet{depth}v1',
        'training_time_seconds': elapsed_time,
        'training_time_formatted': f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
        'test_loss': scores[0],
        'test_accuracy': scores[1],
        'epochs': epochs
    }
    
    with open('resnet_v1.txt', 'w') as f: #guardo es txt
        for key, value in performance.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()