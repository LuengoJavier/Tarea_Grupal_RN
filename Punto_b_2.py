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

# Para ResNet V2, la profundidad se calcula diferente
n = 17  
depth = n * 9 + 2  # 155 capas

fer_path = "/home/cursos/ima543_2025_1/ima543_share/FER"  #Ruta Khipu

def extraer_imagenes(directorio):
    """Carga el dataset FER desde carpetas usando load_img"""
    print(f"Cargando imágenes desde: {directorio}")
    imagenes_train, clases_train = [], []
    imagenes_test, clases_test = [], []

    # Cargar imágenes de entrenamiento y prueba
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


def lr_schedule(epoch): #Función tasa de aprendizaje
    
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

# Función de construccion de capas de ResNet
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

# Implementación ResNet V2
def resnet_v2(input_shape, depth, n_clases=7):
    """ResNet Version 2 Model con pre-activación"""
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (e.g. 29, 47, 56, 92, 110, 164)')
    
    # Empezar definición del modelo
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    
    # v2 realiza Conv2D con BN-ReLU en la entrada antes de dividirse en caminos
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)
    
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                # Primer stage mantiene resolución, duplica filtros
                num_filters_out = num_filters_in * 4
                # Primera capa y primer stage
                if res_block == 0:  
                    activation = None
                    batch_normalization = False
            else:
                # Otros stages reducen resolución, duplican filtros
                num_filters_out = num_filters_in * 2
                # Primera capa pero no primer stage
                if res_block == 0:
                    # downsample
                    strides = 2
            
            # Bloque residual con pre-activación (BN-ReLU-Conv)
            # Primera parte del bloque
            y = resnet_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False
            )
            
            # Segunda parte del bloque
            y = resnet_layer(
                inputs=y,
                num_filters=num_filters_in,
                conv_first=False
            )
            
            # Tercera parte del bloque
            y = resnet_layer(
                inputs=y,
                num_filters=num_filters_out,
                kernel_size=1,
                conv_first=False
            )
            
            # Proyección de la conexión residual si cambia dimensiones
            if res_block == 0:
                # Proyección lineal para ajustar dimensiones
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False
                )
            
            # Suma con el camino directo (sin activación después)
            x = add([x, y])
        
        num_filters_in = num_filters_out
    
    # En ResNet V2, BN-ReLU antes del Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
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
    
    # Crear modelo ResNet V2
    model = resnet_v2(input_shape=input_shape, depth=depth, n_clases=n_clases)
    model.summary()
    
    # Preparar directorio para guardar el modelo
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = f'fer2013_resnet{depth}v2.h5'  # Cambio a v2
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    
    # Guardar también los pesos por separado (según requisitos)
    weights_path = os.path.join(save_dir, f'fer2013_resnet{depth}v2_weights.h5')
    
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
    
    # Guardar modelo completo
    model.save(filepath)
    
    # Guardar también los pesos por separado (según requisitos)
    model.save_weights(weights_path)
    
    plt.figure(figsize=(12, 5))#curvas de entrenamiento
    
    # Grafico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precisión del modelo ResNet V2')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='lower right')
    
    # Grafica de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pérdida del modelo ResNet V2')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('resnet_v2_fer2013_train.png')
    
    performance = {
        'model': f'ResNet{depth}v2',  # Cambio a v2
        'training_time_seconds': elapsed_time,
        'training_time_formatted': f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
        'test_loss': scores[0],
        'test_accuracy': scores[1],
        'epochs': epochs
    }
    
    with open('resnet_v2_results.txt', 'w') as f:  # Cambio a v2
        for key, value in performance.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()