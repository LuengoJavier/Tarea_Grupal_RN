from tensorflow.keras.layers import (
    Input, Activation, GlobalAveragePooling2D
)
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Input, Flatten, Dropout
from tensorflow.keras.layers import concatenate, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
import math
from tensorflow.keras.preprocessing import image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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



# Parametros de entrenamiento
batch_size = 32
#Epocas
epochs = 200
data_augmentation = True

# Parametros de la arquitectura
num_classes = 7
num_dense_blocks = 4
use_max_pool = False
growth_rate = 12
depth = 100
num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
num_filters_bef_dense_block = 2 * growth_rate

# Factor de compresion
compression_factor = 0.7

# Carga de datos a través de la función
x_train, x_test, y_train, y_test = extraer_imagenes("FER")

# Dimensiones de la entrada
input_shape = x_train.shape[1:]

# Se muestran las dimensiones
print('Dimensiones de la entrada:')
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Función de aprendizaje
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


# Definicion del modelo

# DesNet Conv2D-BN-ReLU-Conv2D
inputs = Input(shape=input_shape)
x = BatchNormalization()(inputs)
x = Activation('relu')(x)
x = Conv2D(num_filters_bef_dense_block,kernel_size=3,padding='same',kernel_initializer='he_normal')(x)
x = concatenate([inputs, x])

# stack of dense blocks bridged by transition layers
for i in range(num_dense_blocks):
    # a dense block is a stack of bottleneck layers
    for j in range(num_bottleneck_layers):
        y = BatchNormalization()(x)
        y = Activation('relu')(y)
        y = Conv2D(4 * growth_rate,kernel_size=1,padding='same',kernel_initializer='he_normal')(y)
        if not data_augmentation:
            y = Dropout(0.2)(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(growth_rate,kernel_size=3,padding='same',kernel_initializer='he_normal')(y)
        if not data_augmentation:
            y = Dropout(0.2)(y)
        x = concatenate([x, y])

    # no transition layer after the last dense block
    if i == num_dense_blocks - 1:
        continue

    # transition layer compresses num of feature maps and reduces the size by 2
    num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
    num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
    y = BatchNormalization()(x)
    y = Conv2D(num_filters_bef_dense_block,kernel_size=1,padding='same',kernel_initializer='he_normal')(y)
    if not data_augmentation:
        y = Dropout(0.2)(y)
    x = AveragePooling2D()(y)


# add classifier on top
# after average pooling, size of feature map is 1 x 1
x = AveragePooling2D(pool_size=4)(x)
y = Flatten()(x)
outputs = Dense(num_classes,kernel_initializer='he_normal',activation='softmax')(y)

# instantiate and compile model
# orig paper uses SGD but RMSprop works better for DenseNet
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(1e-3),metrics=['acc'])
model.summary()

# prepare model model saving directory
save_dir = os.path.join(os.getcwd(), 'saved_models_'+ str(epochs) +'_epocas'+'_0.7_compression_factor')
model_name = 'FER2013_densenet_model_compression_factor_0.7.{epoch:02d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Callbacks para guardar el mejor modelo y reducir la perdida

checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_acc',verbose=2,save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

import time 

start=time.time()

# run training, with or without data augmentation
if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,
              validation_data=(x_test, y_test),shuffle=True,callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # preprocessing  and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (deg 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)

    steps_per_epoch = math.ceil(len(x_train) / batch_size)
    # fit the model on the batches generated by datagen.flow().
    history = model.fit(x=datagen.flow(x_train, y_train, batch_size=batch_size),verbose=1,epochs=epochs,
              validation_data=(x_test, y_test),steps_per_epoch=steps_per_epoch,callbacks=callbacks)

fin=time.time()
print('(RUNNING TIME: ' + str(fin-start) )
# score trained model
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

##-------------------
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Funcion de Perdida')
plt.ylabel('Valor perdida')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Test'], loc='best')
plt.savefig('funcion_perdida_modelo_densenet_0.7.png')
##-------------------
# Verifica si es 'accuracy' o 'acc'
acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'

# Guardar exactitud
plt.figure()
plt.plot(history.history[acc_key])
plt.plot(history.history[val_acc_key])
plt.title('Exactitud del modelo')
plt.ylabel('Exactitud')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Validacion'], loc='best')
plt.savefig('funcion_exactitud_modelo_densenet_0.7.png')