import numpy as np
import pandas as pd
import os
import re
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D, add, LeakyReLU
from keras.models import Model
from keras import regularizers

# from keras.preprocessing.image import load_img, img_to_array

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import load_img
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import cv2 as cv
#%% # ------------------------------------------------------------------------------------------------------------------
IMAGE_WIDTH = 800
IMAGE_Length = 1200

#%% # ------------------------------------------------------------------------------------------------------------------
cur_file = os.getcwd()
base_directory = cur_file + '/Image Super Resolution - Unsplash'
hires_folder = os.path.join(base_directory, 'high res')
lowres_folder = os.path.join(base_directory, 'low res')

data = pd.read_csv(base_directory + f"/image_data.csv", encoding='ISO-8859-1')
data['low_res'] = data['low_res'].apply(lambda x: os.path.join(lowres_folder,x))
data['high_res'] = data['high_res'].apply(lambda x: os.path.join(hires_folder,x))
print(data.head())

batch_size = 1

image_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.15)
mask_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.15)

train_hiresimage_generator = image_datagen.flow_from_dataframe(
        data,
        x_col='high_res',
        target_size=(IMAGE_WIDTH, IMAGE_Length),
        class_mode = None,
        batch_size = batch_size,
        seed=42,
        subset='training')

train_lowresimage_generator = image_datagen.flow_from_dataframe(
        data,
        x_col='low_res',
        target_size=(IMAGE_WIDTH, IMAGE_Length),
        class_mode = None,
        batch_size = batch_size,
        seed=42,
        subset='training')

val_hiresimage_generator = image_datagen.flow_from_dataframe(
        data,
        x_col='high_res',
        target_size=(IMAGE_WIDTH, IMAGE_Length),
        class_mode = None,
        batch_size = batch_size,
        seed=42,
        subset='validation')

val_lowresimage_generator = image_datagen.flow_from_dataframe(
        data,
        x_col='low_res',
        target_size=(IMAGE_WIDTH, IMAGE_Length),
        class_mode = None,
        batch_size = batch_size,
        seed=42,
        subset='validation')



train_generator = zip(train_lowresimage_generator, train_hiresimage_generator)
val_generator = zip(val_lowresimage_generator, val_hiresimage_generator)

def imageGenerator(train_generator):
    for (low_res, hi_res) in train_generator:
            yield (low_res, hi_res)

n = 0
for i,m in train_generator:
    img,out = i,m

    if n < 5:
        fig, axs = plt.subplots(1 , 2, figsize=(20,5))
        axs[0].imshow(img[0])
        axs[0].set_title('Low Resolution Image')
        axs[1].imshow(out[0])
        axs[1].set_title('High Resolution Image')
        plt.show()
        n+=1
    else:
        break

input_img = Input(shape=(IMAGE_WIDTH, IMAGE_Length, 3), name='encoder_input')

l1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
l2 = Conv2D(64, (3, 3), padding='same', activation='relu')(l1)
l3 = Conv2D(64, (3, 3), padding='same', activation='relu')(l2)
l4 = MaxPooling2D(padding='same')(l3)
l4 = Dropout(0.3)(l4)
l5 = Conv2D(128, (3, 3),  padding='same', activation='relu')(l4)
l6 = Conv2D(128, (3, 3), padding='same', activation='relu')(l5)
l7 = MaxPooling2D(padding='same')(l6)
l8 = Conv2D(256, (3, 3), padding='same', activation='relu')(l7)

# l9 = UpSampling2D()(l8)
l10 = Conv2DTranspose(256, (3, 3), padding='same', activation='relu', strides=2, output_padding=1)(l8)
l11 = Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(l10)


l12 = add([l6, l11])
# l13 = UpSampling2D()(l12)
l14 = Conv2DTranspose(128, (3, 3), padding='same', activation='relu', strides=2, output_padding=1)(l12)
l15 = Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(l14)
l16 = Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(l15)

l17 = add([l16, l3])

decoded = Conv2DTranspose(3, (3, 3), padding='same', activation='relu')(l17)

autoencoder = Model(input_img, decoded)
# autoencoder = tf.keras.models.load_model('autoencoder_change_deconvolution_1200.h5')
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

autoencoder.summary()

train_samples = train_hiresimage_generator.samples
val_samples = val_hiresimage_generator.samples

train_img_gen = imageGenerator(train_generator)
val_image_gen = imageGenerator(val_generator)

model_path = "autoencoder_change_deconvolution_1200.h5"
checkpoint = ModelCheckpoint(model_path,
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 9,
                          verbose = 1,
                          restore_best_weights = True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=5,
                                            verbose=1,
                                            factor=0.2,
                                            min_lr=0.00000001)

hist = autoencoder.fit(train_img_gen,
                    steps_per_epoch=train_samples//batch_size,
                    validation_data=val_image_gen,
                    validation_steps=val_samples//batch_size,
                    epochs=8, callbacks=[earlystop, checkpoint, learning_rate_reduction])

plt.figure(figsize=(20,8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

n = 0
for i,m in val_generator:
    img,mask = i,m
    sr1 = autoencoder.predict(img)
    if n < 20:
        fig, axs = plt.subplots(1 , 3, figsize=(20,4))
        axs[0].imshow(img[0])
        axs[0].set_title('Low Resolution Image')
        axs[1].imshow(mask[0])
        axs[1].set_title('High Resolution Image')
        axs[2].imshow(sr1[0])
        axs[2].set_title('Predicted High Resolution Image')
        plt.show()
        n+=1
        # cv.imwrite(f'test_img/test_N_6_{n}.jpg', sr1[0]*255)
    else:
        break

