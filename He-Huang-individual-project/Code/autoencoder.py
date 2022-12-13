import numpy as np
import pandas as pd
import os
import re
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------------------------
# hyperparameter

EPOCH = 3

batch_size = 2

image_H = 800
image_W = 1200

low_only = True

ModelName = 'autoencoder1'

#-----------------------------------------------------------------------------------------------------------------
# data preprocessing

# current_path = os.getcwd()
base_directory = '../data'
# base_directory = current_path + '/data'
hires_folder = os.path.join(base_directory, 'high res')
lowres_folder = os.path.join(base_directory, 'low res')

if low_only == True:
    data = pd.read_csv("image_data_low.csv")
else:
    data = pd.read_csv("image_data.csv")
data['low_res'] = data['low_res'].apply(lambda x: os.path.join(lowres_folder, x))
data['high_res'] = data['high_res'].apply(lambda x: os.path.join(hires_folder, x))
# print(data.head())



image_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)
mask_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

train_hiresimage_generator = image_datagen.flow_from_dataframe(
        data,
        x_col='high_res',
        target_size=(image_H, image_W),
        class_mode=None,
        batch_size=batch_size,
        seed=42,
        subset='training')

train_lowresimage_generator = image_datagen.flow_from_dataframe(
        data,
        x_col='low_res',
        target_size=(image_H, image_W),
        class_mode=None,
        batch_size=batch_size,
        seed=42,
        subset='training')

val_hiresimage_generator = image_datagen.flow_from_dataframe(
        data,
        x_col='high_res',
        target_size=(image_H, image_W),
        class_mode=None,
        batch_size=batch_size,
        seed=42,
        subset='validation')

val_lowresimage_generator = image_datagen.flow_from_dataframe(
        data,
        x_col='low_res',
        target_size=(image_H, image_W),
        class_mode=None,
        batch_size=batch_size,
        seed=42,
        subset='validation')

train_generator = zip(train_lowresimage_generator, train_hiresimage_generator)
val_generator = zip(val_lowresimage_generator, val_hiresimage_generator)

def imageGenerator(train_generator):
    for (low_res, hi_res) in train_generator:
            yield (low_res, hi_res)

#-----------------------------------------------------------------------------------------------------------------
# show the first 3 combine pictures

# n = 0
# for i, m in train_generator:
#     img, out = i, m
#
#     if n < 3:
#         fig, axs = plt.subplots(1, 2, figsize=(20, 5))
#         axs[0].imshow(img[0])
#         axs[0].set_title('Low Resolution Image')
#         axs[1].imshow(out[0])
#         axs[1].set_title('High Resolution Image')
#         plt.show()
#         n += 1
#     else:
#         break

#-----------------------------------------------------------------------------------------------------------------
# model design

input_img = Input(shape=(image_H, image_W, 3))

l1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
l2 = Conv2D(64, (3, 3), padding='same', activation='relu')(l1)
l3 = MaxPooling2D(padding='same')(l2)
l3 = Dropout(0.3)(l3)
l4 = Conv2D(128, (3, 3),  padding='same', activation='relu')(l3)
l5 = Conv2D(128, (3, 3), padding='same', activation='relu')(l4)
l6 = MaxPooling2D(padding='same')(l5)
l7 = Conv2D(256, (3, 3), padding='same', activation='relu')(l6)

l8 = UpSampling2D()(l7)

l9 = Conv2D(128, (3, 3), padding='same', activation='relu')(l8)
l10 = Conv2D(128, (3, 3), padding='same', activation='relu')(l9)

l11 = add([l5, l10])
l12 = UpSampling2D()(l11)
l13 = Conv2D(64, (3, 3), padding='same', activation='relu')(l12)
l14 = Conv2D(64, (3, 3), padding='same', activation='relu')(l13)

l15 = add([l14, l2])

decoded = Conv2D(3, (3, 3), padding='same', activation='relu')(l15)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

autoencoder.summary()
with open('summary_{}.txt'.format(ModelName), 'w') as fh:
    autoencoder.summary(print_fn=lambda x: fh.write(x + '\n'))

#-----------------------------------------------------------------------------------------------------------------
# train preparing
train_samples = train_hiresimage_generator.samples
val_samples = val_hiresimage_generator.samples

train_img_gen = imageGenerator(train_generator)
val_image_gen = imageGenerator(val_generator)

model_path = "autoencoder.h5"
checkpoint = ModelCheckpoint('model_{}.h5'.format(ModelName),
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=9,
                          verbose=1,
                          restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=5,
                                            verbose=1,
                                            factor=0.2,
                                            min_lr=0.00000001)

hist = autoencoder.fit(train_img_gen,
                    steps_per_epoch=train_samples//batch_size,
                    validation_data=val_image_gen,
                    validation_steps=val_samples//batch_size,
                    epochs=EPOCH,
                    callbacks=[earlystop, checkpoint, learning_rate_reduction])

#-----------------------------------------------------------------------------------------------------------------
# plot the change of loss
plt.figure(figsize=(20, 8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#-----------------------------------------------------------------------------------------------------------------
# show the prediction of first 3 pictures
n = 0
for i, m in val_generator:
    img, mask = i, m
    sr1 = autoencoder.predict(img)
    if n < 20:
        fig, axs = plt.subplots(1, 3, figsize=(20, 4))
        axs[0].imshow(img[0])
        axs[0].set_title('Low Resolution Image')
        axs[1].imshow(mask[0])
        axs[1].set_title('High Resolution Image')
        axs[2].imshow(sr1[0])
        axs[2].set_title('Predicted High Resolution Image')
        plt.show()
        n += 1
    else:
        break
print('all fine!')
