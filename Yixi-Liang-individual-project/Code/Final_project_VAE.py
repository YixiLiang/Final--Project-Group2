'''
@Project ：test_ml2 
@File    ：Final_project_original.py
@Author  ：Yixi Liang
@Date    ：2022/12/7 10:29 
'''
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
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# %% # ------------------------------------------------------------------------------------------------------------------
IMAGE_WIDTH = 400
IMAGE_Length = 600

# %% # ------------------------------------------------------------------------------------------------------------------
cur_file = os.getcwd()
base_directory = cur_file + '/Image Super Resolution - Unsplash'
hires_folder = os.path.join(base_directory, 'high res')
lowres_folder = os.path.join(base_directory, 'low res')

data = pd.read_csv(base_directory + f"/image_data.csv", encoding='ISO-8859-1')
data['low_res'] = data['low_res'].apply(lambda x: os.path.join(lowres_folder, x))
data['high_res'] = data['high_res'].apply(lambda x: os.path.join(hires_folder, x))
print(data.head())

batch_size = 1

image_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.15)
mask_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.15)

train_hiresimage_generator = image_datagen.flow_from_dataframe(
    data,
    x_col='high_res',
    target_size=(IMAGE_WIDTH, IMAGE_Length),
    class_mode=None,
    batch_size=batch_size,
    seed=42,
    subset='training')

train_lowresimage_generator = image_datagen.flow_from_dataframe(
    data,
    x_col='low_res',
    target_size=(IMAGE_WIDTH, IMAGE_Length),
    class_mode=None,
    batch_size=batch_size,
    seed=42,
    subset='training')

val_hiresimage_generator = image_datagen.flow_from_dataframe(
    data,
    x_col='high_res',
    target_size=(IMAGE_WIDTH, IMAGE_Length),
    class_mode=None,
    batch_size=batch_size,
    seed=42,
    subset='validation')

val_lowresimage_generator = image_datagen.flow_from_dataframe(
    data,
    x_col='low_res',
    target_size=(IMAGE_WIDTH, IMAGE_Length),
    class_mode=None,
    batch_size=batch_size,
    seed=42,
    subset='validation')

# train_generator = zip(train_lowresimage_generator, train_hiresimage_generator)
train_generator = zip(train_lowresimage_generator, train_hiresimage_generator)
val_generator = zip(val_lowresimage_generator, val_hiresimage_generator)


def imageGenerator(train_generator):
    for (low_res, hi_res) in train_generator:
        yield (low_res, hi_res)


# n = 0
# for i, m in train_generator:
#     img, out = i, m
#
#     if n < 5:
#         fig, axs = plt.subplots(1, 2, figsize=(20, 5))
#         axs[0].imshow(img[0])
#         axs[0].set_title('Low Resolution Image')
#         axs[1].imshow(out[0])
#         axs[1].set_title('High Resolution Image')
#         plt.show()
#         n += 1
#     else:
#         break


# input_img = Input(shape=(IMAGE_WIDTH, IMAGE_Length, 3), name='encoder_input')
# l1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
# l2 = Conv2D(64, (3, 3), padding='same', activation='relu')(l1)
# l3 = Conv2D(64, (3, 3), padding='same', activation='relu')(l2)
# l4 = MaxPooling2D(padding='same')(l3)
# l4 = Dropout(0.3)(l4)
# l5 = Conv2D(128, (3, 3),  padding='same', activation='relu')(l4)
# l6 = Conv2D(128, (3, 3), padding='same', activation='relu')(l5)
# l7 = MaxPooling2D(padding='same')(l6)
# l8 = Conv2D(256, (3, 3), padding='same', activation='relu')(l7)
#
# l9 = UpSampling2D()(l8)
#
# l10 = Conv2D(128, (3, 3), padding='same', activation='relu')(l9)
# l11 = Conv2D(128, (3, 3), padding='same', activation='relu')(l10)
#
# l12 = add([l6, l11])
# l13 = UpSampling2D()(l12)
# l14 = Conv2D(64, (3, 3), padding='same', activation='relu')(l13)
# l15 = Conv2D(64, (3, 3), padding='same', activation='relu')(l14)
# l16 = Conv2D(64, (3, 3), padding='same', activation='relu')(l15)
#
# l17 = add([l16, l3])
#
# decoded = Conv2D(3, (3, 3), padding='same', activation='relu')(l17)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 2

encoder_inputs = Input(shape=(IMAGE_WIDTH, IMAGE_Length, 3), name='encoder_input')
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(100 * 150 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((100, 150, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


input_img = Input(shape=(IMAGE_WIDTH, IMAGE_Length, 3))

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

# autoencoder = Model(input_img, decoded)
# autoencoder = tf.keras.models.load_model('autoencoder_change.h5')
# autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#
# autoencoder.summary()

train_samples = train_hiresimage_generator.samples
val_samples = val_hiresimage_generator.samples

train_img_gen = imageGenerator(train_generator)
val_image_gen = imageGenerator(val_generator)

model_path = "autoencoder_VAE.h5"
checkpoint = ModelCheckpoint(model_path,
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

hist = vae.fit(train_lowresimage_generator,
               steps_per_epoch=train_samples // batch_size,
               validation_data=val_image_gen,
               validation_steps=val_samples // batch_size,
               epochs=30, callbacks=[earlystop, checkpoint, learning_rate_reduction])

plt.figure(figsize=(20, 8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

n = 15
for i, m in val_generator:
    img, mask = i, m
    sr1 = vae.predict(img)
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
