import tensorflow as tf
import numpy as np
import pandas as pd
from keras import layers
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2 as cv
from PIL import Image

strategy = tf.distribute.get_strategy()

# paths
cur_file = os.getcwd()
base_directory = cur_file + '/Image Super Resolution - Unsplash'
hires_folder = os.path.join(base_directory, 'high res')
lowres_folder = os.path.join(base_directory, 'low res')
hr_img_path = hires_folder
lr_img_path = lowres_folder
image_data_path = base_directory + '/image_data.csv'

# read csv
img_data_df = pd.read_csv(image_data_path)
img_data_df.head()

def load_img(path,is_hr=False,shape=(128,192)):
    img = tf.keras.utils.load_img(path)
    img = tf.keras.utils.img_to_array(img)
    img = tf.image.resize(img,shape)
    if is_hr:
        img = img
    else:
        img = img / 255.0
    return img.numpy()

lr_imgs = np.array([load_img(os.path.join(lr_img_path,i)) for i in tqdm(img_data_df['low_res'][:300])])
hr_imgs = np.array([load_img(os.path.join(hr_img_path,i),True,(512,768)) for i in tqdm(img_data_df['high_res'][:300])])

def convblock(in_layer,filters,stride=1):
    # weight initializer
    ini = tf.keras.initializers.RandomNormal(stddev=0.02)
    x = layers.Conv2D(filters,(3,3),strides=stride,padding='same',kernel_initializer=ini)(in_layer)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def build_discriminator(image_shape):
    # weight initializer
    ini = tf.keras.initializers.RandomNormal(stddev=0.02)
    # input
    input_img = layers.Input(shape=image_shape)
    d = layers.Conv2D(64, (3, 3), strides=1, padding='same', kernel_initializer=ini)(input_img)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = convblock(d, 64, 2)
    d = convblock(d, 128)
    d = convblock(d, 128, 2)
    d = convblock(d, 256)
    d = convblock(d, 256, 2)
    d = convblock(d, 512)
    d = convblock(d, 512, 2)
    d = layers.Dense(1024)(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Dense(1)(d)
    d = layers.Activation('sigmoid')(d)
    model = tf.keras.models.Model(inputs=input_img, outputs=d)
    return model

def residual_block(in_layer,filters,stride=1):
    # weight initializer
    ini = tf.keras.initializers.RandomNormal(stddev=0.02)
    rb = layers.Conv2D(filters,(3,3),padding='same',strides=stride,kernel_initializer=ini)(in_layer)
    rb = layers.BatchNormalization()(rb)
    rb = layers.PReLU()(rb)
    rb = layers.Conv2D(filters,(3,3),padding='same',strides=stride,kernel_initializer=ini)(rb)
    rb = layers.BatchNormalization()(rb)
    rb = layers.Add()([in_layer,rb])
    return rb

def upsample(in_layer):
    ini = tf.keras.initializers.RandomNormal(stddev=0.02)
    x = layers.Conv2D(256,(3,3),strides=1,kernel_initializer=ini,padding='same')(in_layer)
    x = layers.UpSampling2D()(x)
    x = layers.PReLU(shared_axes=[1,2])(x)
    return x


def build_generator(image_shape, n_res=16):
    # weight initializer
    ini = tf.keras.initializers.RandomNormal(stddev=0.02)

    input_img = layers.Input(shape=image_shape)
    g = layers.Conv2D(64, (9, 9), strides=1, padding='same', kernel_initializer=ini)(input_img)
    g = layers.PReLU(shared_axes=[1, 2])(g)
    g1 = g
    for _ in range(n_res):
        g = residual_block(g, 64)
    g = layers.Conv2D(64, (3, 3), padding='same', strides=1, kernel_initializer=ini)(g)
    g = layers.BatchNormalization()(g)
    g = layers.Add()([g1, g])
    g = upsample(g)
    g = upsample(g)
    g = layers.Conv2D(3, (9, 9), strides=1, padding='same', kernel_initializer=ini)(g)
    return tf.keras.models.Model(input_img, g)

def vgg_model(hr_shape):
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False,weights='imagenet',input_shape=hr_shape)
    out = tf.keras.layers.Rescaling(1/12.75)(vgg.layers[15].output)
    return tf.keras.models.Model(vgg.inputs,out)


def composite_model(gen, disc, vgg, lr_shape, hr_shape):
    lr_in = layers.Input(shape=lr_shape)
    hr_in = layers.Input(shape=hr_shape)
    gen_img = gen(lr_in)
    vgg_feature = vgg(gen_img)  # content loss

    disc.trainable = False
    validity = disc(gen_img)  # adversarial loss
    return tf.keras.models.Model(inputs=[lr_in, hr_in], outputs=[validity, vgg_feature])

# build models
lr_shape = lr_imgs[0].shape
hr_shape = hr_imgs[0].shape

with strategy.scope():
# lr image -> hr image
    gen_model = build_generator(lr_shape)
    gen_model = tf.keras.models.load_model(cur_file + '/save_epoch_128.h5')
# validate lr and hr
    disc = build_discriminator(hr_shape)
    disc.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.9),
            loss='mse',metrics=['accuracy'])
# vgg for feature extraction
    vgg = vgg_model(hr_shape)
    vgg.trainable = False

# composite model
    composite = composite_model(gen_model,disc,vgg,lr_shape,hr_shape)
    composite.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9),
                 loss=['mse','mse'],
                 loss_weights=[1e-3,1])

batch_size = 1
train_lr_batch = []
train_hr_batch = []
for i in tqdm(range(int(len(lr_imgs)/batch_size))):
    start = i * batch_size
    end = start + batch_size
    train_lr_batch.append(lr_imgs[start:end])
    train_hr_batch.append(hr_imgs[start:end])

test = load_img(os.path.join(lowres_folder, '3_6.jpg'))
np.expand_dims(test,axis=0).shape

pred = gen_model.predict(np.expand_dims(test,axis=0))
pred = np.squeeze(pred)
new_pred = pred/255
new_pred = np.maximum(new_pred, 0)
new_pred = np.minimum(new_pred, 255)
cv.imwrite(f'test_img/test_{100}.jpg', pred)

fig, axs = plt.subplots(1 , 2, figsize=(10,4))
axs[0].imshow(test)
axs[0].set_title('Low Resolution Image')
axs[1].imshow(new_pred)
axs[1].set_title('Predicted High Resolution Image')
plt.show()

with strategy.scope():
    epochs = 10
    for e in range(epochs):
        disc_out = disc.output_shape
        patch_shape = (disc_out[1], disc_out[2], disc_out[3])
        fake_label = np.zeros((batch_size, *patch_shape))
        real_label = np.ones((batch_size, *patch_shape))
        g_losses = []
        d_losses = []

        for b in tqdm(range(len(train_lr_batch))):
            lr_img_ap = np.array(train_lr_batch[b])
            hr_img_ap = np.array(train_hr_batch[b])

            # generate fake image using generator
            fake_img = gen_model.predict_on_batch(lr_img_ap)
            # train the discriminator to distinguish between real and fake
            disc.trainable = True
            d_gen_loss = disc.train_on_batch(fake_img, fake_label)
            d_real_loss = disc.train_on_batch(hr_img_ap, real_label)

            avg_d_loss = np.add(d_gen_loss, d_real_loss) * 0.5
            d_losses.append(avg_d_loss)
            disc.trainable = False
            # VGG image feature
            image_feature = vgg.predict(hr_img_ap)

            # train generator using the composite model
            g_loss, _, _ = composite.train_on_batch([lr_img_ap, hr_img_ap], [real_label, image_feature])

            g_losses.append(g_loss)

        g_losses = np.array(g_losses)
        d_losses = np.array(d_losses)

        g_loss = np.sum(g_losses, axis=0) / len(g_losses)
        d_loss = np.sum(d_losses, axis=0) / len(d_losses)

        print(f'epochs :: {e} d_loss :: {d_loss} g_loss :: {g_loss}')
        if (e + 1) % 2 == 0:
            gen_model.save('save_epoch_128.h5')
            print('Model Saved')

        # if (e+1) % 1 == 0:
        #     pred = gen_model.predict(np.expand_dims(test,axis=0))
        #     pred = np.squeeze(pred)
        #     new_pred = pred/255
        #     new_pred = np.maximum(new_pred, 0)
        #     new_pred = np.minimum(new_pred, 255)
        #     cv.imwrite(f'test_img/test_{e}.jpg', pred)
        #
        #     plt.subplot(1,2,1)
        #     plt.imshow(new_pred)
        #     plt.axis('off')
        #     plt.subplot(1,2,2)
        #     plt.imshow(test)
        #     plt.axis('off')
        #     plt.show()

print('Finish')