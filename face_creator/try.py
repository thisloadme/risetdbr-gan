from keras.layers import Input, Flatten, Conv2D, Conv2DTranspose, Dense, LeakyReLU, BatchNormalization, Dropout, Reshape
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import cv2

x_train = []
folder = 'dataset'
H = 32
W = 32

for img_name in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # COLOR_BGR2RGB untuk berwarna, COLOR_BGR2GRAY untuk grayscale
    img = cv2.resize(img, (W,H), interpolation=cv2.INTER_LINEAR)
    np_array = np.array(img)
    x_train.append(np_array)

x_train = np.array(x_train)

x_train = x_train / 255 * 2 - 1

# flatten data
N, H, W = x_train.shape
D = H * W
x_train = x_train.reshape(-1, D)

# latent space dimensionality
latent_dim = 100

# generator model
def build_generator(latent_dim):
    i = Input(shape=(latent_dim,))
    # x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
    # x = BatchNormalization(momentum=0.8)(x)
    # x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
    # x = BatchNormalization(momentum=0.8)(x)
    # x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
    # x = BatchNormalization(momentum=0.8)(x)
    # x = Dense(D, activation='tanh')(x)
    x = Dense((64 * H * W), activation=LeakyReLU(alpha=0.2))(i)
    x = Reshape((H,W,64))(x)
    x = Conv2DTranspose(64, (4,4), padding='same', activation=LeakyReLU(alpha=0.2))(x)
    x = Conv2DTranspose(64, (4,4), padding='same', activation=LeakyReLU(alpha=0.2))(x)
    x = Conv2D(1, (H,W), activation='tanh', padding='same')(x)

    model = Model(i, x)
    return model

# discriminator model
def build_discriminator(img_size):
    i = Input(shape=(H,W,1))
    # x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
    # x = Dropout(0.4)(x)
    # x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
    # x = Dropout(0.4)(x)
    # x = Dense(1, activation='sigmoid')(x)
    # x = Reshape((H,W,1))(i)
    x = Conv2D(32, (3,3), strides=(2, 2), padding='same', activation=LeakyReLU(alpha=0.2))(i)
    x = Dropout(0.4)(x)
    x = Conv2D(32, (3,3), strides=(2, 2), padding='same', activation=LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(i, x)
    return model

# compile
discriminator = build_discriminator(D)
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002, 0.5),
    metrics=['accuracy']
)

generator = build_generator(latent_dim)

# input that represent noise sample from latent space
z = Input(shape=(latent_dim,))

# pass noise to generator to get image
img = generator(z)

# only generator trained
discriminator.trainable = False

# true output is fake, but we label them real
fake_pred = discriminator(img)

# combine model
combined_model = Model(z, fake_pred)
combined_model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002, 0.5)
)

# train gan
batch_size = 128
epochs = 10000
sample_period = 100

# create batch labels to use when call train on batch
ones = np.ones(batch_size)
zeros = np.zeros(batch_size)

# store loses
d_losses = []
g_losses = []

if not os.path.exists('gan_images_cnn'):
    os.makedirs('gan_images_cnn')

# function to generate random samples from generator
def sample_images(epoch):
    rows, cols = 2, 2
    noise = np.random.randn(rows * cols, latent_dim)
    imgs = generator.predict(noise, verbose=0)

    # rescale
    imgs = 0.5 * imgs + 0.5

    fig, axs = plt.subplots(rows, cols)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            axs[i,j].imshow(imgs[idx].reshape(H, W), cmap='gray_r')
            axs[i,j].axis('off')
            # cv2.imwrite('gan_images_cnn/sample_' + str(rows) + str(cols) + str(epoch) + '.png', imgs[idx].reshape(H, W))
            idx += 1
    fig.savefig("gan_images_cnn/%d.png" % epoch)
    plt.close()

# train loop
for epoch in range(epochs):
    ### train discriminator

    # select random batch
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]

    real_imgs = real_imgs.reshape(-1, H, W, 1)
    
    # generate fake images
    noise = np.random.randn(batch_size, latent_dim)
    fake_imgs = generator.predict(noise, verbose=0)

    d_loss_real, d_acc_real = discriminator.train_on_batch(real_imgs, ones)
    d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    d_acc = 0.5 * (d_acc_real + d_acc_fake)

    ### train generator
    noise = np.random.randn(batch_size, latent_dim)
    g_loss = combined_model.train_on_batch(noise, ones)

    # save losses
    d_losses.append(d_loss)
    g_losses.append(g_loss)

    if epoch % 10 == 0:
        print(f"epoch: {epoch}/{epochs}, d_loss: {d_loss:.2f}, d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}")
    
    if epoch % sample_period == 0:
        sample_images(epoch)

# plot
plt.plot(g_losses, label='g_losses')
plt.plot(d_losses, label='d_losses')
plt.show()