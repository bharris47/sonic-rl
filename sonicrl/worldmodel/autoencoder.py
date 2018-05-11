import json
import os
import random
from argparse import ArgumentParser

import cv2
import keras.backend as K
import numpy as np
from keras import Input, Model, metrics
from keras.callbacks import Callback, TensorBoard
from keras.layers import Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose

from sonicrl.environments import get_environments


class MultiModelCheckpoint(Callback):
    def __init__(self, models, filepath):
        self._models = models
        self._filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        for model_name, model in self._models.items():
            filepath = self._filepath.format(model=model_name, epoch=epoch + 1, **logs)
            model.save(filepath, overwrite=True)


def autoencoder(image_shape, filters=64, kernel_size=3, latent_dims=64, intermediate_dims=128, epsilon_std=1.0):
    x = Input(shape=image_shape)
    height, width, channels = image_shape
    conv_1 = Conv2D(filters,
                    kernel_size=kernel_size,
                    padding='same', activation='relu',
                    strides=(2, 2))(x)
    conv_2 = Conv2D(filters,
                    kernel_size=kernel_size,
                    padding='same', activation='relu')(conv_1)
    conv_3 = Conv2D(filters * 2,
                    kernel_size=kernel_size,
                    padding='same', activation='relu',
                    strides=(2, 2))(conv_2)
    flat = Flatten()(conv_3)
    hidden = Dense(intermediate_dims, activation='relu')(flat)

    z_mean = Dense(latent_dims)(hidden)
    z_log_var = Dense(latent_dims)(hidden)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dims),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_log_var])

    encoder = Model(x, [z_mean, z_log_var, z], name='encoder')

    # we instantiate these layers separately so as to reuse them later
    decoder_hid = Dense(intermediate_dims, activation='relu')
    intermediate_height = height // 4
    intermediate_width = width // 4
    decoder_upsample = Dense(filters * intermediate_height * intermediate_width, activation='relu')

    decoder_reshape = Reshape((intermediate_height, intermediate_width, filters))
    decoder_deconv_1 = Conv2DTranspose(filters * 2,
                                       kernel_size=kernel_size,
                                       padding='same',
                                       activation='relu')
    decoder_deconv_2 = Conv2DTranspose(filters,
                                       kernel_size=kernel_size,
                                       strides=(2, 2),
                                       padding='same',
                                       activation='relu')
    decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                              kernel_size=kernel_size,
                                              strides=(2, 2),
                                              padding='same',
                                              activation='relu')
    decoder_mean_squash = Conv2D(channels,
                                 kernel_size=kernel_size,
                                 padding='same',
                                 activation='sigmoid')

    decoder_input = Input(shape=(latent_dims,))
    decoder_features = decoder_hid(decoder_input)
    decoder_features = decoder_upsample(decoder_features)
    decoder_features = decoder_reshape(decoder_features)
    decoder_features = decoder_deconv_1(decoder_features)
    decoder_features = decoder_deconv_2(decoder_features)
    decoder_features = decoder_deconv_3_upsamp(decoder_features)
    decoded_image = decoder_mean_squash(decoder_features)
    decoder = Model(decoder_input, decoded_image, name='decoder')

    # instantiate VAE model
    _, _, z_sampling = encoder(x)
    vae_output = decoder(z_sampling)
    vae = Model(x, vae_output, name='vae')

    # Compute VAE loss
    xent_loss = height * width * metrics.binary_crossentropy(
        K.flatten(x),
        K.flatten(vae_output))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    vae.add_loss(vae_loss)

    vae.compile(optimizer='rmsprop')
    vae.summary()

    return vae, encoder, decoder


def chunks(items, chunk_size):
    chunk = []
    for item in items:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def image_path_generator(image_paths, shuffle=False):
    while True:
        if shuffle:
            random.shuffle(image_paths)
        yield from image_paths


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = image.astype(np.float32)
    image /= 255.
    return image


def image_generator(image_paths, shuffle=False, batch_size=32):
    paths = image_path_generator(image_paths, shuffle)
    images = map(load_image, paths)
    batches = chunks(images, batch_size)
    for batch in batches:
        image_batch = np.array(batch)
        yield image_batch, None


def train_val_split(samples, train_envs, image_directory):
    train_envs = {(env['game'], env['state']) for env in train_envs}
    train_paths = []
    val_paths = []
    for sample in samples:
        path = os.path.join(image_directory, sample['image_id'])
        env = (sample['game'], sample['state'])
        if env in train_envs:
            train_paths.append(path)
        else:
            val_paths.append(path)
    return train_paths, val_paths


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--samples')
    parser.add_argument('--image-directory')
    parser.add_argument('--train-environments')
    parser.add_argument('--checkpoint-directory', default='checkpoints')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    with open(args.samples) as samples_file:
        samples = list(map(json.loads, samples_file))

    train_environments = get_environments(args.train_environments)

    train_paths, val_paths = train_val_split(samples, train_environments, args.image_directory)

    train_generator = image_generator(train_paths, shuffle=True, batch_size=args.batch_size)
    train_steps = 10000

    val_generator = image_generator(train_paths, batch_size=args.batch_size)
    val_steps = len(val_paths) // args.batch_size

    vae, encoder, decoder = autoencoder((224, 320, 3))
    models = {'vae': vae, 'encoder': encoder, 'decoder': decoder}
    filepath = os.path.join(args.checkpoint_directory, '{model}.{epoch:02d}-{val_loss:.6f}.hdf5')

    vae.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=100,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=[
            TensorBoard(),
            MultiModelCheckpoint(models, filepath)
        ]
    )
