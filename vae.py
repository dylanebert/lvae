import keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.utils import HDF5Matrix
from keras import metrics
from keras import optimizers
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import h5py
from tqdm import tqdm
import time
import pickle
import random
import sys
sys.path.append('scripts')
from config import config

class VAE():
    def __init__(self, config):
        input_size = 1024
        latent_size = config.latent_size
        self.model = config.model
        self.data = config.data
        for s in [self.model, self.data]:
            if not os.path.exists(s):
                os.makedirs(s)

        if input_size < latent_size * 4:
            print('Warning: First layer larger than input size')

        h1_size = min(latent_size * 4, input_size)
        h2_size = min(latent_size * 2, input_size)

        x = Input(shape=(input_size,))
        h1 = Dense(h1_size, activation='relu')(x)
        h2 = Dense(h2_size, activation='relu')(h1)
        z_mean = Dense(latent_size)(h2)
        z_stddev = Dense(latent_size)(h2)

        def sampling(args):
            z_mean, z_stddev = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_size), mean=0., stddev=1.0)
            return z_mean + K.exp(z_stddev) * epsilon

        z = Lambda(sampling, output_shape=(latent_size,))([z_mean, z_stddev])

        dec_h1 = Dense(h2_size, activation='relu')
        dec_h2 = Dense(h1_size, activation='relu')
        reconstr_x = Dense(input_size, activation='sigmoid')

        h1_dec = dec_h1(z)
        h2_dec = dec_h2(h1_dec)
        x_reconstr = reconstr_x(h2_dec)

        self.vae = Model(x, x_reconstr)

        xent_loss = input_size * metrics.binary_crossentropy(K.flatten(x), K.flatten(x_reconstr))
        kl_loss = -0.5 * K.sum(1 + z_stddev - K.square(z_mean) - K.exp(z_stddev), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        self.vae.add_loss(vae_loss)

        optimizer = optimizers.Adam(lr=.001)
        self.vae.compile(optimizer=optimizer)
        self.vae.summary()

        self.encoder = Model(x, z)

        _dec_x = Input(shape=(latent_size,))
        _h1_dec = dec_h1(_dec_x)
        _h2_dec = dec_h2(_h1_dec)
        _x_reconstr = reconstr_x(_h2_dec)
        self.decoder = Model(_dec_x, _x_reconstr)

    def load_weights(self):
        try:
            self.vae.load_weights(os.path.join(self.model, 'weights.h5'))
            print('Weights loaded')
            return True
        except:
            print('Failed to load weights')
            return False

    def train(self):
        self.load_weights()
        train_data = HDF5Matrix(os.path.join(self.data, 'train.h5'), 'embeddings')
        dev_data = HDF5Matrix(os.path.join(self.data, 'dev.h5'), 'embeddings')
        checkpoint_callback = keras.callbacks.ModelCheckpoint(os.path.join(self.model, 'weights.h5'), save_best_only=True, verbose=1)
        earlystopping_callback = keras.callbacks.EarlyStopping(verbose=1, patience=5)
        callbacks = [checkpoint_callback, earlystopping_callback]
        history = self.vae.fit(x=train_data, y=None, validation_data=(dev_data, None), epochs=999, shuffle='batch', callbacks=callbacks, batch_size=64, verbose=1)
        with open(os.path.join(self.model, 'history.p'), 'wb+') as o:
            pickle.dump(history.history, o)

    def encode(self):
        if not self.load_weights():
            return
        for s in ['train', 'dev', 'test']:
            print('Encoding {0} data'.format(s))
            dpath = os.path.join(self.data, s + '.h5')
            spath = os.path.join(self.model, s + '_encodings.h5')
            data = HDF5Matrix(dpath, 'embeddings')
            filenames = HDF5Matrix(dpath, 'filenames')
            labels = HDF5Matrix(dpath, 'labels')
            z = self.encoder.predict(data, batch_size=64, verbose=1)
            with h5py.File(spath, 'w') as f:
                f.create_dataset('encodings', data=z)
                f.create_dataset('filenames', data=filenames)
                f.create_dataset('labels', data=labels)

if __name__ == '__main__':
    model = VAE(config)
    model.train()
    model.encode()
