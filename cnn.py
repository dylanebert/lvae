import keras
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
import argparse

def cnn(input, output, image_size):
    batch_size = 32
    datagen = ImageDataGenerator(rescale=1/255.)
    train_gen = datagen.flow_from_directory(os.path.join(input, 'train'), target_size=(image_size, image_size), batch_size=batch_size, follow_links=True)
    dev_gen = datagen.flow_from_directory(os.path.join(input, 'dev'), target_size=(image_size, image_size), batch_size=batch_size, follow_links=True)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='sigmoid')(x)
    y = Dense(train_gen.num_classes, activation='softmax')(x)
    model = Model(base_model.input, y)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(os.path.join(output, 'weights.h5'), save_best_only=True, verbose=1)
    earlystopping_callback = keras.callbacks.EarlyStopping(verbose=1, patience=5)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='logs/')
    callbacks = [checkpoint_callback, earlystopping_callback, tensorboard_callback]
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit_generator(train_gen, validation_data=dev_gen, epochs=999, callbacks=callbacks)

    model.load_weights(os.path.join(output, 'weights.h5'))
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=1e-4, momentum=.9), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit_generator(train_gen, validation_data=dev_gen, epochs=999, callbacks=callbacks)

    model.load_weights(os.path.join(output, 'weights.h5'))
    embedding_model = Model(model.input, model.layers[-2].output)
    embedding_model.summary()
    for type in ['train', 'dev', 'test']:
        gen = datagen.flow_from_directory(os.path.join(input, type), target_size=(image_size, image_size), batch_size=batch_size, follow_links=True, shuffle=False)
        classes = {v: k for k, v in gen.class_indices.items()}
        labels = [classes[k] for k in gen.classes]
        n = len(gen)
        with h5py.File(os.path.join(output, type + '.h5'), 'w') as f:
            f.create_dataset('filenames', data=np.array(gen.filenames, dtype='S'))
            f.create_dataset('labels', data=np.array(labels, dtype='S'))
            embeddings = f.create_dataset('embeddings', (len(gen.filenames), 1024), dtype=float)
            for i in tqdm(range(n)):
                images, labels = gen[i]
                x = embedding_model.predict(images, verbose=0)
                embeddings[i * batch_size : i * batch_size + len(x)] = x

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input directory', type=str, required=True)
parser.add_argument('-o', '--output', help='output directory', type=str, required=True)
parser.add_argument('-s', '--image_size', help='image size (default 299 for imagenet)', type=int, default=299)
args = parser.parse_args()

cnn(args.input, args.output, args.image_size)
