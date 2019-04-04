import keras
from keras.applications import InceptionV3
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
import argparse

def cnn(input, output, image_size):
    inception = InceptionV3(weights='imagenet', include_top=True)
    model = Model(inputs=inception.input, outputs=inception.layers[-2].output)
    batch_size = 100
    datagen = ImageDataGenerator(rescale=1/255.)
    gen = datagen.flow_from_directory(input, target_size=(image_size, image_size), batch_size=100, shuffle=False, follow_links=True)
    n = len(gen)
    classes = {v: k for k, v in gen.class_indices.items()}
    labels = [classes[k] for k in gen.classes]

    with h5py.File(output, 'w') as f:
        f.create_dataset('filenames', data=np.array(gen.filenames, dtype='S'))
        f.create_dataset('labels', data=np.array(labels, dtype='S'))
        embeddings = f.create_dataset('embeddings', (len(gen.filenames), 2048), dtype=float)
        for i in tqdm(range(n)):
            images, labels = gen[i]
            output = model.predict(images, verbose=0)
            embeddings[i * batch_size : i * batch_size + len(output)] = output

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input directory (flow_from_directory)', type=str, required=True)
parser.add_argument('-o', '--output', help='output .h5 file', type=str, required=True)
parser.add_argument('-s', '--image_size', help='image size (default 299 for imagenet)', type=int, default=299)
args = parser.parse_args()

cnn(args.input, args.output, args.image_size)
