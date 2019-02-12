import keras
from keras.applications import InceptionV3
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
import sys

input = sys.argv[1]
output = sys.argv[2]

inception = InceptionV3(weights='imagenet', include_top=True)
model = Model(inputs=inception.input, outputs=inception.layers[-2].output)

batch_size = 100
image_size = 299
embedding_size = 2048
datagen = ImageDataGenerator(rescale=1/255.)
gen = datagen.flow_from_directory(input, target_size=(image_size, image_size), batch_size=batch_size, shuffle=False)
n = len(gen)

with h5py.File(output, 'w') as f:
    f.create_dataset('filenames', data=np.array(gen.filenames, dtype='S'))
    embeddings = f.create_dataset('embeddings', (len(gen.filenames), embedding_size), dtype=float)
    for i in tqdm(range(n)):
        images, labels = gen[i]
        output = model.predict(images, verbose=0)
        embeddings[i * batch_size : i * batch_size + len(output)] = output
