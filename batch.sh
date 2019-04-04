#!/bin/bash
python cnn.py -i /data/nlp/combined/train -o data/combined/train.h5
python cnn.py -i /data/nlp/combined/dev -o data/combined/dev.h5
python cnn.py -i /data/nlp/combined/test -o data/combined/test.h5
python vae.py -m model/combined -d data/combined
python vae.py -m model/combined64 -d data/combined --latent_size 64
