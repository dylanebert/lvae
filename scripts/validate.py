from builder import mmid, imagenet
from datasets import eval_labels
import pandas as pd
import random
import tkinter as tk
from PIL import ImageTk, Image
import os
import numpy as np

labels = eval_labels
i = 0

def validate(label, path, value):
    with open('data/combined/validation.txt', 'a+') as f:
        f.write('\t'.join([label, path, str(value)]) + '\n')
    global i
    i += 1
    print(i)
    next()

def next():
    for widget in root.winfo_children():
        widget.destroy()
    label = random.choice(labels)
    options = []
    if label in mmid:
        options += mmid[label]
    if label in imagenet:
        options += imagenet[label]
    if len(options) == 0:
        return next()
    dir = random.choice(options)
    if dir[0] == 'n':
        files = os.listdir(os.path.join('/data/nlp/imagenet/train', dir))
        filepath = os.path.join('/data/nlp/imagenet/train', dir, random.choice(files))
    else:
        files = [file for file in os.listdir(os.path.join('/data/nlp/mmid', dir)) if file.endswith('.jpg')]
        filepath = os.path.join('/data/nlp/mmid', dir, random.choice(files))
    img = ImageTk.PhotoImage(Image.open(filepath))
    panel = tk.Label(root, image=img)
    panel.pack(side='top', fill='both', expand='yes')
    tk.Label(root, text=label).pack()
    tk.Button(root, text='Yes', command=lambda: validate(label, filepath, 1)).pack()
    tk.Button(root, text='No', command=lambda: validate(label, filepath, 0)).pack()
    root.mainloop()

root = tk.Tk()
next()
