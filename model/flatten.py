#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM
#SBATCH -t 5:00:00
#SBATCH --ntasks-per-node=128

import pandas as pd
import numpy as np

from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

file_name = "{name}_{num:03d}.png"
data_dir  = "../training/"
data_name = "MoonRocks"
data_size = 1000

def process_image(image):
    # Get the benchmark from metadata
    image.load()
    str_loc = image.info['Note'].strip("()").split(", ")
    loc = tuple(map(float, str_loc))

    # Flatten the image out
    data = np.asarray(image)
    data = data.reshape(1024 ** 2)

    return (data, loc)

def process_dir(path, batch_name, lower_bound, upper_bound):
    x_full = []
    y_full = []
    for i in range(lower_bound, upper_bound + 1):
        if not path.endswith("/"):
            path = path + "/"
        abs_path = path + file_name.format(name=batch_name, num=i)
        img = Image.open(abs_path).convert('L')
        x, y = process_image(img)
        x_full.append(x)
        y_full.append(y)
    
    return (np.array(x_full), np.array(y_full))

def process_data(start, end):
    return process_dir(data_dir, data_name, start, end)

def load_data():
    train_size = int(data_size * 0.8)
    valid_size = data_size - train_size
    return (process_data(1, train_size), process_data(1 + train_size, valid_size + train_size))
    


(x_train, y_train), (x_valid, y_valid) = load_data()

# x_train, y_train = process_dir(data_dir, data_name, 1, 20)
stencil = "{}: {}, {}\n\tPic Max: {}\tLoc Max: {}"
print("---------- Data Before Normalization ----------")
print(stencil.format("  Training Data", x_train.shape, y_train.shape, x_train.max(), y_train.max()))
print(stencil.format("Validation Data", x_valid.shape, y_valid.shape, x_valid.max(), y_valid.max()))

x_train = x_train / 255
x_valid = x_valid / 255

y_train = y_train / 360
y_valid = y_valid / 360

print("\n---------- Data After Normalization ----------")
print(stencil.format("  Training Data", x_train.shape, y_train.shape, x_train.max(), y_train.max()))
print(stencil.format("Validation Data", x_valid.shape, y_valid.shape, x_valid.max(), y_valid.max()))

model = Sequential()

model.add(Dense(units=512, activation='relu', input_shape=(1024 ** 2,)))
model.add(Dense(units = 512, activation='relu'))
model.add(Dense(units = 2, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history = model.fit(
    x_train, y_train, epochs=5, verbose=1, validation_data=(x_valid, y_valid)
)
