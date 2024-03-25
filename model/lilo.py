#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM
#SBATCH -t 5:00:00
#SBATCH --ntasks-per-node=128

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from flatten import load_data

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