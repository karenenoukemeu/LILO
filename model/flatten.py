import os
import numpy as np

from PIL import Image

file_name = "{name}_{num:03d}.png"
data_dir  = os.getenv('LILO_DATA')
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

def load_data(set_size=data_size):
    train_size = int(set_size * 0.8)
    valid_size = set_size - train_size
    return (process_data(1, train_size), process_data(1 + train_size, valid_size + train_size))
    
