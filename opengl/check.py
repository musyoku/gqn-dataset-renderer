import os
import random
import argparse

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


def main():
    fig = plt.figure(figsize=(10, 10))
    image_filename_array = os.listdir(
        os.path.join(args.dataset_directory, "images"))
    while True:
        for filename in image_filename_array:
            image_array = np.load(
                os.path.join(args.dataset_directory, "images", filename))
            indices = np.random.choice(
                np.arange(image_array.shape[0]), replace=False, size=10 * 10)
            images = image_array[indices]
            images = images[:, 0, ...]
            images = images.reshape((10, 10, 64, 64, 3))
            images = images.transpose((0, 2, 1, 3, 4))
            images = images.reshape((10 * 64, 10 * 64, 3))

            plt.imshow(images, interpolation="none")
            plt.pause(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-directory", type=str, required=True)
    args = parser.parse_args()
    main()