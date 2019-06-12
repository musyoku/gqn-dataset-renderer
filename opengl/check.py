import os
import random
import argparse
import h5py

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


def main():
    fig = plt.figure(figsize=(10, 10))
    filename_array = os.listdir(args.dataset_directory)
    while True:
        for filename in filename_array:
            with h5py.File(
                    os.path.join(args.dataset_directory, filename), "r") as f:
                images = f["images"].value
                indices = np.random.choice(
                    np.arange(images.shape[0]), replace=False, size=10)
                images = images[indices]
                images = images[:, :10, ...]
                images = images.reshape((10, 10, 64, 64, 3))
                images = images.transpose((0, 2, 1, 3, 4))
                images = images.reshape((10 * 64, 10 * 64, 3))

                plt.imshow(images, interpolation="none")
                plt.pause(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-directory", type=str, required=True)
    args = parser.parse_args()
    main()