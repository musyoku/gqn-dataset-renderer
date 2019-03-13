import os
import random
import argparse

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


def main():
    image_filename_array = os.listdir(
        os.path.join(args.dataset_directory, "images"))
    while True:
        for filename in image_filename_array:
            image_array = np.load(
                os.path.join(args.dataset_directory, "images", filename))
            index = random.choice(list(range(image_array.shape[0])))
            image = image_array[index]

            plt.imshow(image[0], interpolation="none")
            plt.pause(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-directory",
        "-dataset",
        type=str,
        default="dataset_shepard_matzler_train")
    args = parser.parse_args()
    main()