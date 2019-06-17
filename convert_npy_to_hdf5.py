import os
import random
import argparse
import h5py

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


def main():
    try:
        os.mkdir(args.output_dataset_directory)
    except:
        pass

    filename_array = os.listdir(
        os.path.join(args.source_dataset_directory, "images"))
    for source_filename in filename_array:
        target_filename = source_filename.replace(".npy", ".h5")
        target_path = os.path.join(args.output_dataset_directory,
                                   target_filename)
        if os.path.isfile(target_path):
            continue
        images = np.load(
            os.path.join(args.source_dataset_directory, "images",
                         source_filename))
        viewpoints = np.load(
            os.path.join(args.source_dataset_directory, "viewpoints",
                         source_filename))

        with h5py.File(target_path, "w") as f:
            f.create_dataset("images", data=images)
            f.create_dataset("viewpoints", data=viewpoints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dataset-directory", "-s", type=str, required=True)
    parser.add_argument(
        "--output-dataset-directory", "-d", type=str, required=True)
    args = parser.parse_args()
    main()