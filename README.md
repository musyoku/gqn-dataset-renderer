# GQN Dataset Renderer

This is a Python library for building a dataset for GQN and CGQN based on [CUDA raytracer](https://github.com/musyoku/python-rtx).

## Requirements

- Python 3
- OpenCV
    - `pip3 install opencv-python`
- Pillow
    - `pip3 install Pillow`
- tqdm
    - `pip3 install tqdm`
- pybind11
    - `pip3 install pybind11 --user`
- Ubuntu
    - tested on Ubuntu 16.04 / 17.10
- CUDA
    - tested on CUDA 9.1
- NVIDIA GPU
    - tested on GTX 1070 / 1080
- C++14 (gcc-6)

Also, you need Chainer, PyTorch or Keras to download MNIST images.

## Installation

```
make -j4
```

# Shepard-Metzler

![shepard_matzler](https://user-images.githubusercontent.com/15250418/52397603-746c7900-2af9-11e9-97b3-3a823799eaa6.gif)

```
python3 shepard_metzler.py -cubes 5 -colors 12 -k 15
```

# Rooms

![rooms](https://user-images.githubusercontent.com/15250418/52397602-746c7900-2af9-11e9-8ea0-a7985a8bd05e.gif)

```
python3 rooms_ring_camera.py -objects 3 -colors 12 -k 10
```

# MNIST Dice

![mnist_dice](https://user-images.githubusercontent.com/15250418/52397600-746c7900-2af9-11e9-9aa7-1088341e0f16.gif)

```
python3 mnist_dice_ring_camera.py -k 10
```

# Using multiple GPUs

```
python3 rooms_ring_camera.py -objects 3 -colors 12 -k 5 -gpu 0 --total-observations 100000 --num-observations-per-file 2000
python3 rooms_ring_camera.py -objects 3 -colors 12 -k 5 -gpu 1 --total-observations 100000 --num-observations-per-file 2000 --initial-file-number 51
```

# Textures

- https://github.com/deepmind/lab/tree/master/assets/textures/map/lab_games