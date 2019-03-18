**Status**: Maintenance

# GQN Dataset Renderer

This is a Python library for constructing a dataset for GQN and CGQN.

At this time, this renderer has two backend implementations available: **OpenGL** backend and **CUDA** backend.

OpenGL renderer is built on top of a modified version of [pyrender](https://github.com/mmatl/pyrender).

# OpenGL

## Requirements

- Python 3
- Pillow
    - `pip3 install Pillow`
- tqdm
    - `pip3 install tqdm`
- Ubuntu
    - tested on Ubuntu 16.04 / 17.10
- GPU

Also, you need Chainer, PyTorch or Keras to download MNIST images.

## Installation

```
pip3 install -r requirements.txt
```

## Shepard-Metzler

![shepard_matzler](https://user-images.githubusercontent.com/15250418/54495487-92fb3680-4927-11e9-83be-125b669701db.gif)

```
cd opengl
python3 shepard_metzler.py  --num-cubes 7 --num-colors 10 --output-directory shepard_metzler_7_part --total-scenes 2000000
```

## Rooms

![rooms_discrete_position_rotate_object](https://user-images.githubusercontent.com/15250418/54495840-0bafc200-492b-11e9-998f-848f83f45579.gif)


```
python3 rooms_free_camera.py --output-directory rooms_ring_camera_no_object_rotation --anti-aliasing
```

# CUDA

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
cd cuda
make -j4
```

## Shepard-Metzler

![shepard_matzler](https://user-images.githubusercontent.com/15250418/54510705-cf637c80-4991-11e9-86da-73be9711adc0.gif)


```
cd cuda
python3 shepard_metzler.py  --num-cubes 7 --num-colors 10 --output-directory shepard_metzler_7_part --total-scenes 2000000
```

## Rooms

![rooms](https://user-images.githubusercontent.com/15250418/52397602-746c7900-2af9-11e9-8ea0-a7985a8bd05e.gif)

```
cd cuda
python3 rooms_ring_camera.py -objects 3 -colors 12 -k 10
```

## MNIST Dice

![mnist_dice](https://user-images.githubusercontent.com/15250418/52397600-746c7900-2af9-11e9-9aa7-1088341e0f16.gif)

```
cd cuda
python3 mnist_dice_ring_camera.py -k 10
```

## Using multiple GPUs

```
cd cuda
python3 rooms_ring_camera.py -objects 3 -colors 12 -k 5 -gpu 0 --total-observations 100000 --num-observations-per-file 2000
python3 rooms_ring_camera.py -objects 3 -colors 12 -k 5 -gpu 1 --total-observations 100000 --num-observations-per-file 2000 --initial-file-number 51
```

# Textures

- https://github.com/deepmind/lab/tree/master/assets/textures/map/lab_games