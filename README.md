# GQN Dataset Renderer

This is a Python library for constructing a dataset for GQN and CGQN. This renderer has two backend implementations available: **OpenGL** backend and **CUDA** backend. 
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

![rooms_rotate_object](https://user-images.githubusercontent.com/15250418/54522817-799ecc80-49b1-11e9-81de-8dccd8fd68b9.gif)

```
cd opengl
python3 rooms_ring_camera.py --output-directory rooms_ring_camera_no_object_rotation --anti-aliasing
```

```
cd opengl
python3 rooms_free_camera.py --output-directory rooms_free_camera_no_object_rotation --anti-aliasing
```

## MNIST Dice

![mnist_dice](https://user-images.githubusercontent.com/15250418/54579797-8ae6e800-4a48-11e9-8234-9059ae777d9d.gif)

```
cd opengl
python3 mnist_dice_ring_camera.py --output-directory mnist_dice_ring_camera_no_object_rotation --anti-aliasing
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

![rooms_rotate_object](https://user-images.githubusercontent.com/15250418/54522553-e5346a00-49b0-11e9-8149-221a18e68a05.gif)

```
cd cuda
python3 rooms_ring_camera.py --output-directory rooms_ring_camera_no_object_rotation --anti-aliasing
```

```
cd cuda
python3 rooms_free_camera.py --output-directory rooms_free_camera_no_object_rotation --anti-aliasing
```

## MNIST Dice

![rooms](https://user-images.githubusercontent.com/15250418/54581222-119ec380-4a4f-11e9-960b-db679e33723f.gif)

```
cd cuda
python3 mnist_dice_ring_camera.py --output-directory mnist_dice_ring_camera_no_object_rotation --anti-aliasing
```

# Textures

- https://github.com/deepmind/lab/tree/master/assets/textures/map/lab_games