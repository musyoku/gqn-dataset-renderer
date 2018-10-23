import argparse
import colorsys
import math
import random
import time
import cv2

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm
from PIL import Image

import gqn
import rtx


class GeometryType:
    box = 1
    shpere = 2
    cylinder = 3
    cone = 4


geometry_type_array = [
    GeometryType.box,
    GeometryType.shpere,
    GeometryType.cylinder,
    GeometryType.cone,
]


def load_mnist_images():
    import chainer
    train, test = chainer.datasets.get_mnist()
    image_array = []
    for k in range(100):
        image = train[k][0]
        image = image.reshape((28, 28, 1))
        image = np.repeat(image, 3, axis=2)
        image_array.append(image)
    return image_array


def load_texture_image(filename):
    image = Image.open(filename)
    image = image.convert("RGB")
    texture = np.array(image, dtype=np.float32) / 255
    return texture


def build_dice(mnist_images):
    assert len(mnist_images) == 6

    dice = rtx.ObjectGroup()

    # 1
    geometry = rtx.PlainGeometry(1, 1)
    geometry.set_position((0, 0, 0.5))
    material = rtx.LambertMaterial(0.95)
    mapping = build_mapping(mnist_images[0])
    face = rtx.Object(geometry, material, mapping)
    dice.add(face)

    # 2
    geometry = rtx.PlainGeometry(1, 1)
    geometry.set_rotation((0, -math.pi, 0))
    geometry.set_position((0, 0, -0.5))
    material = rtx.LambertMaterial(0.95)
    mapping = build_mapping(mnist_images[1])
    face = rtx.Object(geometry, material, mapping)
    dice.add(face)

    # 3
    geometry = rtx.PlainGeometry(1, 1)
    geometry.set_rotation((0, math.pi / 2, 0))
    geometry.set_position((0.5, 0, 0))
    material = rtx.LambertMaterial(0.95)
    mapping = build_mapping(mnist_images[2])
    face = rtx.Object(geometry, material, mapping)
    dice.add(face)

    # 4
    geometry = rtx.PlainGeometry(1, 1)
    geometry.set_rotation((0, -math.pi / 2, 0))
    geometry.set_position((-0.5, 0, 0))
    material = rtx.LambertMaterial(0.95)
    mapping = build_mapping(mnist_images[3])
    face = rtx.Object(geometry, material, mapping)
    dice.add(face)

    # 5
    geometry = rtx.PlainGeometry(1, 1)
    geometry.set_rotation((math.pi / 2, 0, 0))
    geometry.set_position((0, -0.5, 0))
    material = rtx.LambertMaterial(0.95)
    mapping = build_mapping(mnist_images[4])
    face = rtx.Object(geometry, material, mapping)
    dice.add(face)

    # 5
    geometry = rtx.PlainGeometry(1, 1)
    geometry.set_rotation((-math.pi / 2, 0, 0))
    geometry.set_position((0, 0.5, 0))
    material = rtx.LambertMaterial(0.95)
    mapping = build_mapping(mnist_images[5])
    face = rtx.Object(geometry, material, mapping)
    dice.add(face)

    dice.set_scale((2, 2, 2))

    return dice


def build_mapping(texture, wall_aspect_ratio=1.0, scale=1.0):
    aspect_ratio = texture.shape[1] / texture.shape[0]
    uv_coordinates = np.array(
        [
            [0, 1 / scale],
            [wall_aspect_ratio / aspect_ratio / scale, 1 / scale],
            [0, 0],
            [wall_aspect_ratio / aspect_ratio / scale, 0],
        ],
        dtype=np.float32)
    mapping = rtx.TextureMapping(texture, uv_coordinates)
    return mapping


def build_scene(mnist_image_array,
                wall_texture_filename_array,
                floor_texture_filename_array,
                grid_size=8):
    assert len(mnist_image_array) == 6

    wall_height = 3
    eps = 0.1
    scene = rtx.Scene(ambient_color=(0.5, 1, 1))

    texture = load_texture_image(random.choice(wall_texture_filename_array))
    mapping = build_mapping(texture, grid_size / wall_height)

    # 1
    geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
    geometry.set_rotation((0, 0, 0))
    geometry.set_position((0, 0, -grid_size / 2))
    material = rtx.LambertMaterial(0.95)
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    # 2
    geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
    geometry.set_rotation((0, -math.pi / 2, 0))
    geometry.set_position((grid_size / 2, 0, 0))
    material = rtx.LambertMaterial(0.95)
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    # 3
    geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
    geometry.set_rotation((0, math.pi, 0))
    geometry.set_position((0, 0, grid_size / 2))
    material = rtx.LambertMaterial(0.95)
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    # 4
    geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
    geometry.set_rotation((0, math.pi / 2, 0))
    geometry.set_position((-grid_size / 2, 0, 0))
    material = rtx.LambertMaterial(0.95)
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    # floor
    geometry = rtx.PlainGeometry(grid_size + eps, grid_size + eps)
    geometry.set_rotation((-math.pi / 2, 0, 0))
    geometry.set_position((0, -wall_height / 2, 0))
    material = rtx.LambertMaterial(0.95)
    texture = load_texture_image(random.choice(floor_texture_filename_array))
    mapping = build_mapping(texture, scale=0.5)
    floor = rtx.Object(geometry, material, mapping)
    scene.add(floor)

    # Place lights

    ## Primary light
    primary_lights = rtx.ObjectGroup()
    geometry = rtx.SphereGeometry(2)
    geometry.set_position((grid_size / 2 - 1, 8, grid_size / 2 - 1))
    material = rtx.EmissiveMaterial(40, visible=False)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    light = rtx.Object(geometry, material, mapping)
    primary_lights.add(light)
    scene.add(primary_lights)

    ## Ambient light
    ambient_lights = rtx.ObjectGroup()

    geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
    geometry.set_rotation((0, 0, 0))
    geometry.set_position((0, 0, -grid_size / 2))
    material = rtx.EmissiveMaterial(1, visible=False)
    wall = rtx.Object(geometry, material, mapping)
    ambient_lights.add(wall)

    geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
    geometry.set_rotation((0, -math.pi / 2, 0))
    geometry.set_position((grid_size / 2, 0, 0))
    material = rtx.EmissiveMaterial(1, visible=False)
    wall = rtx.Object(geometry, material, mapping)
    ambient_lights.add(wall)

    geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
    geometry.set_rotation((0, math.pi, 0))
    geometry.set_position((0, 0, grid_size / 2))
    material = rtx.EmissiveMaterial(1, visible=False)
    wall = rtx.Object(geometry, material, mapping)
    ambient_lights.add(wall)

    geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
    geometry.set_rotation((0, math.pi / 2, 0))
    geometry.set_position((-grid_size / 2, 0, 0))
    material = rtx.EmissiveMaterial(1, visible=False)
    wall = rtx.Object(geometry, material, mapping)
    ambient_lights.add(wall)

    ambient_lights.set_position((0, wall_height, 0))
    scene.add(ambient_lights)

    # Place a dice
    dice = build_dice(mnist_image_array)
    spread = grid_size / 3
    dice.set_position((spread * random.uniform(-1, 1), 1 - wall_height / 2,
                       spread * random.uniform(-1, 1)))
    scene.add(dice)

    return scene


def main():
    # Set GPU device
    rtx.set_device(args.gpu_device)

    # Texture
    wall_texture_filename_array = [
        "textures/wall_texture_5.jpg",
    ]
    floor_texture_filename_array = [
        "textures/floor_texture_1.png",
    ]

    # Load MNIST images
    mnist_image_array = load_mnist_images()

    screen_width = args.image_size
    screen_height = args.image_size

    # Setting up a raytracer
    rt_args = rtx.RayTracingArguments()
    rt_args.num_rays_per_pixel = 1024
    rt_args.max_bounce = 3
    rt_args.supersampling_enabled = True
    rt_args.next_event_estimation_enabled = True

    cuda_args = rtx.CUDAKernelLaunchArguments()
    cuda_args.num_threads = 64
    cuda_args.num_rays_per_thread = 16

    renderer = rtx.Renderer()
    render_buffer = np.zeros(
        (screen_height, screen_width, 3), dtype=np.float32)

    perspective_camera = rtx.PerspectiveCamera(
        fov_rad=math.pi / 3, aspect_ratio=screen_width / screen_height)
    orthogonal_camera = rtx.OrthographicCamera()

    plt.tight_layout()

    scene = build_scene(
        random.sample(mnist_image_array, 6), wall_texture_filename_array,
        floor_texture_filename_array)

    view_radius = 3
    rotation = 0

    fig = plt.figure(figsize=(6, 3))
    axis_perspective = fig.add_subplot(1, 2, 1)
    axis_orthogonal = fig.add_subplot(1, 2, 2)
    ims = []

    for _ in range(args.num_views_per_scene):
        eye = (view_radius * math.cos(rotation), -0.125,
               view_radius * math.sin(rotation))
        center = (0, 0, 0)
        perspective_camera.look_at(eye, center, up=(0, 1, 0))

        renderer.render(scene, perspective_camera, rt_args, cuda_args,
                        render_buffer)
        image = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
        image = np.uint8(image * 255)
        image = cv2.bilateralFilter(image, 3, 25, 25)
        im1 = axis_perspective.imshow(
            image, interpolation="none", animated=True)

        eye = (view_radius * math.cos(rotation),
               view_radius * math.sin(math.pi / 6),
               view_radius * math.sin(rotation))
        center = (0, 0, 0)
        orthogonal_camera.look_at(eye, center, up=(0, 1, 0))

        renderer.render(scene, orthogonal_camera, rt_args, cuda_args,
                        render_buffer)
        image = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
        image = np.uint8(image * 255)
        image = cv2.bilateralFilter(image, 3, 25, 25)
        im2 = axis_orthogonal.imshow(
            image, interpolation="none", animated=True)
        ims.append([im1, im2])

        plt.pause(1e-8)

        rotation += math.pi / 36

    ani = animation.ArtistAnimation(
        fig, ims, interval=1 / 24, blit=True, repeat_delay=0)

    ani.save('mnist_dice.gif', writer="imagemagick")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--num-views-per-scene", "-k", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-objects", "-objects", type=int, default=3)
    parser.add_argument("--num-colors", "-colors", type=int, default=12)
    args = parser.parse_args()
    main()
