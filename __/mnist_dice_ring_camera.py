import argparse
import colorsys
import math
import random
import time
import cv2

import matplotlib.pyplot as plt
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

    wall_height = grid_size / 3
    scene = rtx.Scene(ambient_color=(0.5, 1, 1))

    texture = load_texture_image(random.choice(wall_texture_filename_array))
    mapping = build_mapping(texture, grid_size / wall_height)

    # Place walls
    ## 1
    geometry = rtx.PlainGeometry(grid_size, wall_height)
    geometry.set_rotation((0, 0, 0))
    geometry.set_position((0, 0, -grid_size / 2))
    material = rtx.LambertMaterial(0.95)
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    ## 2
    geometry = rtx.PlainGeometry(grid_size, wall_height)
    geometry.set_rotation((0, -math.pi / 2, 0))
    geometry.set_position((grid_size / 2, 0, 0))
    material = rtx.LambertMaterial(0.95)
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    ## 3
    geometry = rtx.PlainGeometry(grid_size, wall_height)
    geometry.set_rotation((0, math.pi, 0))
    geometry.set_position((0, 0, grid_size / 2))
    material = rtx.LambertMaterial(0.95)
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    ## 4
    geometry = rtx.PlainGeometry(grid_size, wall_height)
    geometry.set_rotation((0, math.pi / 2, 0))
    geometry.set_position((-grid_size / 2, 0, 0))
    material = rtx.LambertMaterial(0.95)
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    # floor
    geometry = rtx.PlainGeometry(grid_size, grid_size)
    geometry.set_rotation((-math.pi / 2, 0, 0))
    geometry.set_position((0, -wall_height / 2, 0))
    material = rtx.LambertMaterial(0.95)
    texture = load_texture_image(random.choice(floor_texture_filename_array))
    mapping = build_mapping(texture, scale=0.5)
    floor = rtx.Object(geometry, material, mapping)
    scene.add(floor)

    # Place a light
    geometry = rtx.SphereGeometry(2)
    spread = grid_size / 2 - 1
    geometry.set_position((spread * random.uniform(-1, 1), 8,
                           spread * random.uniform(-1, 1)))
    material = rtx.EmissiveMaterial(20, visible=False)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    light = rtx.Object(geometry, material, mapping)
    scene.add(light)

    # Place a dice
    dice = build_dice(mnist_image_array)
    spread = grid_size / 3
    dice.set_position((spread * random.uniform(-1, 1), 1 - wall_height / 2,
                       spread * random.uniform(-1, 1)))
    dice.set_rotation((0, random.uniform(0, math.pi * 2), 0))
    scene.add(dice)

    return scene


def main():
    # Set GPU device
    rtx.set_device(args.gpu_device)

    # Texture
    wall_texture_filename_array = [
        "textures/lg_style_01_wall_cerise_d.tga",
        "textures/lg_style_01_wall_green_bright_d.tga",
        "textures/lg_style_01_wall_red_bright_d.tga",
        "textures/lg_style_02_wall_yellow_d.tga",
        "textures/lg_style_03_wall_orange_bright_d.tga",
    ]
    floor_texture_filename_array = [
        "textures/lg_floor_d.tga",
        "textures/lg_style_01_floor_blue_d.tga",
        "textures/lg_style_01_floor_orange_bright_d.tga",
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
    rt_args.ambient_light_intensity = 0.2

    cuda_args = rtx.CUDAKernelLaunchArguments()
    cuda_args.num_threads = 64
    cuda_args.num_rays_per_thread = 16

    renderer = rtx.Renderer()
    render_buffer = np.zeros(
        (screen_height, screen_width, 3), dtype=np.float32)

    dataset = gqn.archiver.Archiver(
        directory=args.output_directory,
        total_observations=args.total_observations,
        num_observations_per_file=min(args.num_observations_per_file,
                                      args.total_observations),
        image_size=(args.image_size, args.image_size),
        num_views_per_scene=args.num_views_per_scene,
        initial_file_number=args.initial_file_number)

    camera = rtx.PerspectiveCamera(
        fov_rad=math.pi / 3, aspect_ratio=screen_width / screen_height)

    for _ in tqdm(range(args.total_observations)):
        scene = build_scene(
            random.sample(mnist_image_array, 6), wall_texture_filename_array,
            floor_texture_filename_array)
        scene_data = gqn.archiver.SceneData((args.image_size, args.image_size),
                                            args.num_views_per_scene)

        view_radius = 3

        for _ in range(args.num_views_per_scene):
            rotation = random.uniform(0, math.pi * 2)
            eye = (view_radius * math.cos(rotation), -0.125,
                   view_radius * math.sin(rotation))
            center = (0, -0.125, 0)
            camera.look_at(eye, center, up=(0, 1, 0))

            renderer.render(scene, camera, rt_args, cuda_args, render_buffer)

            # Convert to sRGB
            image = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
            image = np.uint8(image * 255)
            image = cv2.bilateralFilter(image, 3, 25, 25)

            yaw = gqn.math.yaw(eye, center)
            pitch = gqn.math.pitch(eye, center)
            scene_data.add(image, eye, math.cos(yaw), math.sin(yaw),
                           math.cos(pitch), math.sin(pitch))

            # plt.imshow(image, interpolation="none")
            # plt.pause(1e-8)

        dataset.add(scene_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument(
        "--total-observations", "-total", type=int, default=2000000)
    parser.add_argument(
        "--num-observations-per-file", "-per-file", type=int, default=2000)
    parser.add_argument("--initial-file-number", "-f", type=int, default=1)
    parser.add_argument("--num-views-per-scene", "-k", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--output-directory",
        "-out",
        type=str,
        default="dataset_shepard_matzler_train")
    args = parser.parse_args()
    main()