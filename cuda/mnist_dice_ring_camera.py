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

import rtx
from archiver import Archiver, SceneData
from rooms_ring_camera import (build_scene, compute_yaw_and_pitch,
                               generate_texture_mapping, floor_textures,
                               wall_textures, wall_height)


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


def place_dice(scene, mnist_images, discrete_position=False,
               rotate_dice=False):
    indices = np.random.choice(
        np.arange(len(mnist_images)), replace=False, size=6)

    dice = rtx.ObjectGroup()

    # 1
    geometry = rtx.PlainGeometry(1, 1)
    geometry.set_position((0, 0, 0.5))
    material = rtx.LambertMaterial(0.95)
    mapping = generate_texture_mapping(mnist_images[indices[0]])
    face = rtx.Object(geometry, material, mapping)
    dice.add(face)

    # 2
    geometry = rtx.PlainGeometry(1, 1)
    geometry.set_rotation((0, -math.pi, 0))
    geometry.set_position((0, 0, -0.5))
    material = rtx.LambertMaterial(0.95)
    mapping = generate_texture_mapping(mnist_images[indices[1]])
    face = rtx.Object(geometry, material, mapping)
    dice.add(face)

    # 3
    geometry = rtx.PlainGeometry(1, 1)
    geometry.set_rotation((0, math.pi / 2, 0))
    geometry.set_position((0.5, 0, 0))
    material = rtx.LambertMaterial(0.95)
    mapping = generate_texture_mapping(mnist_images[indices[2]])
    face = rtx.Object(geometry, material, mapping)
    dice.add(face)

    # 4
    geometry = rtx.PlainGeometry(1, 1)
    geometry.set_rotation((0, -math.pi / 2, 0))
    geometry.set_position((-0.5, 0, 0))
    material = rtx.LambertMaterial(0.95)
    mapping = generate_texture_mapping(mnist_images[indices[3]])
    face = rtx.Object(geometry, material, mapping)
    dice.add(face)

    # 5
    geometry = rtx.PlainGeometry(1, 1)
    geometry.set_rotation((math.pi / 2, 0, 0))
    geometry.set_position((0, -0.5, 0))
    material = rtx.LambertMaterial(0.95)
    mapping = generate_texture_mapping(mnist_images[indices[4]])
    face = rtx.Object(geometry, material, mapping)
    dice.add(face)

    # 6
    geometry = rtx.PlainGeometry(1, 1)
    geometry.set_rotation((-math.pi / 2, 0, 0))
    geometry.set_position((0, 0.5, 0))
    material = rtx.LambertMaterial(0.95)
    mapping = generate_texture_mapping(mnist_images[indices[5]])
    face = rtx.Object(geometry, material, mapping)
    dice.add(face)

    dice.set_scale((1.5, 1.5, 1.5))

    directions = [-1.0, 0.0, 1.0]
    available_positions = []
    for z in directions:
        for x in directions:
            available_positions.append((x, z))
    xz = np.array(random.choice(available_positions))
    if discrete_position == False:
        xz += np.random.uniform(-0.25, 0.25, size=xz.shape)
    dice.set_position((xz[0], 0.75, xz[1]))

    if rotate_dice:
        yaw = np.random.uniform(0, math.pi * 2, size=1)[0]
        dice.set_rotation((0, yaw, 0))

    scene.add(dice)

def main():
    try:
        os.makedirs(args.output_directory)
    except:
        pass

    # Load MNIST images
    mnist_images = load_mnist_images()

    # Set GPU device
    rtx.set_device(args.gpu_device)

    screen_width = args.image_size
    screen_height = args.image_size

    # Setting up a raytracer
    rt_args = rtx.RayTracingArguments()
    rt_args.num_rays_per_pixel = 1024
    rt_args.max_bounce = 3
    rt_args.supersampling_enabled = args.anti_aliasing
    rt_args.next_event_estimation_enabled = True
    rt_args.ambient_light_intensity = 0.1

    cuda_args = rtx.CUDAKernelLaunchArguments()
    cuda_args.num_threads = 64
    cuda_args.num_rays_per_thread = 32

    renderer = rtx.Renderer()
    render_buffer = np.zeros(
        (screen_height, screen_width, 3), dtype=np.float32)

    archiver = Archiver(
        directory=args.output_directory,
        total_scenes=args.total_scenes,
        num_scenes_per_file=min(args.num_scenes_per_file, args.total_scenes),
        image_size=(args.image_size, args.image_size),
        num_observations_per_scene=args.num_observations_per_scene,
        initial_file_number=args.initial_file_number)

    camera = rtx.PerspectiveCamera(
        fov_rad=math.pi / 3, aspect_ratio=screen_width / screen_height)
    camera_distance = 2

    for _ in tqdm(range(args.total_scenes)):
        scene = build_scene(
            floor_textures,
            wall_textures,
            fix_light_position=args.fix_light_position)
        place_dice(
            scene,
            mnist_images,
            discrete_position=args.discrete_position,
            rotate_dice=args.rotate_dice)
        scene_data = SceneData((args.image_size, args.image_size),
                               args.num_observations_per_scene)
        for _ in range(args.num_observations_per_scene):
            # Sample camera position
            rand_position_xz = np.random.normal(size=2)
            rand_position_xz = camera_distance * rand_position_xz / np.linalg.norm(
                rand_position_xz)
            camera_position = np.array((rand_position_xz[0], wall_height / 2,
                                        rand_position_xz[1]))
            center = np.array((0, wall_height / 2, 0))

            # Compute yaw and pitch
            camera_direction = camera_position - center
            yaw, pitch = compute_yaw_and_pitch(camera_direction)

            camera.look_at(tuple(camera_position), tuple(center), up=(0, 1, 0))
            renderer.render(scene, camera, rt_args, cuda_args, render_buffer)

            # Convert to sRGB
            image = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
            image = np.uint8(image * 255)
            image = cv2.bilateralFilter(image, 3, 25, 25)

            scene_data.add(image, camera_position, math.cos(yaw),
                           math.sin(yaw), math.cos(pitch), math.sin(pitch))

            if args.visualize:
                plt.clf()
                plt.imshow(image)
                plt.pause(1e-10)

        archiver.add(scene_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--total-scenes", "-total", type=int, default=2000000)
    parser.add_argument("--num-scenes-per-file", type=int, default=2000)
    parser.add_argument("--initial-file-number", type=int, default=1)
    parser.add_argument("--num-observations-per-scene", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--anti-aliasing", default=False, action="store_true")
    parser.add_argument(
        "--discrete-position", default=False, action="store_true")
    parser.add_argument("--rotate-dice", default=False, action="store_true")
    parser.add_argument(
        "--fix-light-position", default=False, action="store_true")
    parser.add_argument("--visualize", default=False, action="store_true")
    args = parser.parse_args()
    main()
