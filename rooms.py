import argparse
import colorsys
import math
import random
import time
import cv2

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import gqn
import rtx


def build_scene(color_array):
    grid_size = 7
    wall_height = 2
    eps = 10
    scene = rtx.Scene(ambient_color=(0.5, 1, 1))

    # 1
    geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
    geometry.set_rotation((0, 0, 0))
    geometry.set_position((0, 0, -grid_size / 2))
    material = rtx.LambertMaterial(0.95)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    # 2
    geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
    geometry.set_rotation((0, -math.pi / 2, 0))
    geometry.set_position((grid_size / 2, 0, 0))
    material = rtx.LambertMaterial(0.95)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    # 3
    geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
    geometry.set_rotation((0, math.pi, 0))
    geometry.set_position((0, 0, grid_size / 2))
    material = rtx.LambertMaterial(0.95)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    # 4
    geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
    geometry.set_rotation((0, math.pi / 2, 0))
    geometry.set_position((-grid_size / 2, 0, 0))
    material = rtx.LambertMaterial(0.95)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    # floor
    geometry = rtx.PlainGeometry(grid_size + eps, grid_size + eps)
    geometry.set_rotation((-math.pi / 2, 0, 0))
    geometry.set_position((0, -wall_height / 2, 0))
    material = rtx.LambertMaterial(0.95)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    ceil = rtx.Object(geometry, material, mapping)
    scene.add(ceil)

    # Place lights
    size = 50
    group = rtx.ObjectGroup()
    geometry = rtx.PlainGeometry(size, size)
    geometry.set_rotation((0, math.pi / 2, 0))
    geometry.set_position((-10, 0, 0))
    material = rtx.EmissiveMaterial(1, visible=False)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    light = rtx.Object(geometry, material, mapping)
    group.add(light)
    group.set_rotation((-math.pi / 2.5, math.pi / 2, 0))
    scene.add(group)

    return scene


def main():
    # Set GPU device
    rtx.set_device(args.gpu_device)

    # Initialize colors
    color_array = []
    for n in range(args.num_colors):
        hue = n / (args.num_colors - 1)
        saturation = 1
        lightness = 1
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, lightness)
        color_array.append((red, green, blue, 1))

    screen_width = args.image_size
    screen_height = args.image_size

    # Setting up a raytracer
    rt_args = rtx.RayTracingArguments()
    rt_args.num_rays_per_pixel = 2048
    rt_args.max_bounce = 3
    rt_args.supersampling_enabled = True

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
        start_file_number=args.start_file_number)

    camera = rtx.PerspectiveCamera(
        eye=(0, 0, 1),
        center=(0, 0, 0),
        up=(0, 1, 0),
        fov_rad=math.pi / 3,
        aspect_ratio=screen_width / screen_height,
        z_near=0.01,
        z_far=100)

    for _ in tqdm(range(args.total_observations)):
        scene = build_scene(color_array)
        scene_data = gqn.archiver.SceneData((args.image_size, args.image_size),
                                            args.num_views_per_scene)

        view_radius = 3
        rotation = 0

        for _ in range(args.num_views_per_scene):
            eye = (view_radius * math.cos(rotation), 0.0,
                        view_radius * math.sin(rotation))
            center = (0, 0, 0)
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

            plt.imshow(image, interpolation="none")
            plt.pause(1e-8)

            rotation += math.pi / 36

        dataset.add(scene_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument(
        "--total-observations", "-total", type=int, default=2000000)
    parser.add_argument(
        "--num-observations-per-file", "-per-file", type=int, default=2000)
    parser.add_argument("--start-file-number", type=int, default=1)
    parser.add_argument("--num-views-per-scene", "-k", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-cubes", "-cubes", type=int, default=5)
    parser.add_argument("--num-colors", "-colors", type=int, default=20)
    parser.add_argument(
        "--output-directory",
        "-out",
        type=str,
        default="dataset_shepard_matzler_train")
    args = parser.parse_args()
    main()
