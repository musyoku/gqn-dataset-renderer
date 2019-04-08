import argparse
import colorsys
import math
import random
import time
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

import rtx
from archiver import Archiver, SceneData
from rooms_ring_camera import (build_scene, compute_yaw_and_pitch,
                               place_objects, floor_textures, wall_textures,
                               wall_height)


def main():
    try:
        os.makedirs(args.output_directory)
    except:
        pass
        
    # Set GPU device
    rtx.set_device(args.gpu_device)

    # Initialize colors
    colors = []
    for n in range(args.num_colors):
        hue = n / args.num_colors
        saturation = 1
        lightness = 1
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, lightness)
        colors.append((red, green, blue, 1))

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

    for _ in tqdm(range(args.total_scenes)):
        scene = build_scene(
            floor_textures,
            wall_textures,
            fix_light_position=args.fix_light_position)
        place_objects(
            scene,
            colors,
            max_num_objects=args.max_num_objects,
            discrete_position=args.discrete_position,
            rotate_object=args.rotate_object)
        scene_data = SceneData((args.image_size, args.image_size),
                               args.num_observations_per_scene)
        for _ in range(args.num_observations_per_scene):
            # Sample camera position
            rand_position_xz = np.random.uniform(-2.5, 2.5, size=2)
            rand_lookat_xz = np.random.uniform(-6, 6, size=2)
            camera_position = np.array(
                [rand_position_xz[0], 1, rand_position_xz[1]])
            look_at = np.array([rand_lookat_xz[0], 1, rand_lookat_xz[1]])

            # Compute yaw and pitch
            camera_direction = rand_position_xz - rand_lookat_xz
            camera_direction = np.array(
                [camera_direction[0], 0, camera_direction[1]])
            yaw, pitch = compute_yaw_and_pitch(camera_direction)

            camera.look_at(
                tuple(camera_position), tuple(look_at), up=(0, 1, 0))
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
    parser.add_argument("--max-num-objects", type=int, default=3)
    parser.add_argument("--num-colors", type=int, default=6)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--anti-aliasing", default=False, action="store_true")
    parser.add_argument(
        "--discrete-position", default=False, action="store_true")
    parser.add_argument("--rotate-object", default=False, action="store_true")
    parser.add_argument(
        "--fix-light-position", default=False, action="store_true")
    parser.add_argument("--visualize", default=False, action="store_true")
    args = parser.parse_args()
    main()
