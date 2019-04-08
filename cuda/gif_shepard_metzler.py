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

from shepard_metzler import (build_scene, compute_yaw_and_pitch)
import rtx


def main():
    # Set GPU device
    rtx.set_device(args.gpu_device)

    # Initialize colors
    color_array = []
    for n in range(args.num_colors):
        hue = n / (args.num_colors - 1)
        saturation = 0.9
        lightness = 1
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, lightness)
        color_array.append((red, green, blue, 1))

    screen_width = args.image_size
    screen_height = args.image_size

    # Setting up a raytracer
    rt_args = rtx.RayTracingArguments()
    rt_args.num_rays_per_pixel = 512
    rt_args.max_bounce = 2
    rt_args.supersampling_enabled = args.anti_aliasing
    rt_args.ambient_light_intensity = 0.05

    cuda_args = rtx.CUDAKernelLaunchArguments()
    cuda_args.num_threads = 64
    cuda_args.num_rays_per_thread = 32

    renderer = rtx.Renderer()
    render_buffer = np.zeros(
        (screen_height, screen_width, 3), dtype=np.float32)

    camera = rtx.OrthographicCamera()

    fig = plt.figure(figsize=(3, 3))
    ims = []

    camera_distance = 1
    current_rad = 0
    rad_step = math.pi / 18
    total_frames = int(math.pi * 2 / rad_step)

    for num_cubes in range(1, 8):
        scene = build_scene(num_cubes, color_array)

        for _ in range(total_frames):
            camera_position = (camera_distance * math.sin(current_rad),
                               camera_distance * math.sin(math.pi / 6),
                               camera_distance * math.cos(current_rad))
            center = (0, 0, 0)
            camera.look_at(camera_position, center, up=(0, 1, 0))

            renderer.render(scene, camera, rt_args, cuda_args, render_buffer)

            # Convert to sRGB
            image = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
            image = np.uint8(image * 255)
            image = cv2.bilateralFilter(image, 3, 25, 25)

            im = plt.imshow(image, interpolation="none", animated=True)
            ims.append([im])

            # plt.pause(1e-8)

            current_rad += rad_step

    ani = animation.ArtistAnimation(
        fig, ims, interval=1 / 24, blit=True, repeat_delay=0)

    ani.save('shepard_matzler.gif', writer="imagemagick")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-colors", "-colors", type=int, default=10)
    parser.add_argument("--anti-aliasing", default=False, action="store_true")
    args = parser.parse_args()
    main()
