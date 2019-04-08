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

from rooms_ring_camera import (build_scene, floor_textures, wall_textures,
                               wall_height)
from mnist_dice_ring_camera import place_dice, load_mnist_images
import rtx


def main():
    # Set GPU device
    rtx.set_device(args.gpu_device)

    # Load MNIST images
    mnist_images = load_mnist_images()

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

    perspective_camera = rtx.PerspectiveCamera(
        fov_rad=math.pi / 3, aspect_ratio=screen_width / screen_height)
    orthographic_camera = rtx.OrthographicCamera()
    camera_distance = 2

    plt.tight_layout()

    scene = build_scene(
        floor_textures,
        wall_textures,
        fix_light_position=args.fix_light_position)
    place_dice(
        scene,
        mnist_images,
        discrete_position=args.discrete_position,
        rotate_dice=args.rotate_dice)

    current_rad = 0
    rad_step = math.pi / 36
    total_frames = int(math.pi * 2 / rad_step)

    fig = plt.figure(figsize=(6, 3))
    axis_perspective = fig.add_subplot(1, 2, 1)
    axis_orthogonal = fig.add_subplot(1, 2, 2)
    ims = []

    for _ in range(total_frames):
        # Perspective camera
        camera_position = (camera_distance * math.cos(current_rad),
                           wall_height / 2,
                           camera_distance * math.sin(current_rad))
        center = (0, wall_height / 2, 0)

        perspective_camera.look_at(camera_position, center, up=(0, 1, 0))
        renderer.render(scene, perspective_camera, rt_args, cuda_args,
                        render_buffer)

        image = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
        image = np.uint8(image * 255)
        image = cv2.bilateralFilter(image, 3, 25, 25)
        im1 = axis_perspective.imshow(
            image, interpolation="none", animated=True)

        # Orthographic camera
        offset_y = 1
        camera_position = (2 * math.cos(current_rad),
                           2 * math.sin(math.pi / 6) + offset_y,
                           2 * math.sin(current_rad))
        center = (0, offset_y, 0)

        orthographic_camera.look_at(camera_position, center, up=(0, 1, 0))
        renderer.render(scene, orthographic_camera, rt_args, cuda_args,
                        render_buffer)

        image = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
        image = np.uint8(image * 255)
        image = cv2.bilateralFilter(image, 3, 25, 25)
        im2 = axis_orthogonal.imshow(
            image, interpolation="none", animated=True)
        ims.append([im1, im2])

        plt.pause(1e-8)

        current_rad += rad_step

    ani = animation.ArtistAnimation(
        fig, ims, interval=1 / 24, blit=True, repeat_delay=0)
    filename = "mnist_dice"
    if args.discrete_position:
        filename += "_discrete_position"
    if args.rotate_dice:
        filename += "_rotate_dice"
    if args.fix_light_position:
        filename += "_fix_light_position"
    filename += ".gif"
    ani.save(filename, writer="imagemagick")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-objects", "-objects", type=int, default=3)
    parser.add_argument("--num-colors", "-colors", type=int, default=6)
    parser.add_argument("--anti-aliasing", default=False, action="store_true")
    parser.add_argument(
        "--discrete-position", default=False, action="store_true")
    parser.add_argument("--rotate-dice", default=False, action="store_true")
    parser.add_argument(
        "--fix-light-position", default=False, action="store_true")
    args = parser.parse_args()
    main()
