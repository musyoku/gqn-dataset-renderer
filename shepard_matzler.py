import math
import time
import argparse
import random
import colorsys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import rtx


def generate_block_positions(num_cubes):
    assert num_cubes > 0

    current_position = (0, 0, 0)
    block_locations = [current_position]

    for _ in range(num_cubes - 1):
        axis = random.choice([0, 1, 2])
        direction = random.choice([-1, 1])
        offset = [0, 0, 0]
        offset[axis] = direction
        new_position = (offset[0] + current_position[0],
                        offset[1] + current_position[1],
                        offset[2] + current_position[2])
        block_locations.append(new_position)
        current_position = new_position

    position_array = []
    center_of_gravity = [0, 0, 0]

    for location in block_locations:
        shift = 1
        position = (shift * location[0], shift * location[1],
                    shift * location[2])

        position_array.append(position)

        center_of_gravity[0] += position[0]
        center_of_gravity[1] += position[1]
        center_of_gravity[2] += position[2]

    center_of_gravity[0] /= num_cubes
    center_of_gravity[1] /= num_cubes
    center_of_gravity[2] /= num_cubes

    return position_array, center_of_gravity


def build_scene(color_array):
    # Generate positions of each cube
    cube_position_array, shift = generate_block_positions(args.num_cubes)
    assert len(cube_position_array) == args.num_cubes

    # Place block
    scene = rtx.Scene(ambient_color=(0, 0, 0))
    for position in cube_position_array:
        geometry = rtx.BoxGeometry(1, 1, 1)
        geometry.set_position((
            position[0] - shift[0],
            position[1] - shift[1],
            position[2] - shift[2],
        ))
        material = rtx.LambertMaterial(0.5)
        mapping = rtx.SolidColorMapping(random.choice(color_array))
        cube = rtx.Object(geometry, material, mapping)
        scene.add(cube)

    # Place lights
    group = rtx.ObjectGroup()
    geometry = rtx.PlainGeometry(10, 10)
    geometry.set_rotation((0, math.pi / 2, 0))
    geometry.set_position((-5, 0, 0))
    material = rtx.EmissiveMaterial(10.0, visible=False)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    light = rtx.Object(geometry, material, mapping)
    group.add(light)

    geometry = rtx.PlainGeometry(10, 10)
    geometry.set_rotation((0, -math.pi / 2, 0))
    geometry.set_position((5, 0, 0))
    material = rtx.EmissiveMaterial(0.5, visible=False)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    light = rtx.Object(geometry, material, mapping)
    group.add(light)

    group.set_rotation((-math.pi / 4, math.pi / 4, 0))
    scene.add(group)

    return scene


def main():
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
    rt_args.num_rays_per_pixel = 512
    rt_args.max_bounce = 3
    rt_args.next_event_estimation_enabled = True
    rt_args.supersampling_enabled = False

    cuda_args = rtx.CUDAKernelLaunchArguments()
    cuda_args.num_threads = 128
    cuda_args.num_rays_per_thread = 32

    renderer = rtx.Renderer()
    render_buffer = np.zeros(
        (screen_height, screen_width, 3), dtype=np.float32)

    for _ in range(args.total_observations):
        scene = build_scene(color_array)

        view_radius = 2
        camera = rtx.OrthographicCamera(
            eye=(0, 1, 1), center=(0, 0, 0), up=(0, 1, 0))

        total_iterations = 3000
        cos = 0
        for _ in range(args.num_views_per_scene):
            eye = (view_radius * math.sin(cos), view_radius,
                   view_radius * math.cos(cos))
            center = (0, 0, 0)
            camera.look_at(eye, center, up=(0, 1, 0))

            renderer.render(scene, camera, rt_args, cuda_args, render_buffer)

            # Convert to sRGB
            pixels = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)

            plt.imshow(pixels, interpolation="none")
            plt.pause(1e-8)

            cos += 0.1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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