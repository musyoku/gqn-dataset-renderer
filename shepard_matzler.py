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


def get_available_axis_and_direction(space, pos):
    ret = []
    # x-axis
    for direction in (-1, 1):
        abs_pos = (pos[0] + direction, pos[1], pos[2])
        if space[abs_pos] == True:
            continue
        ret.append((0, direction))
    # y-axis
    for direction in (-1, 1):
        abs_pos = (pos[0], pos[1] + direction, pos[2])
        if space[abs_pos] == True:
            continue
        ret.append((1, direction))
    # z-axis
    for direction in (-1, 1):
        abs_pos = (pos[0], pos[1], pos[2] + direction)
        if space[abs_pos] == True:
            continue
        ret.append((2, direction))

    return ret


def generate_block_positions(num_cubes):
    assert num_cubes > 0

    current_relative_pos = (0, 0, 0)
    block_locations = [current_relative_pos]
    block_abs_locations = np.zeros(
        (num_cubes * 2 - 1, num_cubes * 2 - 1, num_cubes * 2 - 1), dtype=bool)
    p = num_cubes - 1
    current_absolute_pos = (p, p, p)
    block_abs_locations[current_absolute_pos] = True

    for _ in range(num_cubes - 1):
        available_axis_and_direction = get_available_axis_and_direction(
            block_abs_locations, current_absolute_pos)
        axis, direction = random.choice(available_axis_and_direction)
        offset = [0, 0, 0]
        offset[axis] = direction
        new_relative_pos = (offset[0] + current_relative_pos[0],
                            offset[1] + current_relative_pos[1],
                            offset[2] + current_relative_pos[2])
        block_locations.append(new_relative_pos)
        current_relative_pos = new_relative_pos
        current_absolute_pos = (
            new_relative_pos[0] + p,
            new_relative_pos[1] + p,
            new_relative_pos[2] + p,
        )
        block_abs_locations[current_absolute_pos] = True

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
        material = rtx.LambertMaterial(0.3)
        mapping = rtx.SolidColorMapping(random.choice(color_array))
        cube = rtx.Object(geometry, material, mapping)
        scene.add(cube)

    # Place lights
    size = 50
    group = rtx.ObjectGroup()
    geometry = rtx.PlainGeometry(size, size)
    geometry.set_rotation((0, math.pi / 2, 0))
    geometry.set_position((-10, 0, 0))
    material = rtx.EmissiveMaterial(10, visible=False)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    light = rtx.Object(geometry, material, mapping)
    group.add(light)

    geometry = rtx.PlainGeometry(size, size)
    geometry.set_rotation((0, -math.pi / 2, 0))
    geometry.set_position((10, 0, 0))
    material = rtx.EmissiveMaterial(1, visible=False)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    light = rtx.Object(geometry, material, mapping)
    group.add(light)

    group.set_rotation((-math.pi / 4, math.pi / 4, 0))
    scene.add(group)

    return scene


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
    rt_args.supersampling_enabled = False

    cuda_args = rtx.CUDAKernelLaunchArguments()
    cuda_args.num_threads = 64
    cuda_args.num_rays_per_thread = 32

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

    camera = rtx.OrthographicCamera()

    fig = plt.figure()

    for _ in tqdm(range(args.total_observations)):
        scene = build_scene(color_array)
        scene_data = gqn.archiver.SceneData((args.image_size, args.image_size),
                                            args.num_views_per_scene)

        view_radius = 3

        for _ in range(args.num_views_per_scene):
            eye = np.random.normal(size=3)
            eye = tuple(view_radius * (eye / np.linalg.norm(eye)))
            center = (0, 0, 0)
            camera.look_at(eye, center, up=(0, 1, 0))

            renderer.render(scene, camera, rt_args, cuda_args, render_buffer)

            # Convert to sRGB
            image = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
            image = np.uint8(image * 255)
            image = cv2.bilateralFilter(image, 3, 25, 25)

            # plt.imshow(image, interpolation="none")
            # plt.pause(1e-8)

            yaw = gqn.math.yaw(eye, center)
            pitch = gqn.math.pitch(eye, center)
            scene_data.add(image, eye, math.cos(yaw), math.sin(yaw),
                           math.cos(pitch), math.sin(pitch))

        dataset.add(scene_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument(
        "--total-observations", "-total", type=int, default=2000000)
    parser.add_argument(
        "--num-observations-per-file", "-per-file", type=int, default=2000)
    parser.add_argument("--initial-file-number", "-f", type=int, default=1)
    parser.add_argument("--num-views-per-scene", "-k", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-cubes", "-cubes", type=int, default=5)
    parser.add_argument("--num-colors", "-colors", type=int, default=12)
    parser.add_argument(
        "--output-directory",
        "-out",
        type=str,
        default="dataset_shepard_matzler_train")
    args = parser.parse_args()
    main()
