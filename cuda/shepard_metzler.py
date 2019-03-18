import argparse
import colorsys
import math
import random
import time
import cv2

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from archiver import Archiver, SceneData
import rtx

cube_size = 0.3


def compute_yaw_and_pitch(vec):
    x, y, z = vec
    norm = np.linalg.norm(vec)
    if z < 0:
        yaw = math.pi + math.atan(x / z)
    elif x < 0:
        yaw = math.pi * 2 + math.atan(x / z)
    else:
        yaw = math.atan(x / z)
    pitch = -math.asin(y / norm)
    return yaw, pitch


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
    barycenter = np.array([0.0, 0.0, 0.0])

    for location in block_locations:
        shift = cube_size
        position = (shift * location[0], shift * location[1],
                    shift * location[2])

        position_array.append(position)

        barycenter[0] += position[0]
        barycenter[1] += position[1]
        barycenter[2] += position[2]

    barycenter[0] /= num_cubes
    barycenter[1] /= num_cubes
    barycenter[2] /= num_cubes

    # discretize
    barycenter = np.round(barycenter / cube_size) * cube_size

    return position_array, barycenter


def build_scene(num_cubes, color_array):
    # Generate positions of each cube
    cube_position_array, barycenter = generate_block_positions(num_cubes)

    # Place cubes
    scene = rtx.Scene(ambient_color=(0, 0, 0))
    for position in cube_position_array:
        geometry = rtx.BoxGeometry(cube_size, cube_size, cube_size)
        geometry.set_position((
            position[0] - barycenter[0],
            position[1] - barycenter[1],
            position[2] - barycenter[2],
        ))
        material = rtx.LambertMaterial(0.3)
        mapping = rtx.SolidColorMapping(random.choice(color_array))
        cube = rtx.Object(geometry, material, mapping)
        scene.add(cube)

    # Place a light
    size = 50
    geometry = rtx.SphereGeometry(size)
    geometry.set_position((size * 2, size * 2, 0))
    material = rtx.EmissiveMaterial(100, visible=False)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    light = rtx.Object(geometry, material, mapping)
    scene.add(light)

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
    rt_args.supersampling_enabled = args.anti_aliasing
    rt_args.ambient_light_intensity = 0.05

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

    camera = rtx.OrthographicCamera()

    for _ in tqdm(range(args.total_scenes)):
        scene = build_scene(args.num_cubes, color_array)
        scene_data = SceneData((args.image_size, args.image_size),
                               args.num_observations_per_scene)

        camera_distance = 1

        for _ in range(args.num_observations_per_scene):
            # Generate random point on a sphere
            camera_position = np.random.normal(size=3)
            camera_position = camera_distance * camera_position / np.linalg.norm(
                camera_position)
            # Compute yaw and pitch
            yaw, pitch = compute_yaw_and_pitch(camera_position)

            center = (0, 0, 0)
            camera.look_at(tuple(camera_position), center, up=(0, 1, 0))

            renderer.render(scene, camera, rt_args, cuda_args, render_buffer)

            # Convert to sRGB
            image = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
            image = np.uint8(image * 255)
            image = cv2.bilateralFilter(image, 3, 25, 25)

            if args.visualize:
                plt.clf()
                plt.imshow(image)
                plt.pause(1e-10)

            scene_data.add(image, camera_position, math.cos(yaw),
                           math.sin(yaw), math.cos(pitch), math.sin(pitch))

        archiver.add(scene_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--total-scenes", "-total", type=int, default=2000000)
    parser.add_argument("--num-scenes-per-file", type=int, default=2000)
    parser.add_argument("--initial-file-number", type=int, default=1)
    parser.add_argument("--num-observations-per-scene", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-cubes", type=int, default=5)
    parser.add_argument("--num-colors", type=int, default=10)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--anti-aliasing", default=False, action="store_true")
    parser.add_argument("--visualize", default=False, action="store_true")
    args = parser.parse_args()
    main()
