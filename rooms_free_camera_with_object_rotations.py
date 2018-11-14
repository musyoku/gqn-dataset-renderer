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


def load_texture_image(filename):
    image = Image.open(filename)
    image = image.convert("RGB")
    texture = np.array(image, dtype=np.float32) / 255
    return texture


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


def build_geometry_by_type(geometry_type):
    if geometry_type == GeometryType.box:
        return rtx.BoxGeometry(width=1, height=1, depth=1)

    if geometry_type == GeometryType.shpere:
        return rtx.SphereGeometry(radius=0.5)

    if geometry_type == GeometryType.cylinder:
        return rtx.CylinderGeometry(radius=0.5, height=1)

    if geometry_type == GeometryType.cone:
        return rtx.ConeGeometry(radius=0.5, height=1)

    raise NotImplementedError


def generate_object_positions(num_objects, grid_size):
    available_positions = []
    for y in range(grid_size):
        for x in range(grid_size):
            available_positions.append((x, y))
    ret = random.sample(available_positions, num_objects)
    return ret


def build_scene(color_array, wall_texture_filename_array,
                floor_texture_filename_array, grid_size, wall_height):
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
    primary_light = rtx.ObjectGroup()
    geometry = rtx.SphereGeometry(2)
    material = rtx.EmissiveMaterial(40, visible=False)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    light = rtx.Object(geometry, material, mapping)
    primary_light.add(light)

    spread = grid_size / 2 - 1
    primary_light.set_position((spread * random.uniform(-1, 1), 8,
                                spread * random.uniform(-1, 1)))
    scene.add(primary_light)

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

    # Place objects
    r = grid_size // 4
    r2 = r * 2
    object_positions = generate_object_positions(args.num_objects, r2 - 1)
    for position_index in object_positions:
        geometry_type = random.choice(geometry_type_array)
        geometry = build_geometry_by_type(geometry_type)
        geometry.set_rotation((0, random.uniform(0, math.pi * 2), 0))

        noise = np.random.uniform(-0.125, 0.125, size=2)
        spread = 1.5
        geometry.set_position((
            spread * (position_index[0] - r + 1) + noise[0],
            -wall_height / 2 + 0.5,
            spread * (position_index[1] - r + 1) + noise[1],
        ))
        material = rtx.LambertMaterial(0.9)
        color = random.choice(color_array)
        mapping = rtx.SolidColorMapping(color)
        obj = rtx.Object(geometry, material, mapping)
        scene.add(obj)
    return scene


def main():
    # Set GPU device
    rtx.set_device(args.gpu_device)

    # Texture
    wall_texture_filename_array = [
        "textures/wall_texture_1.png",
        "textures/wall_texture_2.jpg",
        "textures/wall_texture_3.jpg",
        "textures/wall_texture_4.jpg",
        "textures/wall_texture_5.jpg",
        "textures/wall_texture_6.jpg",
        "textures/wall_texture_7.jpg",
    ]
    floor_texture_filename_array = [
        "textures/floor_texture_1.png",
    ]

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

    grid_size = 8
    wall_height = 3

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
        scene = build_scene(color_array, wall_texture_filename_array,
                            floor_texture_filename_array, grid_size,
                            wall_height)
        scene_data = gqn.archiver.SceneData((args.image_size, args.image_size),
                                            args.num_views_per_scene)

        for _ in range(args.num_views_per_scene):
            rotation = random.uniform(0, math.pi * 2)
            scale = grid_size - 5
            eye = (random.uniform(-scale, scale), -0.125,
                   random.uniform(-scale, scale))
            center = (random.uniform(-scale, scale), -0.125,
                      random.uniform(-scale, scale))
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
    parser.add_argument("--num-objects", "-objects", type=int, default=3)
    parser.add_argument("--num-colors", "-colors", type=int, default=12)
    parser.add_argument(
        "--output-directory",
        "-out",
        type=str,
        default="dataset_shepard_matzler_train")
    args = parser.parse_args()
    main()
