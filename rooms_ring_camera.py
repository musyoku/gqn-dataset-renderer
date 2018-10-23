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
                floor_texture_filename_array):
    grid_size = 8
    wall_height = 3
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
    light_group = rtx.ObjectGroup()

    geometry = rtx.SphereGeometry(5)
    geometry.set_position((grid_size / 2 - 1, 10, grid_size / 2 - 1))
    material = rtx.EmissiveMaterial(10, visible=True)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    light = rtx.Object(geometry, material, mapping)
    light_group.add(light)

    geometry = rtx.SphereGeometry(3)
    geometry.set_position((-grid_size / 2 - 1, 10, -grid_size / 2 - 1))
    material = rtx.EmissiveMaterial(2, visible=True)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    light = rtx.Object(geometry, material, mapping)
    light_group.add(light)

    light_group
    scene.add(light_group)

    # Place objects
    r = grid_size // 4
    r2 = r * 2
    object_positions = generate_object_positions(args.num_objects, r2 - 1)
    for position_index in object_positions:
        geometry_type = random.choice(geometry_type_array)
        geometry = build_geometry_by_type(geometry_type)
        geometry.set_rotation((0, math.pi * random.uniform(0, 1), 0))

        noise = np.random.uniform(-0.125, 0.125, size=2)
        spread = 1.5
        geometry.set_position((
            spread * (position_index[0] - r + 0.5) + noise[0],
            -wall_height / 2 + 0.5,
            spread * (position_index[1] - r + 0.5) + noise[1],
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
        scene = build_scene(color_array, wall_texture_filename_array,
                            floor_texture_filename_array)
        scene_data = gqn.archiver.SceneData((args.image_size, args.image_size),
                                            args.num_views_per_scene)

        view_radius = 3
        rotation = 0

        fig = plt.figure()
        plt.title("Rooms")
        ims = []

        for _ in range(args.num_views_per_scene):
            eye = (view_radius * math.cos(rotation), -0.125,
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

            im = plt.imshow(image, interpolation="none", animated=True)
            ims.append([im])

            plt.pause(1e-8)

            rotation += math.pi / 24

        ani = animation.ArtistAnimation(
            fig, ims, interval=1 / 24, blit=True, repeat_delay=0)

        ani.save('anim.gif', writer="imagemagick")
        exit()

        dataset.add(scene_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument(
        "--total-observations", "-total", type=int, default=2000000)
    parser.add_argument(
        "--num-observations-per-file", "-per-file", type=int, default=2000)
    parser.add_argument("--start-file-number", "-start", type=int, default=1)
    parser.add_argument("--num-views-per-scene", "-k", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-objects", "-objects", type=int, default=3)
    parser.add_argument("--num-colors", "-colors", type=int, default=20)
    parser.add_argument(
        "--output-directory",
        "-out",
        type=str,
        default="dataset_shepard_matzler_train")
    args = parser.parse_args()
    main()
