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

floor_size = 7
wall_height = floor_size / 3

# Textures
floor_textures = [
    "../textures/lg_floor_d.tga",
    "../textures/lg_style_01_floor_blue_d.tga",
    "../textures/lg_style_01_floor_orange_bright_d.tga",
]

wall_textures = [
    "../textures/lg_style_01_wall_cerise_d.tga",
    "../textures/lg_style_01_wall_green_bright_d.tga",
    "../textures/lg_style_01_wall_red_bright_d.tga",
    "../textures/lg_style_02_wall_yellow_d.tga",
    "../textures/lg_style_03_wall_orange_bright_d.tga",
]


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


def generate_texture_mapping(texture, wall_aspect_ratio=1.0, scale=1.0):
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


def create_geometry_by_type(geometry_type):
    if geometry_type == GeometryType.box:
        return rtx.BoxGeometry(width=0.75, height=0.75, depth=0.75), 0.375

    if geometry_type == GeometryType.shpere:
        return rtx.SphereGeometry(radius=0.375), 0.375

    if geometry_type == GeometryType.cylinder:
        return rtx.CylinderGeometry(radius=0.25, height=1), 0.5

    if geometry_type == GeometryType.cone:
        return rtx.ConeGeometry(radius=0.375, height=1), 0.375

    raise NotImplementedError


def compute_yaw_and_pitch(vec):
    x, y, z = vec
    norm = np.linalg.norm(vec)
    if z < 0:
        yaw = math.pi + math.atan(x / z)
    elif x < 0:
        if z == 0:
            yaw = math.pi * 1.5
        else:
            yaw = math.pi * 2 + math.atan(x / z)
    elif z == 0:
        yaw = math.pi / 2
    else:
        yaw = math.atan(x / z)
    pitch = -math.asin(y / norm)
    return yaw, pitch


def build_scene(floor_textures, wall_textures, fix_light_position=False):
    scene = rtx.Scene(ambient_color=(153 / 255, 226 / 255, 249 / 255))

    texture = load_texture_image(random.choice(wall_textures))
    mapping = generate_texture_mapping(texture, floor_size / wall_height)

    # Place walls
    ## 1
    geometry = rtx.PlainGeometry(floor_size, wall_height)
    geometry.set_rotation((0, 0, 0))
    geometry.set_position((0, wall_height / 2, -floor_size / 2))
    material = rtx.LambertMaterial(0.95)
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    ## 2
    geometry = rtx.PlainGeometry(floor_size, wall_height)
    geometry.set_rotation((0, -math.pi / 2, 0))
    geometry.set_position((floor_size / 2, wall_height / 2, 0))
    material = rtx.LambertMaterial(0.95)
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    ## 3
    geometry = rtx.PlainGeometry(floor_size, wall_height)
    geometry.set_rotation((0, math.pi, 0))
    geometry.set_position((0, wall_height / 2, floor_size / 2))
    material = rtx.LambertMaterial(0.95)
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    ## 4
    geometry = rtx.PlainGeometry(floor_size, wall_height)
    geometry.set_rotation((0, math.pi / 2, 0))
    geometry.set_position((-floor_size / 2, wall_height / 2, 0))
    material = rtx.LambertMaterial(0.95)
    wall = rtx.Object(geometry, material, mapping)
    scene.add(wall)

    # floor
    geometry = rtx.PlainGeometry(floor_size, floor_size)
    geometry.set_rotation((-math.pi / 2, 0, 0))
    geometry.set_position((0, 0, 0))
    material = rtx.LambertMaterial(0.95)
    texture = load_texture_image(random.choice(floor_textures))
    mapping = generate_texture_mapping(texture, scale=0.5)
    floor = rtx.Object(geometry, material, mapping)
    scene.add(floor)

    # Place a light
    geometry = rtx.SphereGeometry(2)
    spread = floor_size / 2 - 1
    geometry.set_position((random.uniform(-spread, spread), 8,
                           random.uniform(-spread, spread)))
    material = rtx.EmissiveMaterial(20, visible=False)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    light = rtx.Object(geometry, material, mapping)
    scene.add(light)

    return scene


def place_objects(scene,
                  colors,
                  max_num_objects=3,
                  min_num_objects=1,
                  discrete_position=False,
                  rotate_object=False):
    # Place objects
    directions = [-1.5, 0.0, 1.5]
    available_positions = []
    for z in directions:
        for x in directions:
            available_positions.append((x, z))
    available_positions = np.array(available_positions)
    num_objects = random.choice(range(min_num_objects, max_num_objects + 1))
    indices = np.random.choice(
        np.arange(len(available_positions)), replace=False, size=num_objects)
    for xz in available_positions[indices]:
        geometry_type = random.choice(geometry_type_array)
        geometry, offset_y = create_geometry_by_type(geometry_type)
        if rotate_object:
            geometry.set_rotation((0, random.uniform(0, math.pi * 2), 0))
        if discrete_position == False:
            xz += np.random.uniform(-0.3, 0.3, size=xz.shape)

        geometry.set_position((
            xz[0],
            offset_y,
            xz[1],
        ))
        material = rtx.LambertMaterial(0.9)
        color = random.choice(colors)
        mapping = rtx.SolidColorMapping(color)
        obj = rtx.Object(geometry, material, mapping)
        scene.add(obj)


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
    camera_distance = 2

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
