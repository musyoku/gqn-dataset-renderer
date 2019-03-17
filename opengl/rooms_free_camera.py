import argparse
import colorsys
import math
import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyglet
import trimesh
from tqdm import tqdm

import pyrender
from archiver import Archiver, SceneData
from pyrender import (DirectionalLight, Mesh, Node, OffscreenRenderer,
                      PerspectiveCamera, PointLight, Primitive, RenderFlags,
                      Scene)
from rooms_ring_camera import (
    build_scene, place_objects, compute_yaw_and_pitch,
    genearte_camera_quaternion, set_random_texture, udpate_vertex_buffer)


def main():
    try:
        os.makedirs(args.output_directory)
    except:
        pass

    # Colors
    colors = []
    for n in range(args.num_colors):
        hue = n / args.num_colors
        saturation = 1
        lightness = 1
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, lightness)
        colors.append(np.array((red, green, blue, 1)))

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

    objects = [
        pyrender.objects.Capsule,
        pyrender.objects.Cylinder,
        pyrender.objects.Icosahedron,
        pyrender.objects.Box,
        pyrender.objects.Sphere,
    ]

    renderer = OffscreenRenderer(
        viewport_width=args.image_size, viewport_height=args.image_size)

    archiver = Archiver(
        directory=args.output_directory,
        total_scenes=args.total_scenes,
        num_scenes_per_file=min(args.num_scenes_per_file, args.total_scenes),
        image_size=(args.image_size, args.image_size),
        num_observations_per_scene=args.num_observations_per_scene,
        initial_file_number=args.initial_file_number)

    for scene_index in tqdm(range(args.total_scenes)):
        scene = build_scene(floor_textures, wall_textures)
        place_objects(
            scene,
            colors,
            objects,
            max_num_objects=args.max_num_objects,
            discrete_position=args.discrete_position,
            rotate_object=args.rotate_object)
        camera_distance = 4
        camera = PerspectiveCamera(yfov=math.pi / 4)
        camera_node = Node(camera=camera, translation=np.array([0, 1, 1]))
        scene.add_node(camera_node)
        scene_data = SceneData((args.image_size, args.image_size),
                               args.num_observations_per_scene)
        for observation_index in range(args.num_observations_per_scene):
            rand_position_xz = np.random.uniform(-3, 3, size=2)
            rand_position_xz = 3 * rand_position_xz / np.linalg.norm(
                rand_position_xz)
            rand_lookat_xz = np.random.uniform(-6, 6, size=2)
            camera_position = np.array(
                [rand_position_xz[0], 1, rand_position_xz[1]])
            camera_direction = rand_position_xz - rand_lookat_xz
            camera_direction = np.array(
                [camera_direction[0], 0, camera_direction[1]])
            # Compute yaw and pitch
            yaw, pitch = compute_yaw_and_pitch(camera_direction)

            camera_node.rotation = genearte_camera_quaternion(yaw, pitch)
            camera_node.translation = camera_position

            # Rendering
            flags = RenderFlags.SHADOWS_DIRECTIONAL
            if args.anti_aliasing:
                flags |= RenderFlags.ANTI_ALIASING
            image = renderer.render(scene, flags=flags)[0]
            scene_data.add(image, camera_position, math.cos(yaw),
                           math.sin(yaw), math.cos(pitch), math.sin(pitch))

            if args.visualize:
                plt.clf()
                plt.imshow(image)
                plt.pause(1e-10)

        archiver.add(scene_data)

    renderer.delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--visualize", default=False, action="store_true")
    args = parser.parse_args()
    main()
