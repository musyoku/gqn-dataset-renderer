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
from PIL import Image, ImageEnhance
from tqdm import tqdm

from OpenGL.GL import GL_LINEAR_MIPMAP_LINEAR

import pyrender
from archiver import Archiver, SceneData
from pyrender import (DirectionalLight, Mesh, Node, OffscreenRenderer,
                      PerspectiveCamera, PointLight, RenderFlags, Scene,
                      Primitive)


def set_random_texture(node, path, intensity=1.0):
    texture_image = Image.open(path).convert("RGB")
    if intensity < 1.0:
        enhancer = ImageEnhance.Brightness(texture_image)
        texture_image = enhancer.enhance(intensity)
    primitive = node.mesh.primitives[0]
    assert isinstance(primitive, Primitive)
    primitive.material.baseColorTexture.source = texture_image
    primitive.material.baseColorTexture.sampler.minFilter = GL_LINEAR_MIPMAP_LINEAR


def build_scene(colors, floor_textures, wall_textures, objects):
    scene = Scene(
        bg_color=np.array([153 / 255, 226 / 255, 249 / 255]),
        ambient_light=np.array([0.5, 0.5, 0.5, 1.0]))

    floor_trimesh = trimesh.load("objects/floor.obj")
    mesh = Mesh.from_trimesh(floor_trimesh)
    node = Node(
        mesh=mesh,
        rotation=pyrender.quaternion.from_pitch(-math.pi / 2),
        translation=np.array([0, 0, 0]))
    texture_path = random.choice(floor_textures)
    set_random_texture(node, texture_path, intensity=0.8)
    scene.add_node(node)

    texture_path = random.choice(wall_textures)

    wall_trimesh = trimesh.load("objects/wall.obj")
    mesh = Mesh.from_trimesh(wall_trimesh)
    node = Node(mesh=mesh, translation=np.array([0, 1.15, -3.5]))
    set_random_texture(node, texture_path)
    scene.add_node(node)

    mesh = Mesh.from_trimesh(wall_trimesh)
    node = Node(
        mesh=mesh,
        rotation=pyrender.quaternion.from_yaw(math.pi),
        translation=np.array([0, 1.15, 3.5]))
    set_random_texture(node, texture_path)
    scene.add_node(node)

    mesh = Mesh.from_trimesh(wall_trimesh)
    node = Node(
        mesh=mesh,
        rotation=pyrender.quaternion.from_yaw(-math.pi / 2),
        translation=np.array([3.5, 1.15, 0]))
    set_random_texture(node, texture_path)
    scene.add_node(node)

    mesh = Mesh.from_trimesh(wall_trimesh)
    node = Node(
        mesh=mesh,
        rotation=pyrender.quaternion.from_yaw(math.pi / 2),
        translation=np.array([-3.5, 1.15, 0]))
    set_random_texture(node, texture_path)
    scene.add_node(node)

    # light = PointLight(color=np.ones(3), intensity=200.0)
    # node = Node(
    #     light=light,
    #     translation=np.array([0, 5, 5]))
    # scene.add_node(node)

    light = DirectionalLight(color=np.ones(3), intensity=10)
    position = np.array([0, 1, 1])
    position = position / np.linalg.norm(position)
    yaw, pitch = compute_yaw_and_pitch(position)
    node = Node(
        light=light,
        rotation=genearte_camera_quaternion(yaw, pitch),
        translation=np.array([0, 1, 1]))
    scene.add_node(node)

    # Place objects
    directions = [-1.0, 0.0, 1.0]
    available_positions = []
    for z in directions:
        for x in directions:
            available_positions.append((x, z))
    available_positions = np.array(available_positions)
    num_objects = random.choice(range(args.max_num_objects)) + 1
    indices = np.random.choice(
        np.arange(len(available_positions)), replace=False, size=num_objects)
    for xz in available_positions[indices]:
        node = random.choice(objects)()
        node.mesh.primitives[0].color_0 = random.choice(colors)
        if args.discrete_position == False:
            xz += np.random.uniform(-0.25, 0.25, size=xz.shape)
        parent = Node(children=[node], translation=np.array([xz[0], 0, xz[1]]))
        scene.add_node(parent)

    return scene


def udpate_vertex_buffer(cube_nodes):
    for node in (cube_nodes):
        node.mesh.primitives[0].update_vertex_buffer_data()


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


def genearte_camera_quaternion(yaw, pitch):
    quaternion_yaw = pyrender.quaternion.from_yaw(yaw)
    quaternion_pitch = pyrender.quaternion.from_pitch(pitch)
    quaternion = pyrender.quaternion.multiply(quaternion_pitch, quaternion_yaw)
    quaternion = quaternion / np.linalg.norm(quaternion)
    return quaternion


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
        scene = build_scene(colors, floor_textures, wall_textures, objects)
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
            camera_direction = np.array([camera_direction[0], 0, camera_direction[1]])
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

            # plt.clf()
            # plt.imshow(image)
            # plt.pause(0.1)

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
    parser.add_argument("--num-colors", type=int, default=10)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--anti-aliasing", default=False, action="store_true")
    parser.add_argument(
        "--discrete-position", default=False, action="store_true")
    args = parser.parse_args()
    main()
