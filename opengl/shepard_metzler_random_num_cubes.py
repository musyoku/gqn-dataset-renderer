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
                      OrthographicCamera, RenderFlags, Scene)

cube_size = 0.25
combinations = [1, 6, 30, 150, 726, 3534, 16926, 81390, 387966, 1853886]


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


def build_scene(max_num_cubes, color_candidates, probabilities):
    num_cubes = np.random.choice(
        np.arange(1, max_num_cubes + 1), size=1, p=probabilities)[0]

    # Generate positions of each cube
    cube_position_array, barycenter = generate_block_positions(num_cubes)
    assert len(cube_position_array) == num_cubes

    # Place cubes
    scene = Scene(
        bg_color=np.array([0.0, 0.0, 0.0]),
        ambient_light=np.array([0.3, 0.3, 0.3, 1.0]))
    cube_nodes = []
    for position in cube_position_array:
        mesh = trimesh.creation.box(extents=cube_size * np.ones(3))
        mesh = Mesh.from_trimesh(mesh, smooth=False)
        node = Node(
            mesh=mesh,
            translation=np.array(([
                position[0] - barycenter[0],
                position[1] - barycenter[1],
                position[2] - barycenter[2],
            ])))
        scene.add_node(node)
        cube_nodes.append(node)

    # Generate positions of each cube
    cube_position_array, barycenter = generate_block_positions(num_cubes)

    for position, node in zip(cube_position_array, cube_nodes):
        color = np.array(random.choice(color_candidates))
        vertex_colors = np.broadcast_to(
            color, node.mesh.primitives[0].positions.shape)
        node.mesh.primitives[0].color_0 = vertex_colors
        node.translation = np.array(([
            position[0] - barycenter[0],
            position[1] - barycenter[1],
            position[2] - barycenter[2],
        ]))

    # Place a light
    light = DirectionalLight(color=np.ones(3), intensity=15.0)
    quaternion_yaw = pyrender.quaternion.from_yaw(math.pi / 4)
    quaternion_pitch = pyrender.quaternion.from_pitch(-math.pi / 5)
    quaternion = pyrender.quaternion.multiply(quaternion_pitch, quaternion_yaw)
    quaternion = quaternion / np.linalg.norm(quaternion)
    node = Node(
        light=light, rotation=quaternion, translation=np.array([1, 1, 1]))
    scene.add_node(node)

    return scene, cube_nodes


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

    assert args.max_num_cubes <= 10

    probabilities = np.array(combinations[:args.max_num_cubes]) / np.sum(
        np.array(combinations[:args.max_num_cubes]))

    last_file_number = args.initial_file_number + args.total_scenes // args.num_scenes_per_file - 1
    initial_file_number = args.initial_file_number
    if os.path.isdir(args.output_directory):
        files = os.listdir(args.output_directory)
        for name in files:
            number = int(name.replace(".h5", ""))
            if number > last_file_number:
                continue
            if number < args.initial_file_number:
                continue
            if number < initial_file_number:
                continue
            initial_file_number = number + 1
    total_scenes_to_render = args.total_scenes - args.num_scenes_per_file * (
        initial_file_number - args.initial_file_number)

    assert args.num_scenes_per_file <= total_scenes_to_render

    # Initialize colors
    color_candidates = []
    for n in range(args.num_colors):
        hue = n / args.num_colors
        saturation = 1
        lightness = 1
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, lightness)
        color_candidates.append((red, green, blue))

    renderer = OffscreenRenderer(
        viewport_width=args.image_size, viewport_height=args.image_size)

    archiver = Archiver(
        directory=args.output_directory,
        num_scenes_per_file=args.num_scenes_per_file,
        image_size=(args.image_size, args.image_size),
        num_observations_per_scene=args.num_observations_per_scene,
        initial_file_number=initial_file_number)

    for scene_index in tqdm(range(total_scenes_to_render)):
        scene, cube_nodes = build_scene(args.max_num_cubes, color_candidates,
                                        probabilities)
        camera = OrthographicCamera(xmag=0.9, ymag=0.9)
        camera_node = Node(camera=camera)
        scene.add_node(camera_node)

        camera_distance = 2
        scene_data = SceneData((args.image_size, args.image_size),
                               args.num_observations_per_scene)
        for observation_index in range(args.num_observations_per_scene):
            # Generate random point on a sphere
            camera_position = np.random.normal(size=3)
            camera_position = camera_distance * camera_position / np.linalg.norm(
                camera_position)
            # Compute yaw and pitch
            yaw, pitch = compute_yaw_and_pitch(camera_position)

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
    parser.add_argument("--num-observations-per-scene", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-num-cubes", type=int, default=5)
    parser.add_argument("--num-colors", type=int, default=10)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--anti-aliasing", default=False, action="store_true")
    parser.add_argument("--visualize", default=False, action="store_true")
    args = parser.parse_args()
    main()
