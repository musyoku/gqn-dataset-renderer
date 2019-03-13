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

from archiver import Archiver, SceneData
from pyrender import (DirectionalLight, Mesh, Node, OffscreenRenderer,
                      OrthographicCamera, RenderFlags, Scene)

cube_size = 0.25


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
        shift = cube_size
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


def build_scene(color_candidates):
    # Generate positions of each cube
    cube_position_array, center_of_gravity = generate_block_positions(
        args.num_cubes)
    assert len(cube_position_array) == args.num_cubes

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
                position[0] - center_of_gravity[0],
                position[1] - center_of_gravity[1],
                position[2] - center_of_gravity[2],
            ])))
        scene.add_node(node)
        cube_nodes.append(node)

    update_cube_color_and_position(cube_nodes, color_candidates)

    # Place a light
    light = DirectionalLight(color=np.ones(3), intensity=20.0)
    quaternion_yaw = generate_quaternion(yaw=(math.pi / 4))
    quaternion_pitch = generate_quaternion(pitch=(-math.pi / 5))
    quaternion = multiply_quaternion(quaternion_pitch, quaternion_yaw)
    quaternion = quaternion / np.linalg.norm(quaternion)
    node = Node(
        light=light, rotation=quaternion, translation=np.array([1, 1, 1]))
    scene.add_node(node)

    return scene, cube_nodes


def update_cube_color_and_position(cube_nodes, color_candidates):
    assert len(cube_nodes) == args.num_cubes

    # Generate positions of each cube
    cube_position_array, center_of_gravity = generate_block_positions(
        args.num_cubes)
    assert len(cube_position_array) == args.num_cubes

    for position, node in zip(cube_position_array, cube_nodes):
        color = np.array(random.choice(color_candidates))
        vertex_colors = np.broadcast_to(
            color, node.mesh.primitives[0].positions.shape)
        node.mesh.primitives[0].color_0 = vertex_colors
        node.translation = np.array(([
            position[0] - center_of_gravity[0],
            position[1] - center_of_gravity[1],
            position[2] - center_of_gravity[2],
        ]))


def udpate_vertex_buffer(cube_nodes):
    for node in (cube_nodes):
        node.mesh.primitives[0].update_vertex_buffer_data()


def generate_quaternion(yaw=None, pitch=None):
    if yaw is not None:
        return np.array([
            0,
            math.sin(yaw / 2),
            0,
            math.cos(yaw / 2),
        ])
    if pitch is not None:
        return np.array([
            math.sin(pitch / 2),
            0,
            0,
            math.cos(pitch / 2),
        ])
    raise NotImplementedError


def multiply_quaternion(A, B):
    a = A[3]
    b = B[3]
    U = A[:3]
    V = B[:3]
    W = a * V + b * U + np.cross(V, U)
    return np.array([W[0], W[1], W[2], a * b - U @ V])


def compute_yaw_and_pitch(position, distance):
    x, y, z = position
    if z < 0:
        yaw = math.pi + math.atan(x / z)
    elif x < 0:
        yaw = math.pi * 2 + math.atan(x / z)
    else:
        yaw = math.atan(x / z)
    pitch = -math.asin(y / distance)
    return yaw, pitch


def genearte_camera_quaternion(yaw, pitch):
    quaternion_yaw = generate_quaternion(yaw=yaw)
    quaternion_pitch = generate_quaternion(pitch=pitch)
    quaternion = multiply_quaternion(quaternion_pitch, quaternion_yaw)
    quaternion = quaternion / np.linalg.norm(quaternion)
    return quaternion


def main():
    try:
        os.makedirs(args.output_directory)
    except:
        pass

    # Initialize colors
    color_candidates = []
    for n in range(args.num_colors):
        hue = n / args.num_colors
        saturation = 1
        lightness = 1
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, lightness)
        color_candidates.append((red, green, blue))

    scene, cube_nodes = build_scene(color_candidates)
    camera = OrthographicCamera(xmag=0.9, ymag=0.9)
    camera_node = Node(camera=camera)
    scene.add_node(camera_node)
    renderer = OffscreenRenderer(
        viewport_width=args.image_size, viewport_height=args.image_size)

    camera_distance = 2

    archiver = Archiver(
        directory=args.output_directory,
        total_scenes=args.total_scenes,
        num_scenes_per_file=min(args.num_scenes_per_file, args.total_scenes),
        image_size=(args.image_size, args.image_size),
        num_observations_per_scene=args.num_observations_per_scene,
        initial_file_number=args.initial_file_number)

    for scene_index in tqdm(range(args.total_scenes)):

        scene_data = SceneData((args.image_size, args.image_size),
                               args.num_observations_per_scene)
        for observation_index in range(args.num_observations_per_scene):
            # Generate random point on a sphere
            camera_position = np.random.normal(size=3)
            camera_position = camera_distance * camera_position / np.linalg.norm(
                camera_position)
            # Compute yaw and pitch
            yaw, pitch = compute_yaw_and_pitch(camera_position,
                                               camera_distance)

            camera_node.rotation = genearte_camera_quaternion(yaw, pitch)
            camera_node.translation = camera_position

            # Rendering
            image = renderer.render(
                scene,
                flags=(RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.OFFSCREEN
                       | RenderFlags.ALL_SOLID))[0]
            scene_data.add(image, camera_position, math.cos(yaw),
                           math.sin(yaw), math.cos(pitch), math.sin(pitch))
            # plt.clf()
            # plt.imshow(image)
            # plt.pause(0.1)

        archiver.add(scene_data)

        # Change cube color and position
        update_cube_color_and_position(cube_nodes, color_candidates)

        # Transfer changes to the vertex buffer on gpu
        udpate_vertex_buffer(cube_nodes)

    renderer.delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-scenes", "-total", type=int, default=2000000)
    parser.add_argument("--num-scenes-per-file", type=int, default=2000)
    parser.add_argument("--initial-file-number", type=int, default=1)
    parser.add_argument("--num-observations-per-scene", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-cubes", type=int, default=5)
    parser.add_argument("--num-colors", type=int, default=10)
    parser.add_argument("--output-directory", type=str, required=True)
    args = parser.parse_args()
    main()
