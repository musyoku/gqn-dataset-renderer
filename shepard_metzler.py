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
from pyrender import (DirectionalLight, Mesh, MetallicRoughnessMaterial, Node,
                      OffscreenRenderer, PerspectiveCamera, PointLight,
                      Primitive, Scene, SpotLight, Viewer)
from tqdm import tqdm

import gqn

pyglet.options["shadow_window"] = False


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
    cube_position_array, center_of_gravity = generate_block_positions(
        args.num_cubes)
    assert len(cube_position_array) == args.num_cubes

    # Place cubes
    scene = Scene(
        bg_color=np.array([0.0, 0.0, 0.0]),
        ambient_light=np.array([0.2, 0.2, 0.2, 1.0]))
    for position in cube_position_array:
        boxv_trimesh = trimesh.creation.box(extents=1 * np.ones(3))
        color = np.array(random.choice(color_array))
        vertex_colors = np.broadcast_to(color, boxv_trimesh.vertices.shape)
        boxv_trimesh.visual.vertex_colors = vertex_colors
        boxv_mesh = Mesh.from_trimesh(boxv_trimesh, smooth=False)

        node = Node(
            mesh=boxv_mesh,
            translation=np.array(([
                position[0] - center_of_gravity[0],
                position[1] - center_of_gravity[1],
                position[2] - center_of_gravity[2],
            ])))
        scene.add_node(node)

    light = PointLight(color=np.ones(3), intensity=0.0)
    light = DirectionalLight(color=np.ones(3), intensity=10.0)
    node = Node(light=light, translation=np.array([0, 5, 5]))
    scene.add_node(node)

    return scene


def multiply_quaternion(A, B):
    a = A[3]
    b = B[3]
    U = A[:3]
    V = B[:3]
    W = a * V + b * U + U * V
    return np.array([W[0], W[1], W[2], a * b - U @ V])


def main():
    # Initialize colors
    color_array = []
    for n in range(args.num_colors):
        hue = n / (args.num_colors - 1)
        saturation = 0.9
        lightness = 1
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, lightness)
        color_array.append((red, green, blue))

    screen_width = args.image_size
    screen_height = args.image_size

    scene = build_scene(color_array)
    camera = PerspectiveCamera(yfov=(np.pi / 3.0))
    camera_node = Node(
        camera=camera,
        rotation=np.array([0,
                           math.sin(math.pi / 2), 0,
                           math.cos(math.pi / 2)]),
        translation=np.array([0, 0, -5]))
    scene.add_node(camera_node)
    r = OffscreenRenderer(viewport_width=64, viewport_height=64)
    start_time = time.time()
    for k in range(1000):
        yaw = math.pi * 2 * k / 100
        pitch = -math.pi / 4
        quaternion_yaw = np.array([
            0,
            math.sin(yaw / 2),
            0,
            math.cos(yaw / 2),
        ])
        quaternion_pitch = np.array([
            math.sin(pitch / 2),
            0,
            0,
            math.cos(pitch / 2),
        ])
        quaternion = multiply_quaternion(quaternion_pitch, quaternion_yaw)
        quaternion = quaternion / np.linalg.norm(quaternion)
        camera_node.rotation = quaternion
        camera_node.translation = np.array(
            [5 * math.sin(yaw), 3, 5 * math.cos(yaw)])
        color, depth = r.render(scene)
        plt.clf()
        plt.imshow(color)
        plt.pause(1e-10)
    r.delete()

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

    for _ in tqdm(range(args.total_observations)):
        scene = build_scene(color_array)
        scene_data = gqn.archiver.SceneData((args.image_size, args.image_size),
                                            args.num_views_per_scene)

        view_radius = 3.3

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

boxv_trimesh = trimesh.creation.box(extents=0.1 * np.ones(3))
boxv_vertex_colors = np.zeros((boxv_trimesh.vertices.shape), dtype=np.float32)
boxv_vertex_colors[:, 0] = 1.0
boxv_trimesh.visual.vertex_colors = boxv_vertex_colors
boxv_mesh = Mesh.from_trimesh(boxv_trimesh, smooth=False)

direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
spot_l = SpotLight(
    color=np.ones(3),
    intensity=10.0,
    innerConeAngle=np.pi / 16,
    outerConeAngle=np.pi / 6)
point_l = PointLight(color=np.ones(3), intensity=10.0)

camera = PerspectiveCamera(yfov=(np.pi / 3.0))
cam_pose = np.array([[0.0, -np.sqrt(2) / 2,
                      np.sqrt(2) / 2, 0.5], [1.0, 0.0, 0.0, 0.0],
                     [0.0, np.sqrt(2) / 2,
                      np.sqrt(2) / 2, 0.4], [0.0, 0.0, 0.0, 1.0]])

scene = Scene(
    bg_color=np.array([0.0, 0.0, 0.0]),
    ambient_light=np.array([0.2, 0.2, 0.2, 1.0]))

#==============================================================================
# Adding objects to the scene
#==============================================================================

#------------------------------------------------------------------------------
# By manually creating nodes
#------------------------------------------------------------------------------
boxv_node = Node(mesh=boxv_mesh, translation=np.array([0.0, 0.0, 0.0]))
scene.add_node(boxv_node)

#------------------------------------------------------------------------------
# By using the add() utility function
#------------------------------------------------------------------------------

light_node = Node(light=point_l, translation=np.array([1, 1, 0]))
scene.add_node(light_node)

#==============================================================================
# Using the viewer with a default camera
#==============================================================================

# v = Viewer(scene, shadows=True)

#==============================================================================
# Using the viewer with a pre-specified camera
#==============================================================================
cam_node = scene.add(camera, pose=cam_pose)
v = Viewer(scene, central_node=boxv_node)

#==============================================================================
# Rendering offscreen from that camera
#==============================================================================
r = OffscreenRenderer(viewport_width=64, viewport_height=64)
start_time = time.time()
for k in range(1000):
    boxv_node.translation = np.array([k / 100, 0.0, 0.0])
    color, depth = r.render(scene)
    # plt.imshow(color)
    # plt.pause(1e-10)
print(1000 / (time.time() - start_time))
r.delete()