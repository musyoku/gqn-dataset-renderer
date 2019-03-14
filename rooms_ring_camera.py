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
from pyrender import (PointLight, DirectionalLight, Mesh, Node,
                      OffscreenRenderer, PerspectiveCamera, RenderFlags, Scene,
                      Viewer)


def build_scene(color_candidates):
    scene = Scene(
        bg_color=np.array([153 / 255, 226 / 255, 249 / 255]),
        ambient_light=np.array([0.5, 0.5, 0.5, 1.0]))

    floor_trimesh = trimesh.load("models/floor_1.obj")
    mesh = Mesh.from_trimesh(floor_trimesh)
    node = Node(
        mesh=mesh,
        rotation=generate_quaternion(pitch=-math.pi / 2),
        translation=np.array([0, 0, 0]))
    scene.add_node(node)

    wall_trimesh = trimesh.load("models/wall_1.obj")
    mesh = Mesh.from_trimesh(wall_trimesh)
    node = Node(mesh=mesh, translation=np.array([0, 1.15, -3.5]))
    scene.add_node(node)

    mesh = Mesh.from_trimesh(wall_trimesh)
    node = Node(
        mesh=mesh,
        rotation=generate_quaternion(yaw=math.pi),
        translation=np.array([0, 1.15, 3.5]))
    scene.add_node(node)

    mesh = Mesh.from_trimesh(wall_trimesh)
    node = Node(
        mesh=mesh,
        rotation=generate_quaternion(yaw=-math.pi / 2),
        translation=np.array([3.5, 1.15, 0]))
    scene.add_node(node)

    mesh = Mesh.from_trimesh(wall_trimesh)
    node = Node(
        mesh=mesh,
        rotation=generate_quaternion(yaw=math.pi / 2),
        translation=np.array([-3.5, 1.15, 0]))
    scene.add_node(node)

    # light = PointLight(color=np.ones(3), intensity=100.0)
    # node = Node(
    #     light=light,
    #     translation=np.array([0, 5, 5]))
    # scene.add_node(node)

    camera_distance = 5
    camera_position = np.array([0, 5, 5])
    camera_position = camera_distance * camera_position / np.linalg.norm(
        camera_position)
    # Compute yaw and pitch
    yaw, pitch = compute_yaw_and_pitch(camera_position,
                                        camera_distance)
                                        
    light = DirectionalLight(color=np.ones(3), intensity=10)
    node = Node(
        light=light,
        rotation=genearte_camera_quaternion(yaw, pitch),
        translation=np.array([0, 5, 0]))
    scene.add_node(node)

    capsule_trimesh = trimesh.creation.capsule(radius=0.5, height=0)
    mesh = Mesh.from_trimesh(capsule_trimesh, smooth=True)
    node = Node(mesh=mesh, translation=np.array([0, 0.5, 0]))
    scene.add_node(node)

    return scene


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

    scene = build_scene(color_candidates)
    camera = PerspectiveCamera(yfov=math.pi / 4)
    camera_node = Node(camera=camera, translation=np.array([0, 1, 1]))
    scene.add_node(camera_node)
    renderer = OffscreenRenderer(
        viewport_width=args.image_size, viewport_height=args.image_size)

    v = Viewer(scene, shadows=True, viewport_size=(400, 400))

    # Rendering
    image = renderer.render(
        scene,
        flags=(RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.OFFSCREEN
               | RenderFlags.ALL_SOLID))[0]
    plt.clf()
    plt.imshow(image)
    plt.show()

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
