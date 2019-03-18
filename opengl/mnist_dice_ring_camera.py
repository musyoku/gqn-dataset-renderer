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
from OpenGL.GL import GL_LINEAR_MIPMAP_LINEAR
from PIL import Image
from tqdm import tqdm

import pyrender
from archiver import Archiver, SceneData
from pyrender import (DirectionalLight, Mesh, Node, OffscreenRenderer,
                      PerspectiveCamera, PointLight, Primitive, RenderFlags,
                      Scene)
from rooms_ring_camera import (
    compute_yaw_and_pitch, genearte_camera_quaternion, set_random_texture,
    udpate_vertex_buffer, build_scene, floor_textures, wall_textures, objects)


def load_mnist_images():
    import chainer
    train = chainer.datasets.get_mnist()[0]
    image_array = []
    for k in range(100):
        image = train[k][0] * 255
        image = image.reshape((28, 28, 1))
        image = np.repeat(image, 3, axis=2)
        image = np.uint8(image)
        image_array.append(image)
    return image_array


def generate_mnist_texture(mnist_images):
    indices = np.random.choice(
        np.arange(len(mnist_images)), replace=False, size=6)
    texture = np.zeros((28 * 4, 28 * 4, 3), dtype=np.uint8)
    for k, image_index in enumerate(indices):
        image = mnist_images[image_index]
        xi = (k % 4) * 28
        yi = (3 - k // 4) * 28
        texture[yi:yi + 28, xi:xi + 28] = image

    return texture


def place_dice(scene, mnist_images, discrete_position=False,
               rotate_dice=False):
    dice_trimesh = trimesh.load("objects/dice.obj")
    mesh = Mesh.from_trimesh(dice_trimesh, smooth=False)
    node = Node(
        mesh=mesh,
        scale=np.array([0.75, 0.75, 0.75]),
        translation=np.array([0, 0.75, 0]))
    texture_image = generate_mnist_texture(mnist_images)
    primitive = node.mesh.primitives[0]
    primitive.material.baseColorTexture.source = texture_image
    primitive.material.baseColorTexture.sampler.minFilter = GL_LINEAR_MIPMAP_LINEAR

    directions = [-1.0, 0.0, 1.0]
    available_positions = []
    for z in directions:
        for x in directions:
            available_positions.append((x, z))
    xz = np.array(random.choice(available_positions))

    if discrete_position == False:
        xz += np.random.uniform(-0.25, 0.25, size=xz.shape)
    if rotate_dice:
        yaw = np.random.uniform(0, math.pi * 2, size=1)[0]
        rotation = pyrender.quaternion.from_yaw(yaw)
        parent = Node(
            children=[node],
            rotation=rotation,
            translation=np.array([xz[0], 0, xz[1]]))
    else:
        parent = Node(children=[node], translation=np.array([xz[0], 0, xz[1]]))
    scene.add_node(parent)


def main():
    try:
        os.makedirs(args.output_directory)
    except:
        pass

    # Load MNIST images
    mnist_images = load_mnist_images()

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
        scene = build_scene(
            floor_textures,
            wall_textures,
            fix_light_position=args.fix_light_position)
        place_dice(
            scene,
            mnist_images,
            discrete_position=args.discrete_position,
            rotate_dice=args.rotate_dice)

        camera_distance = 4
        camera = PerspectiveCamera(yfov=math.pi / 4)
        camera_node = Node(camera=camera, translation=np.array([0, 1, 1]))
        scene.add_node(camera_node)
        scene_data = SceneData((args.image_size, args.image_size),
                               args.num_observations_per_scene)
        for observation_index in range(args.num_observations_per_scene):
            rand_position_xz = np.random.normal(size=2)
            rand_position_xz = camera_distance * rand_position_xz / np.linalg.norm(
                rand_position_xz)

            # Compute yaw and pitch
            camera_direction = np.array(
                [rand_position_xz[0], 0, rand_position_xz[1]])
            yaw, pitch = compute_yaw_and_pitch(camera_direction)

            camera_node.rotation = genearte_camera_quaternion(yaw, pitch)
            camera_position = np.array(
                [rand_position_xz[0], 1, rand_position_xz[1]])
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
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--anti-aliasing", default=False, action="store_true")
    parser.add_argument(
        "--discrete-position", default=False, action="store_true")
    parser.add_argument("--rotate-dice", default=False, action="store_true")
    parser.add_argument(
        "--fix-light-position", default=False, action="store_true")
    parser.add_argument("--visualize", default=False, action="store_true")
    args = parser.parse_args()
    main()
