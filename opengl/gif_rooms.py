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

import pyrender
from pyrender import (RenderFlags, PerspectiveCamera, OrthographicCamera,
                      OffscreenRenderer, Node)
from rooms_ring_camera import (build_scene, place_objects,
                               compute_yaw_and_pitch,
                               genearte_camera_quaternion)


def main():
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

    plt.tight_layout()
    fig = plt.figure(figsize=(6, 3))
    axis_perspective = fig.add_subplot(1, 2, 1)
    axis_orthogonal = fig.add_subplot(1, 2, 2)
    ims = []

    scene = build_scene(
        floor_textures,
        wall_textures,
        fix_light_position=args.fix_light_position)
    place_objects(
        scene,
        colors,
        objects,
        min_num_objects=args.num_objects,
        max_num_objects=args.num_objects,
        discrete_position=args.discrete_position,
        rotate_object=args.rotate_object)

    camera_distance = 5
    perspective_camera = PerspectiveCamera(yfov=math.pi / 4)
    perspective_camera_node = Node(
        camera=perspective_camera, translation=np.array([0, 1, 1]))
    orthographic_camera = OrthographicCamera(xmag=3, ymag=3)
    orthographic_camera_node = Node(camera=orthographic_camera)

    rad_step = math.pi / 36
    total_frames = int(math.pi * 2 / rad_step)
    current_rad = 0
    for _ in range(total_frames):
        scene.add_node(perspective_camera_node)

        # Perspective camera
        camera_xz = camera_distance * np.array(
            (math.sin(current_rad), math.cos(current_rad)))
        # Compute yaw and pitch
        camera_direction = np.array([camera_xz[0], 0, camera_xz[1]])
        yaw, pitch = compute_yaw_and_pitch(camera_direction)

        perspective_camera_node.rotation = genearte_camera_quaternion(
            yaw, pitch)
        camera_position = np.array([camera_xz[0], 1, camera_xz[1]])
        perspective_camera_node.translation = camera_position

        # Rendering
        flags = RenderFlags.SHADOWS_DIRECTIONAL
        if args.anti_aliasing:
            flags |= RenderFlags.ANTI_ALIASING
        image = renderer.render(scene, flags=flags)[0]
        im1 = axis_perspective.imshow(
            image, interpolation="none", animated=True)
        scene.remove_node(perspective_camera_node)

        # Orthographic camera
        scene.add_node(orthographic_camera_node)
        camera_direction = camera_distance * np.array(
            (math.sin(current_rad), math.sin(math.pi / 6),
             math.cos(current_rad)))
        yaw, pitch = compute_yaw_and_pitch(camera_direction)

        orthographic_camera_node.rotation = genearte_camera_quaternion(
            yaw, pitch)
        orthographic_camera_node.translation = np.array(
            [camera_direction[0], 4, camera_direction[2]])

        image = renderer.render(scene, flags=flags)[0]

        im2 = axis_orthogonal.imshow(
            image, interpolation="none", animated=True)
        ims.append([im1, im2])

        plt.pause(1e-8)

        current_rad += rad_step
        scene.remove_node(orthographic_camera_node)

    ani = animation.ArtistAnimation(
        fig, ims, interval=1 / 24, blit=True, repeat_delay=0)
    filename = "rooms"
    if args.discrete_position:
        filename += "_discrete_position"
    if args.rotate_object:
        filename += "_rotate_object"
    if args.fix_light_position:
        filename += "_fix_light_position"
    filename += ".gif"
    ani.save(filename, writer="imagemagick")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-objects", "-objects", type=int, default=3)
    parser.add_argument("--num-colors", "-colors", type=int, default=6)
    parser.add_argument("--anti-aliasing", default=False, action="store_true")
    parser.add_argument(
        "--discrete-position", default=False, action="store_true")
    parser.add_argument("--rotate-object", default=False, action="store_true")
    parser.add_argument(
        "--fix-light-position", default=False, action="store_true")
    args = parser.parse_args()
    main()
