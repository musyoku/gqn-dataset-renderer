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

from pyrender import (RenderFlags, OrthographicCamera, OffscreenRenderer, Node)
from shepard_metzler import (compute_yaw_and_pitch, genearte_camera_quaternion,
                             build_scene)


def main():
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

    rad_step = math.pi / 18
    total_frames = int(math.pi * 2 / rad_step)
    camera_distance = 2

    fig = plt.figure(figsize=(3, 3))
    ims = []

    for num_cubes in range(1, 8):
        scene = build_scene(num_cubes, color_candidates)[0]
        camera = OrthographicCamera(xmag=0.9, ymag=0.9)
        camera_node = Node(camera=camera)
        scene.add_node(camera_node)

        current_rad = 0
        
        for _ in range(total_frames):
            camera_position = np.array((math.sin(current_rad),
                                        math.sin(math.pi / 6),
                                        math.cos(current_rad)))
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
            im = plt.imshow(image, interpolation="none", animated=True)
            ims.append([im])

            current_rad += rad_step

    renderer.delete()

    ani = animation.ArtistAnimation(
        fig, ims, interval=1 / 24, blit=True, repeat_delay=0)
    ani.save("shepard_metzler.gif", writer="imagemagick")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-colors", "-colors", type=int, default=10)
    parser.add_argument("--anti-aliasing", default=False, action="store_true")
    args = parser.parse_args()
    main()
