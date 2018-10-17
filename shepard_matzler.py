import math
import time
import cv2

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import rtx

scene = rtx.Scene()

box_width = 6
box_height = 5

# 1
geometry = rtx.PlainGeometry(box_width, box_height)
geometry.set_rotation((0, 0, 0))
geometry.set_position((0, 0, -box_width / 2))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# 2
geometry = rtx.PlainGeometry(box_width, box_height)
geometry.set_rotation((0, -math.pi / 2, 0))
geometry.set_position((box_width / 2, 0, 0))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# 3
geometry = rtx.PlainGeometry(box_width, box_height)
geometry.set_rotation((0, math.pi, 0))
geometry.set_position((0, 0, box_width / 2))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# 4
geometry = rtx.PlainGeometry(box_width, box_height)
geometry.set_rotation((0, math.pi / 2, 0))
geometry.set_position((-box_width / 2, 0, 0))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# ceil
geometry = rtx.PlainGeometry(box_width, box_width)
geometry.set_rotation((math.pi / 2, 0, 0))
geometry.set_position((0, box_height / 2, 0))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
ceil = rtx.Object(geometry, material, mapping)
scene.add(ceil)

# floor
geometry = rtx.PlainGeometry(box_width, box_width)
geometry.set_rotation((-math.pi / 2, 0, 0))
geometry.set_position((0, -box_height / 2, 0))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
ceil = rtx.Object(geometry, material, mapping)
scene.add(ceil)

# light
geometry = rtx.PlainGeometry(box_width / 2, box_width / 2)
geometry.set_rotation((0, math.pi / 2, 0))
geometry.set_position((0.01 - box_width / 2, -box_height / 4, 0))
material = rtx.EmissiveMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
light = rtx.Object(geometry, material, mapping)
scene.add(light)

geometry = rtx.PlainGeometry(box_width / 2, box_width / 2)
geometry.set_rotation((0, -math.pi / 2, 0))
geometry.set_position((box_width / 2 - 0.01, -box_height / 4, 0))
material = rtx.EmissiveMaterial(1.0)
mapping = rtx.SolidColorMapping((0, 1, 1))
light = rtx.Object(geometry, material, mapping)
scene.add(light)

geometry = rtx.BoxGeometry(3, 3, 3)
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
box = rtx.Object(geometry, material, mapping)
scene.add(box)

screen_width = 64
screen_height = 64

rt_args = rtx.RayTracingArguments()
rt_args.num_rays_per_pixel = 1024
rt_args.max_bounce = 3
rt_args.next_event_estimation_enabled = False

cuda_args = rtx.CUDAKernelLaunchArguments()
cuda_args.num_threads = 64
cuda_args.num_rays_per_thread = 64

renderer = rtx.Renderer()

camera = rtx.PerspectiveCamera(
    eye=(0, 0.0, 6),
    center=(0, 0.0, 0),
    up=(0, 1, 0),
    fov_rad=math.pi / 3,
    aspect_ratio=screen_width / screen_height,
    z_near=0.01,
    z_far=100)

render_buffer = np.zeros((screen_height, screen_width, 3), dtype="float32")
total_iterations = 30
camera_rad = 0
radius = 6
start = time.time()
for n in range(total_iterations):
    renderer.render(scene, camera, rt_args, cuda_args, render_buffer)
    # linear -> sRGB
    pixels = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
    # pixels = cv2.medianBlur(pixels, 3)
    # pixels = cv2.bilateralFilter(pixels, 2, 1, 0)

    plt.imshow(pixels, interpolation="none")
    plt.pause(1e-8)

    camera_rad += math.pi / 10
    eye = (radius * math.sin(camera_rad), 0.0, radius * math.cos(camera_rad))
    camera.look_at(eye=eye, center=(0, 0.0, 0), up=(0, 1, 0))

print(total_iterations / (time.time() - start))
image = Image.fromarray(np.uint8(pixels * 255))
image.save("result.png")
