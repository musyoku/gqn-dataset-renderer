import math

import numpy as np
import trimesh

from . import Mesh, Node, quaternion


def Sphere():
    sphere = trimesh.creation.capsule(radius=0.375, height=0)
    mesh = Mesh.from_trimesh(sphere, smooth=True)
    node = Node(mesh=mesh, translation=np.array([0, 0.375, 0]))
    return node


def Box():
    sphere = trimesh.creation.box(extents=np.array([0.75, 0.75, 0.75]))
    mesh = Mesh.from_trimesh(sphere, smooth=False)
    node = Node(mesh=mesh, translation=np.array([0, 0.375, 0]))
    return node


def Capsule():
    sphere = trimesh.creation.capsule(radius=0.25, height=0.75)
    mesh = Mesh.from_trimesh(sphere, smooth=True)
    node = Node(
        mesh=mesh,
        rotation=quaternion.from_pitch(math.pi / 2),
        translation=np.array([0, 0.75, 0]))
    return node


def Cylinder():
    sphere = trimesh.creation.cylinder(radius=0.375, height=0.75)
    mesh = Mesh.from_trimesh(sphere, smooth=False)
    node = Node(
        mesh=mesh,
        rotation=quaternion.from_pitch(math.pi / 2),
        translation=np.array([0, 0.375, 0]))
    return node


def Icosahedron():
    sphere = trimesh.creation.icosahedron()
    mesh = Mesh.from_trimesh(sphere, smooth=False)
    node = Node(
        mesh=mesh,
        scale=np.array([0.35, 0.35, 0.35]),
        translation=np.array([0, 0.35, 0]))
    return node
