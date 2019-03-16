import trimesh
from . import Mesh, Node


def get_sphere_node():
    sphere = trimesh.creation.capsule(radius=0.25, height=0)
    mesh = Mesh.from_trimesh(sphere, smooth=True)
    node = Node(mesh=mesh, translation=np.array([0, 0, 0]))
    return node


def get_capsule_node():
    sphere = trimesh.creation.capsule(radius=0.25, height=0.5)
    mesh = Mesh.from_trimesh(sphere, smooth=True)
    node = Node(
        mesh=mesh,
        rotation=generate_quaternion(pitch=math.pi / 2),
        translation=np.array([0, 1.0, 0]))
    return node


def get_cylinder_node():
    sphere = trimesh.creation.cylinder(radius=0.25, height=0.5)
    mesh = Mesh.from_trimesh(sphere, smooth=True)
    node = Node(
        mesh=mesh,
        rotation=generate_quaternion(pitch=math.pi / 2),
        translation=np.array([0, 0, 0]))
    return node
