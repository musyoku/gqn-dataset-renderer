import numpy as np
import math


def from_yaw(yaw):
    return np.array([
        0,
        math.sin(yaw / 2),
        0,
        math.cos(yaw / 2),
    ])


def from_pitch(pitch):
    return np.array([
        math.sin(pitch / 2),
        0,
        0,
        math.cos(pitch / 2),
    ])


def multiply(A, B):
    a = A[3]
    b = B[3]
    U = A[:3]
    V = B[:3]
    W = a * V + b * U + np.cross(V, U)
    return np.array([W[0], W[1], W[2], a * b - U @ V])
