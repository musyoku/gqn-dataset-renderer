import os
import h5py
import numpy as np


class SceneData:
    def __init__(self, image_size, num_observations_per_scene):
        self.images = np.zeros(
            (num_observations_per_scene, ) + image_size + (3, ), dtype="uint8")
        self.viewpoints = np.zeros((num_observations_per_scene, 7),
                                   dtype="float32")
        self.view_index = 0
        self.num_observations_per_scene = num_observations_per_scene
        self.image_size = image_size

    def add(self, image, camera_position, cos_camera_yaw_rad,
            sin_camera_yaw_rad, cos_camera_pitch_rad, sin_camera_pitch_rad):
        assert isinstance(image, np.ndarray)
        assert isinstance(cos_camera_yaw_rad, float)
        assert isinstance(sin_camera_yaw_rad, float)
        assert isinstance(cos_camera_pitch_rad, float)
        assert isinstance(sin_camera_pitch_rad, float)
        assert image.ndim == 3
        assert image.shape[0] == self.image_size[0]
        assert image.shape[1] == self.image_size[1]
        assert image.shape[2] == 3
        assert len(camera_position) == 3
        assert self.view_index < self.num_observations_per_scene

        self.images[self.view_index] = image
        self.viewpoints[self.view_index] = (
            camera_position[0],
            camera_position[1],
            camera_position[2],
            cos_camera_yaw_rad,
            sin_camera_yaw_rad,
            cos_camera_pitch_rad,
            sin_camera_pitch_rad,
        )
        self.view_index += 1


class Archiver:
    def __init__(self,
                 directory,
                 num_scenes_per_file=2000,
                 image_size=(64, 64),
                 num_observations_per_scene=5,
                 initial_file_number=1):
        assert directory is not None
        self.images = np.zeros(
            (num_scenes_per_file, num_observations_per_scene) + image_size +
            (3, ),
            dtype="uint8")
        self.viewpoints = np.zeros(
            (num_scenes_per_file, num_observations_per_scene, 7),
            dtype="float32")
        self.current_num_observations = 0
        self.current_pool_index = 0
        self.current_file_number = initial_file_number
        self.num_scenes_per_file = num_scenes_per_file
        self.directory = directory
        self.image_size = image_size
        self.num_observations_per_scene = num_observations_per_scene
        self.total_scenes = 0
        try:
            os.mkdir(directory)
        except:
            pass

    def add(self, scene: SceneData):
        assert isinstance(scene, SceneData)

        self.images[self.current_pool_index] = scene.images
        self.viewpoints[self.current_pool_index] = scene.viewpoints

        self.current_pool_index += 1
        if self.current_pool_index >= self.num_scenes_per_file:
            self.save_subset()
            self.current_pool_index = 0
            self.current_file_number += 1
            self.total_scenes += self.num_scenes_per_file

    def save_subset(self):
        filename = "{:04d}.h5".format(self.current_file_number)
        with h5py.File(os.path.join(self.directory, filename), "w") as f:
            f.create_dataset("images", data=self.images)
            f.create_dataset("viewpoints", data=self.viewpoints)
