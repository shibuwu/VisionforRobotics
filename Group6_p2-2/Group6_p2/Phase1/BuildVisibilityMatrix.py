import numpy as np


def build_visibility_matrix(num_cameras, num_points, camera_indices, point_indices):
    V = np.zeros((num_cameras, num_points), dtype=np.uint8)
    for cam_idx, pt_idx in zip(camera_indices, point_indices):
        V[int(cam_idx), int(pt_idx)] = 1
    return V

