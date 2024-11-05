import numpy as np
import scipy.io as sio
import os
import pickle
from plyfile import PlyData, PlyElement
import trimesh

class SplatContainer(object):
    """
    Helper class to store points and their features (e.g., color) as numpy arrays.
    
    Controls I/O, conversions, and other utilities for splats (point clouds).
    """
    def __init__(self, points=None, features=None):
        if (points is None and features is None): return
        if (not type(points).__module__ == np.__name__):
            points = points.cpu().detach().numpy()
        if (features is not None and not type(features).__module__ == np.__name__):
            features = features.cpu().detach().numpy()
        
        self.points, self.features = points, features
        self.n = self.points.shape[0]
    
    def copy(self):
        return SplatContainer(self.points.copy(), self.features.copy())

    def _load_from_file_ply(self, file_path):
        self.load_file_name = file_path
        ply_data = PlyData.read(file_path)
        vx = np.array(ply_data['vertex'].data['x'])[:, np.newaxis]
        vy = np.array(ply_data['vertex'].data['y'])[:, np.newaxis]
        vz = np.array(ply_data['vertex'].data['z'])[:, np.newaxis]
        self.points = np.concatenate((vx, vy, vz), axis=1)
        
        if 'red' in ply_data['vertex'].data.dtype.names:
            red = np.array(ply_data['vertex'].data['red'])[:, np.newaxis]
            green = np.array(ply_data['vertex'].data['green'])[:, np.newaxis]
            blue = np.array(ply_data['vertex'].data['blue'])[:, np.newaxis]
            self.features = np.concatenate((red, green, blue), axis=1) / 255.0  # Normalize to [0, 1]
        else:
            self.features = None
        
        self.n = self.points.shape[0]

    def load_from_file(self, file_path):
        self.file_path = file_path
        filename = os.path.basename(file_path)
        if filename.endswith('ply'):
            self._load_from_file_ply(file_path)
        return self
    
    def save_to_ply(self, file_path=''):
        if file_path == '':
            file_path = self.load_file_name
        
        vertex = np.array([tuple(p) for p in self.points], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        if self.features is not None:
            features = np.array([tuple(c * 255.0) for c in self.features], dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")])
            vertex = np.array([(*v, *f) for v, f in zip(vertex, features)], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        
        el = PlyElement.describe(vertex, "vertex")
        plydata = PlyData([el])
        plydata.write(file_path)

    def save_as_mat(self, file_path=''):
        if file_path == '':
            file_path = self.file_path
        data = {'points': self.points.tolist()}
        if self.features is not None:
            data['features'] = self.features.tolist()
        sio.savemat(file_path[:-4] + '.mat', data)

    def compute_distances(self):
        """
        Compute pairwise distances between points.
        """
        from scipy.spatial import distance_matrix
        return distance_matrix(self.points, self.points)

# Example usage
if __name__ == "__main__":
    ply_file = 'path_to_your_splat_file.ply'
    
    # Load the point cloud data from PLY file
    splat_container = SplatContainer().load_from_file(ply_file)
    
    # Print loaded points and features (if any)
    print("Loaded points:", splat_container.points)
    print("Loaded features:", splat_container.features)
    
    # Save the point cloud data to a new PLY file
    splat_container.save_to_ply('saved_splat_file.ply')
    
    # Save the point cloud data to a MAT file
    splat_container.save_as_mat('saved_splat_file.mat')
    
    # Compute pairwise distances between points
    distances = splat_container.compute_distances()
    print("Pairwise distances:", distances)
