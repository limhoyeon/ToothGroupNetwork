from sklearn.neighbors import KDTree
import torch
import numpy as np
import gen_utils as gu
from sklearn.decomposition import PCA
class Augmentator:
    def __init__(self, augmentation_list):
        self.augmentation_list = augmentation_list
    
    def run(self, mesh_arr):
        for augmentation in self.augmentation_list:
            mesh_arr = augmentation.augment(mesh_arr)
        return mesh_arr

    def reload_vals(self):
        for augmentation in self.augmentation_list:
            augmentation.reload_val()

class Scaling:
    def __init__(self, trans_range):
        self.trans_range = trans_range
        assert self.trans_range[1] > self.trans_range[0]

    def augment(self, vert_arr):
        vert_arr[:,:3] = vert_arr[:,:3] * self.trans_val
        return vert_arr

    def reload_val(self):
        trans_val = np.random.rand(1)
        trans_val = (trans_val) * (self.trans_range[1]-self.trans_range[0]) + self.trans_range[0]
        self.trans_val = trans_val

class Rotation:
    def __init__(self, angle_range, angle_axis):
        self.angle_range = angle_range
        self.angle_axis = angle_axis
        assert self.angle_range[1] > self.angle_range[0]

    def augment(self, vert_arr):
        if self.angle_axis == "pca":
            pca_axis = PCA(n_components=3).fit(vert_arr[:,:3]).components_
            rotation_mat = pca_axis
            flap_rand = ((np.random.rand(3)>0.5).astype(np.float)-0.5)*2
            pca_axis[0] *= flap_rand[0]
            pca_axis[1] *= flap_rand[1]
            pca_axis[2] *= flap_rand[2]
        else:
            rotation_mat = gu.axis_rotation(self.angle_axis_val, self.rot_val)
        if type(vert_arr) == torch.Tensor:
            rotation_mat = torch.from_numpy(rotation_mat).type(torch.float32).cuda()
        vert_arr[:,:3] = (rotation_mat @ vert_arr[:,:3].T).T
        if vert_arr.shape[1]==6:
            vert_arr[:,3:] = (rotation_mat @ vert_arr[:,3:].T).T
        return vert_arr

    def reload_val(self):
        if self.angle_axis == "rand":
            self.angle_axis_val = np.random.rand(3)
            self.angle_axis_val /= np.linalg.norm(self.angle_axis_val)
        elif self.angle_axis == "fixed":
            self.angle_axis_val = np.array([0,0,1])
        elif self.angle_axis == "pca":
            pass
        else:
            raise "rotation augmentation parameter error"
        rot_val = np.random.rand(1)
        rot_val = (rot_val) * (self.angle_range[1]-self.angle_range[0]) + self.angle_range[0]
        self.rot_val = rot_val

class Translation:
    def __init__(self, trans_range):
        self.trans_range = trans_range
        assert self.trans_range[1] > self.trans_range[0]

    def augment(self, vert_arr):
        vert_arr[:,:3] = vert_arr[:,:3] + self.trans_val
        return vert_arr

    def reload_val(self):
        trans_val = np.random.rand(1,3)
        trans_val = (trans_val) * (self.trans_range[1]-self.trans_range[0]) + self.trans_range[0]
        self.trans_val = trans_val
