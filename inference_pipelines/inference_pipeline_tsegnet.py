import gen_utils as gu
import numpy as np
import torch
import ops_utils as tu
from sklearn.neighbors import KDTree
import open3d as o3d
from sklearn.cluster import DBSCAN

class InferencePipeLine:
    def __init__(self, model):
        self.model = model

        self.scaler = 1.8
        self.shifter = 0.8

    def __call__(self, stl_path):
        DEBUG=False
        _, mesh = gu.read_txt_obj_ls(stl_path, ret_mesh=True, use_tri_mesh=True) #TODO slow processing speed
        vertices = np.array(mesh.vertices)
        n_vertices = vertices.shape[0]
        vertices[:,:3] -= np.mean(vertices[:,:3], axis=0)
        vertices[:, :3] = ((vertices[:, :3]-np.min(vertices[:,1]))/(np.max(vertices[:,1])- np.min(vertices[:,1])))*self.scaler-self.shifter
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        org_feats = np.array(np.concatenate([np.array(mesh.vertices), np.array(mesh.vertex_normals)], axis=1))

        if np.asarray(mesh.vertices).shape[0] < 24000:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        vertices = np.array(np.concatenate([np.array(mesh.vertices), np.array(mesh.vertex_normals)], axis=1))
        sampled_feats = gu.resample_pcd([vertices.copy()], 24000, "fps")[0] #TODO slow processing speed

        input_cuda_feats = torch.from_numpy(np.array([sampled_feats.astype('float32')])).cuda().permute(0,2,1)

        with torch.no_grad():
            l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result = self.model.cent_module(input_cuda_feats)


        moved_points = gu.torch_to_numpy(l3_xyz + offset_result).T.reshape(-1,3)
        moved_points = moved_points[gu.torch_to_numpy(dist_result).reshape(-1)<0.3,:]
        dbscan_results = DBSCAN(eps=0.05, min_samples=3).fit(moved_points, 3)
        #gu.print_3d(gu.np_to_pcd_with_label(moved_points, dbscan_results.labels_))
        
        center_points = []
        for label in np.unique(dbscan_results.labels_):
            if label == -1: continue
            center_points.append(moved_points[dbscan_results.labels_==label].mean(axis=0))
        center_points = np.array(center_points)
        center_points = center_points[None,:,:]
        #gu.print_3d(gu.np_to_pcd(moved_points,color=[0,1,0]), center_points[0])
        
        nn_crop_indexes = tu.get_nearest_neighbor_idx(gu.torch_to_numpy(l0_xyz.permute(0,2,1)), center_points, 3072)

        cropped_input_ls = tu.get_indexed_features(input_cuda_feats, nn_crop_indexes)
        cropped_feature_ls = tu.get_indexed_features(l0_points, nn_crop_indexes)
        ddf = self.model.get_ddf(cropped_input_ls[:,:3,:].permute(0,2,1), center_points)

        cropped_feature_ls = torch.cat([cropped_input_ls[:,:3,:], cropped_feature_ls, ddf], axis=1)
        with torch.no_grad():
            pd_1, weight_1, pd_2, id_pred = self.model.seg_module(cropped_feature_ls)
        
        pred_labels = np.zeros(sampled_feats.shape[0])
        for i in range(cropped_feature_ls.shape[0]):
            pred_bin_labels = np.zeros(cropped_feature_ls[i].shape[1])
            #pred_bin_labels[gu.torch_to_numpy(pd_1.argmax(axis=1)[i])==1] = 1
            pred_bin_labels[gu.torch_to_numpy(torch.sigmoid(pd_2[i].reshape(-1)))>0.5] = 1
            #gu.print_3d(gu.np_to_pcd_with_label(gu.torch_to_numpy(cropped_feature_ls)[i,:3,:].T, pred_bin_labels))
            pred_labels[nn_crop_indexes[0][i][pred_bin_labels==1]] = gu.torch_to_numpy(id_pred.argmax(axis=1))[i]


        pred_labels[pred_labels>=9] += 2
        pred_labels[pred_labels>0] += 10
        
        tree = KDTree(sampled_feats[:,:3], leaf_size=2)
        near_points = tree.query(org_feats[:,:3], k=1, return_distance=False)
        result_ins_labels = pred_labels.reshape(-1)[near_points.reshape(-1)].reshape(-1,1)
        
        #gu.print_3d(gu.np_to_pcd_with_label(org_feats[:,:3], result_ins_labels))
        return {
            "sem":result_ins_labels.reshape(-1),
            "ins":result_ins_labels.reshape(-1),
        }       
