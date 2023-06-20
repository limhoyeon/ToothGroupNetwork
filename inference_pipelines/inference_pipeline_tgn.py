import gen_utils as gu
import numpy as np
from models.modules.grouping_network_module import GroupingNetworkModule
import torch
import ops_utils as tu
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
import open3d as o3d

class InferencePipeLine:
    def __init__(self, config):
        self.scaler = 1.8
        self.shifter = 0.8
        self.config = config
        
        self.first_module = GroupingNetworkModule(self.config["fps_model_info"])
        self.first_module.cuda()
        self.first_module.load_state_dict(torch.load(self.config["fps_model_info"]["load_ckpt_path"]))

        self.bdl_module = GroupingNetworkModule(self.config["boundary_model_info"])
        self.bdl_module.cuda()
        self.bdl_module.load_state_dict(torch.load(self.config["boundary_model_info"]["load_ckpt_path"]))

    def __call__(self, stl_path):
        DEBUG=False
        _, mesh = gu.read_txt_obj_ls(stl_path, ret_mesh=True, use_tri_mesh=True) #TODO slow processing speed
        mesh = mesh.remove_duplicated_vertices()
        vertices = np.array(mesh.vertices)
        n_vertices = vertices.shape[0]
        vertices[:,:3] -= np.mean(vertices[:,:3], axis=0)
        vertices[:, :3] = ((vertices[:, :3]-np.min(vertices[:,1]))/(np.max(vertices[:,1])- np.min(vertices[:,1])))*self.scaler-self.shifter
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        org_feats = np.array(np.concatenate([np.array(mesh.vertices), np.array(mesh.vertex_normals)], axis=1))

        if np.asarray(mesh.vertices).shape[0] < 24000:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            bdl_feats = np.array(np.concatenate([np.array(mesh.vertices), np.array(mesh.vertex_normals)], axis=1))
        else:
            bdl_feats = org_feats.copy()
            
        vertices = np.array(np.concatenate([np.array(mesh.vertices), np.array(mesh.vertex_normals)], axis=1))

        sampled_feats = gu.resample_pcd([vertices.copy()], 24000, "fps")[0] #TODO slow processing speed

        input_cuda_feats = torch.from_numpy(np.array([sampled_feats.astype('float32')])).cuda().permute(0,2,1)
        first_results = self.get_first_module_results(input_cuda_feats, self.first_module)

        sampled_boundary_feats, sampled_boundary_seg_label, only_boundary_feats, only_boundary_seg_label = self.get_boundary_sampled_feats(
            first_results["ins"]["full_ins_labeled_points"][:,3], 
            bdl_feats, 
            sampled_feats,
            None
        )

        input_cuda_bdl_feats = torch.from_numpy(np.array([sampled_boundary_feats.astype('float32')])).permute(0,2,1).cuda()
        sampled_boundary_seg_label = torch.from_numpy(np.array([sampled_boundary_seg_label.astype(int)])).permute(0,2,1).cuda() - 1
        bdl_results = self.get_second_module_results(input_cuda_bdl_feats, sampled_boundary_seg_label, self.bdl_module)
        
        if DEBUG: gu.print_3d(gu.np_to_pcd_with_label(first_results["ins"]["full_ins_labeled_points"]), gu.np_to_pcd_with_label(bdl_results["ins"]["full_ins_labeled_points"]))

        first_xyz = first_results["ins"]["full_ins_labeled_points"][:,:3]
        first_ps_label = first_results["ins"]["full_ins_labeled_points"][:,3].astype(int)
        first_sem_xyz = first_results["sem_1"]["full_labeled_points"][:,:3]
        first_sem_label = first_results["sem_1"]["full_labeled_points"][:,3]
        bdl_xyz = bdl_results["ins"]["full_ins_labeled_points"][:only_boundary_feats.shape[0],:3]
        bdl_ps_label = bdl_results["ins"]["full_ins_labeled_points"][:only_boundary_feats.shape[0],3].astype(int)

        gin_mean = np.mean(first_xyz[first_ps_label==0],axis=0).reshape(1,3)
        teeth_mean = np.mean(first_xyz[first_ps_label!=0],axis=0).reshape(1,3)
        

        first_ps_label_unique_except_zero = np.unique(first_ps_label); first_ps_label_unique_except_zero =first_ps_label_unique_except_zero[first_ps_label_unique_except_zero!=0]
        ins_label_center_points = np.array([np.mean(first_xyz[first_ps_label==label],axis=0) for label in first_ps_label_unique_except_zero])
        pca = PCA(n_components=3); pca.fit(ins_label_center_points); pca_axis = pca.components_
        pca_axis[2] = pca_axis[2] if np.dot((teeth_mean - gin_mean).reshape(3), pca_axis[2])>0 else -pca_axis[2]

        if np.where(first_sem_label==1)[0].shape[0] + np.where(first_sem_label==9)[0].shape[0] > 20:
            centerpoints_of_11_12 = np.array(np.mean([np.mean(first_sem_xyz[first_sem_label==1],axis = 0), np.mean(first_sem_xyz[first_sem_label==9], axis=0)], axis=0))
        else:
            for i in range(2,9):
                if np.where(first_sem_label==i)[0].shape[0]>20:
                    centerpoints_of_11_12 = np.array(np.mean([np.mean(first_sem_xyz[first_sem_label==i],axis = 0), np.mean(ins_label_center_points, axis=0)], axis=0))
                    break

            
        center_line = centerpoints_of_11_12 - np.mean(ins_label_center_points,axis=0)
        checking_axis_vector = np.cross(pca_axis[2], center_line)
        new_sem_labels = np.zeros(first_ps_label.shape[0])
        for ins_label in np.unique(first_ps_label): 
            if ins_label==0:
                continue
            ins_cluster_mask = first_ps_label == ins_label
            ins_points_center = first_xyz[ins_cluster_mask].mean(axis=0)
            sem_labels_in_cluster = first_sem_label[ins_cluster_mask]
            sem_labels_in_cluster = sem_labels_in_cluster[sem_labels_in_cluster!=0]
            if(sem_labels_in_cluster.shape[0]==0):
                new_sem_labels[ins_cluster_mask] = 0
                first_ps_label[ins_cluster_mask] = 0
                continue
            max_freq_first_cluster_label = np.argmax(np.bincount(sem_labels_in_cluster.astype(int)))
            if max_freq_first_cluster_label not in [1, 9]:
                if(np.dot(ins_points_center-centerpoints_of_11_12, checking_axis_vector)<0): max_freq_first_cluster_label = max_freq_first_cluster_label + 8
            new_sem_labels[ins_cluster_mask] = max_freq_first_cluster_label
        new_sem_labels = new_sem_labels.astype(int)

        #================boundary part ===========================#
        tree = KDTree(first_xyz, leaf_size=2)
        mod_bdl_ps_label = np.zeros((bdl_ps_label.shape[0]))
        mod_bdl_sem_label = np.zeros((bdl_ps_label.shape[0]))
        for bdl_cluster_label in np.unique(bdl_ps_label):
            if bdl_cluster_label==0:
                continue
            bdl_cluster_points = bdl_xyz[bdl_cluster_label == bdl_ps_label]
            cluster_near_points = tree.query(bdl_cluster_points, k=1, return_distance=False)

            cluster_first_cluster_label = first_ps_label[cluster_near_points.reshape(-1)]
            max_freq_first_cluster_label = np.argmax(np.bincount(cluster_first_cluster_label))
            
            ins_cluster_mask = first_ps_label == max_freq_first_cluster_label
            sem_labels = new_sem_labels[ins_cluster_mask]
            if np.unique(sem_labels).shape[0] != 1:
                raise "sem label error"
            sem_label = sem_labels[0]
            
            mod_bdl_ps_label[bdl_cluster_label == bdl_ps_label] = max_freq_first_cluster_label
            mod_bdl_sem_label[bdl_cluster_label == bdl_ps_label] = sem_label

        final_ins_points = np.concatenate([first_xyz, bdl_xyz], axis=0)
        mod_bdl_ps_label = mod_bdl_ps_label.astype(int)
        final_ins_labels = np.concatenate([first_ps_label.reshape(-1,1), mod_bdl_ps_label.reshape(-1,1)], axis=0)
        final_ins_labels = final_ins_labels.astype(int)
        final_sem_labels = np.concatenate([new_sem_labels.reshape(-1,1), mod_bdl_sem_label.reshape(-1,1)], axis=0)
        final_sem_labels = final_sem_labels.astype(int)



        tree = KDTree(final_ins_points, leaf_size=2)
        near_points = tree.query(org_feats[:,:3], k=1, return_distance=False)
        result_ins_labels = final_ins_labels.reshape(-1)[near_points.reshape(-1)].reshape(-1,1)
        result_sem_labels = final_sem_labels.reshape(-1)[near_points.reshape(-1)].reshape(-1,1)
        if DEBUG:
            gu.print_3d(
                gu.np_to_pcd_with_label(org_feats[:,:3], result_ins_labels), 
            )
            gu.print_3d(
                gu.np_to_pcd_with_label(org_feats[:,:3], result_sem_labels)
            )

        result_sem_labels[result_sem_labels>=9] += 2
        result_sem_labels[result_sem_labels>0] += 10
        assert result_sem_labels.shape[0] == n_vertices
        assert result_ins_labels.shape[0] == n_vertices
        
        return {
            "sem":result_sem_labels.reshape(-1),
            "ins":result_ins_labels.reshape(-1),
        }       

    def get_first_module_results(self, feats, base_model):
        """

        Args:
            batch_idx (_type_): _description_

        Returns:
            labels: N
        """
        points = feats
        with torch.no_grad():
            output = base_model([points])
        results = {}
        results["first_features"] = output["first_features"] 

        org_xyz_cpu = gu.torch_to_numpy(points)[0,:3,:].T

        whole_pd_sem_1 = gu.torch_to_numpy(output["sem_1"])[0,:,:].T
        whole_cls_1 = np.argmax(whole_pd_sem_1, axis=1)
        full_labeled_points_1 = np.concatenate([org_xyz_cpu, whole_cls_1.reshape(-1,1)], axis=1)

        results["sem_1"] = {}
        results["sem_1"]["full_labeled_points"] = full_labeled_points_1
        results["sem_1"]["whole_pd_sem"] = whole_pd_sem_1

        crop_num = output["sem_2"].shape[0]

        whole_pd_mask_2 = torch.zeros((points.shape[2], 2)).cuda()
        whole_pd_mask_count_2 = torch.zeros(points.shape[2]).cuda()
        for crop_idx in range(crop_num):
            pd_mask = output["sem_2"][crop_idx, :, :].permute(1,0) # 3072,17
            inside_crop_idx = output["nn_crop_indexes"][0][crop_idx]
            whole_pd_mask_2[inside_crop_idx] += pd_mask
            whole_pd_mask_count_2[inside_crop_idx] += 1
        
        whole_pd_mask_2 = gu.torch_to_numpy(whole_pd_mask_2)
        whole_mask_2 = np.argmax(whole_pd_mask_2, axis=1)
        full_masked_points_2 = np.concatenate([org_xyz_cpu, whole_mask_2.reshape(-1,1)], axis=1)

        results["sem_2"] = {}
        results["sem_2"]["full_masked_points"] = full_masked_points_2
        results["sem_2"]["whole_pd_mask"] = whole_pd_mask_2

        moved_points_cpu = org_xyz_cpu + gu.torch_to_numpy(output["offset_1"])[0,:3,:].T
        fg_moved_points = moved_points_cpu[results["sem_2"]["full_masked_points"][:,3]==1, :]

        
        fg_points_labels_ls = tu.get_clustering_labels(moved_points_cpu, results["sem_2"]["full_masked_points"][:,3])

        points_ins_labels = np.zeros(org_xyz_cpu.shape[0])
        points_ins_labels[:] = -1
        points_ins_labels[np.where(results["sem_2"]["full_masked_points"][:,3])] = fg_points_labels_ls
        points_ins_labels += 1

        full_ins_labeled_points = np.concatenate([org_xyz_cpu, points_ins_labels.reshape(-1,1)], axis=1)
        results["ins"] = {}
        results["ins"]["full_ins_labeled_points"] = full_ins_labeled_points
        return results

    def get_second_module_results(self, feats, sampled_boundary_seg_label, base_model):
        """

        Args:
            batch_idx (_type_): _description_

        Returns:
            labels: N
        """
        points = feats
        with torch.no_grad():
            output = base_model([points, sampled_boundary_seg_label], test=True)
        results = {}

        crop_num = output["sem_2"].shape[0]
        org_xyz_cpu = gu.torch_to_numpy(points)[0,:3,:].T

        whole_pd_mask_2 = torch.zeros((points.shape[2], 2)).cuda()
        whole_pd_mask_count_2 = torch.zeros(points.shape[2]).cuda()
        for crop_idx in range(crop_num):
            pd_mask = output["sem_2"][crop_idx, :, :].permute(1,0) # 3072,17
            inside_crop_idx = output["nn_crop_indexes"][0][crop_idx]
            whole_pd_mask_2[inside_crop_idx] += pd_mask
            whole_pd_mask_count_2[inside_crop_idx] += 1
    
        if False:
            t = gu.torch_to_numpy(output['first_features'])
            x_p = gu.torch_to_numpy(points.permute(2,1,0)[:,:3])

            pca = PCA(n_components=3, svd_solver='full')
            pca.fit(t)
            colors = np.matmul(t,pca.components_.T)
            colors = ((colors - colors.min(axis=0))/(colors.max(axis=0)-colors.min(axis=0)))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(x_p[:,:3])
            pcd.colors = o3d.utility.Vector3dVector(colors)


        whole_pd_mask_2 = gu.torch_to_numpy(whole_pd_mask_2)
        whole_mask_2 = np.argmax(whole_pd_mask_2, axis=1)
        full_masked_points_2 = np.concatenate([org_xyz_cpu, whole_mask_2.reshape(-1,1)], axis=1)

        results["sem_2"] = {}
        results["sem_2"]["full_masked_points"] = full_masked_points_2
        results["sem_2"]["whole_pd_mask"] = whole_pd_mask_2

        moved_points_cpu = org_xyz_cpu + gu.torch_to_numpy(output["offset_1"])[0,:3,:].T
        fg_moved_points = moved_points_cpu[results["sem_2"]["full_masked_points"][:,3]==1, :]

        num_of_clusters = []
        num_of_clusters.append(len(np.unique(gu.torch_to_numpy(sampled_boundary_seg_label)))-1)
        cluster_centroids, cluster_centroids_labels, fg_points_labels_ls = tu.clustering_points(
            [fg_moved_points], 
            method="kmeans", 
            num_of_clusters=num_of_clusters
        )
        
        points_ins_labels = np.zeros(org_xyz_cpu.shape[0])
        points_ins_labels -= 1
        points_ins_labels[np.where(results["sem_2"]["full_masked_points"][:,3])] = fg_points_labels_ls
        points_ins_labels += 1

        full_ins_labeled_points = np.concatenate([org_xyz_cpu, points_ins_labels.reshape(-1,1)], axis=1)
        results["ins"] = {}
        results["ins"]["full_ins_labeled_points"] = full_ins_labeled_points
        results["num_of_clusters"] = num_of_clusters[0]

        return results


    def get_boundary_sampled_feats(self,point_labels, org_feats, sampled_feats, sample_output_features):
        xyz_cpu = sampled_feats[:,:3].copy() # N',3
        tree = KDTree(xyz_cpu, leaf_size=40)

        bd_labels = np.zeros(org_feats.shape[0]) # N
        ps_labels = np.zeros(org_feats.shape[0]) # N
        near_points = tree.query(org_feats[:,:3], k=40, return_distance=False, )

        labels_arr = point_labels[near_points]
        label_counts = gu.count_unique_by_row(labels_arr)
        label_ratio = label_counts[:, 0] / 40.

        bd_labels[label_ratio < self.config["boundary_sampling_info"]["bdl_ratio"]] = 1

        k_1_near_points = tree.query(org_feats[:,:3], k=1, return_distance=False)
        ps_labels = point_labels[k_1_near_points[:,0]].reshape(-1,1)


        bd_org_feat_cpu = org_feats[bd_labels==1, :]
        bd_org_ps_label_cpu = ps_labels[bd_labels==1, :]
        bd_org_feat_cpu, bd_org_ps_label_cpu = gu.resample_pcd([bd_org_feat_cpu, bd_org_ps_label_cpu], self.config["boundary_sampling_info"]["num_of_bdl_points"], "uniformly")
        non_bd_org_feat_cpu = org_feats[bd_labels!=1, :]
        non_bd_org_ps_label_cpu = ps_labels[bd_labels!=1, :]
        non_bd_org_feat_cpu, non_bd_org_ps_label_cpu = gu.resample_pcd([non_bd_org_feat_cpu, non_bd_org_ps_label_cpu], self.config["boundary_sampling_info"]["num_of_all_points"]-bd_org_feat_cpu.shape[0], "fps")
        
        results_feat_cpu = np.concatenate([bd_org_feat_cpu, non_bd_org_feat_cpu], axis=0)
        results_label_cpu = np.concatenate([bd_org_ps_label_cpu, non_bd_org_ps_label_cpu], axis=0)

        if sample_output_features is not None:
            sample_output_features = gu.torch_to_numpy(sample_output_features)[0,:,:].T

            near_points = tree.query(results_feat_cpu[:,:3], k=3, return_distance=True)
            near_points_idxes = near_points[1]
            near_points_prop = near_points[0]
            near_points_prop = near_points_prop / np.sum(near_points_prop,axis=1).reshape(-1,1)

            nn_features = sample_output_features[near_points_idxes] * near_points_prop.reshape(24000,3,1)
            nn_features = nn_features.sum(axis=1)
            nn_features = nn_features.astype('float32')
            results_feat_cpu = np.concatenate([results_feat_cpu, nn_features], axis=1)

        return results_feat_cpu, results_label_cpu, bd_org_feat_cpu, bd_org_ps_label_cpu
