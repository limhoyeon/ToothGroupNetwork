import sys
import os

from models.tf_cbl_first_mask_model import TfCblFirstMaskModule

sys.path.append(os.getcwd())

import gen_utils as gu
from models.tf_cbl_second_model import TfCblSecondModule
import numpy as np
from models import tf_cbl_two_step_half_num_model
import torch
import tsg_utils as tu
from sklearn.neighbors import KDTree
from glob import glob
import os
from sklearn.decomposition import PCA
import open3d as o3d
from models.tf_cbl_first_model import TfCblFirstModule

class InferencePipelineFinal:
    def _get_mask_tf_cbl_module(self):
        config = {
            "model_parameter":{
                "input_feat": 6,
                "stride": [1, 4, 4, 4, 4],
                "nstride": [2, 2, 2, 2],
                "nsample": [36, 24, 24, 24, 24],
                "blocks": [2, 3, 4, 6, 3],
                "block_num": 5,
                "planes": [32, 64, 128, 256, 512],

                "contain_weight": False

                #"input_feat": 6,
                #"stride": [1, 4, 4],
                #"nstride": [2, 2],
                #"nsample": [8, 16, 16],
                #"blocks": [1, 1, 1],
                #"block_num": 3,
                #"planes": [8, 8, 8]
            },
        }
        module = TfCblFirstMaskModule(config).cuda()
        module.train()
        return module

    def _get_offset_tf_cbl_module(self):
        module_config = {
            "model_parameter":{
                "concat_first_module": False,
                "use_uniform_sampled_on_crop": False,
                "contain_weight": False,

                "input_feat": 6,
                "stride": [1, 4, 4, 4, 4],
                "nstride": [2, 2, 2, 2],
                "nsample": [36, 24, 24, 24, 24],
                "blocks": [2, 3, 4, 6, 3],
                "block_num": 5,
                "planes": [32, 64, 128, 256, 512],
                "crop_sample_size": 3072
            },
            "checkpoint_path":"testmodel/ckpts/0707_cosannealing_val.h5",
        }
        module = tf_cbl_two_step_half_num_model.TfCblTwoStepHalfNumOnlyOffsetModule(module_config)
        module.cuda()
        module.train()
        return module

    def _get_first_tf_cbl_module(self):
        config = {
            "model_parameter":{
                "input_feat": 6,
                "stride": [1, 4, 4, 4, 4],
                "nstride": [2, 2, 2, 2],
                "nsample": [36, 24, 24, 24, 24],
                "blocks": [2, 3, 4, 6, 3],
                "block_num": 5,
                "planes": [32, 64, 128, 256, 512],

                "contain_weight": False

                #"input_feat": 6,
                #"stride": [1, 4, 4],
                #"nstride": [2, 2],
                #"nsample": [8, 16, 16],
                #"blocks": [1, 1, 1],
                #"block_num": 3,
                #"planes": [8, 8, 8]
            },
        }
        module = TfCblFirstModule(config).cuda()
        module.train()
        return module

    def _get_second_tf_cbl_module(self):
        config = {
            "model_parameter":{
                "input_feat": 6,
                "stride": [1, 4, 4, 4, 4],
                "nstride": [2, 2, 2, 2],
                "nsample": [36, 24, 24, 24, 24],
                "blocks": [2, 3, 4, 6, 3],
                "block_num": 5,
                "planes": [32, 64, 128, 256, 512],

                "contain_weight": False

                #"input_feat": 6,
                #"stride": [1, 4, 4],
                #"nstride": [2, 2],
                #"nsample": [8, 16, 16],
                #"blocks": [1, 1, 1],
                #"block_num": 3,
                #"planes": [8, 8, 8]
            },
        }
        module = TfCblSecondModule(config).cuda()
        module.train()
        return module


    def __init__(self):
        self.scaler = 1.8
        self.shifter = 0.8
        self.offset_module_ckpt_paths = ["ckpts/0707_cosannealing_val.h5"]
        self.first_module_ckpt_paths = ["ckpts/0809_sched_v2_fixed_Flip(fixed)_weight0.1_val.h5", "ckpts/0809_reverse_sched_v2_fixed_Flip(fixed)_weight0.1_val.h5"]
        self.mask_module_ckpt_paths = ["ckpts/0805_mask_model_val.h5"]
        self.second_module_ckpt_paths = ["ckpts/0808_sched_fixed_tf_cbl_2.0_second_Flip_val.h5", "ckpts/0809_reverse_sched_fixed_tf_cbl_2.0_second_Flip_val.h5"]

        bdl_module_config = {
            "model_parameter":{
                "concat_first_module": False,
                "use_uniform_sampled_on_crop": False,
                "contain_weight": False,

                "input_feat": 6,
                "stride": [1, 4, 4, 4, 4],
                "nstride": [2, 2, 2, 2],
                "nsample": [36, 24, 24, 24, 24],
                "blocks": [2, 3, 4, 6, 3],
                "block_num": 5,
                "planes": [32, 64, 128, 256, 512],
                "crop_sample_size": 4608
            },
            #"checkpoint_path":"testmodel/ckpts/0704_boundary_cbl_feature_propagation_zerochamf_val.h5"
            "checkpoint_path":"ckpts/0805_bd_cbl_normal_rand_crop_4608_val.h5"
            #"checkpoint_path":"testmodel/ckpts/0711_bd_cbl_aug_test_val.h5"
        }

        self.mask_ens_module = self._get_mask_tf_cbl_module()
        self.offset_ens_module = self._get_offset_tf_cbl_module()
        self.cls_ens_module = self._get_first_tf_cbl_module()
        self.tf_cbl_ens_module = self._get_second_tf_cbl_module()

        self.bdl_module = tf_cbl_two_step_half_num_model.TfCblTwoStepHalfNumModule(bdl_module_config)
        self.bdl_module.cuda()
        self.bdl_module.load_state_dict(torch.load(bdl_module_config["checkpoint_path"]))
        self.bdl_module.train()

        pass

    def __call__(self, stl_path, ret_results=False):

        DEBUG=False
        _, mesh = gu.read_txt_obj_ls(stl_path, ret_mesh=True, use_tri_mesh=False)
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

        sampled_feats = gu.resample_pcd([vertices.copy()], 24000, "fps")[0]

        input_cuda_feats = torch.from_numpy(np.array([sampled_feats.astype('float32')])).cuda().permute(0,2,1)
        first_results = self.get_ens_module_results(input_cuda_feats)

        first_xyz = first_results["ins"]["full_ins_labeled_points"][:,:3]
        first_ps_label = first_results["ins"]["full_ins_labeled_points"][:,3].astype(int)
        first_sem_xyz = first_results["sem_1"]["full_labeled_points"][:,:3]
        first_sem_label = first_results["sem_1"]["full_labeled_points"][:,3]
        sampled_boundary_feats, sampled_boundary_seg_label, only_boundary_feats, only_boundary_seg_label = self.get_boundary_sampled_feats(
            first_results["ins"]["full_ins_labeled_points"][:,3], 
            bdl_feats, 
            sampled_feats,
            None
            #first_results["first_features"]
            )

        input_cuda_bdl_feats = torch.from_numpy(np.array([sampled_boundary_feats.astype('float32')])).permute(0,2,1).cuda()
        sampled_boundary_seg_label = torch.from_numpy(np.array([sampled_boundary_seg_label.astype(int)])).permute(0,2,1).cuda() - 1
        bdl_results = self.get_bdl_module_results(input_cuda_bdl_feats, sampled_boundary_seg_label, self.bdl_module)
        
        if DEBUG: gu.print_3d(gu.np_to_pcd_with_label(first_results["ins"]["full_ins_labeled_points"]), gu.np_to_pcd_with_label(bdl_results["ins"]["full_ins_labeled_points"]))

        #label propagation
        first_xyz = first_results["ins"]["full_ins_labeled_points"][:,:3]
        first_ps_label = first_results["ins"]["full_ins_labeled_points"][:,3].astype(int)
        first_sem_xyz = first_results["sem_1"]["full_labeled_points"][:,:3]
        first_sem_label = first_results["sem_1"]["full_labeled_points"][:,3]
        bdl_xyz = bdl_results["ins"]["full_ins_labeled_points"][:only_boundary_feats.shape[0],:3]
        bdl_ps_label = bdl_results["ins"]["full_ins_labeled_points"][:only_boundary_feats.shape[0],3].astype(int)


        #================================== sem 보정 부분 ======================================#
        #sem_label_mod
        gin_mean = np.mean(first_xyz[first_ps_label==0],axis=0).reshape(1,3)
        teeth_mean = np.mean(first_xyz[first_ps_label!=0],axis=0).reshape(1,3)
        

        first_ps_label_unique_except_zero = np.unique(first_ps_label); first_ps_label_unique_except_zero =first_ps_label_unique_except_zero[first_ps_label_unique_except_zero!=0]
        ins_label_center_points = np.array([np.mean(first_xyz[first_ps_label==label],axis=0) for label in first_ps_label_unique_except_zero])
        pca = PCA(n_components=3); pca.fit(ins_label_center_points); pca_axis = pca.components_
        pca_axis[2] = pca_axis[2] if np.dot((teeth_mean - gin_mean).reshape(3), pca_axis[2])>0 else -pca_axis[2]

        if np.where(first_sem_label==1)[0].shape[0] + np.where(first_sem_label==9)[0].shape[0] > 20:
            centerpoints_of_11_12 = np.array(np.mean([np.mean(first_sem_xyz[first_sem_label==1],axis = 0), np.mean(first_sem_xyz[first_sem_label==9], axis=0)], axis=0))
        else:
            #1번이 없는 상태
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



        if DEBUG:
            for ins_label in np.unique(first_ps_label):
                ins_cluster_mask = first_ps_label == ins_label
                sem_labels_in_cluster = new_sem_labels[ins_cluster_mask]
                print(np.unique(new_sem_labels[ins_cluster_mask]).shape[0])

        if DEBUG: gu.print_3d(gu.np_to_pcd_with_label(first_xyz, new_sem_labels))

        #================boundary 부분 ===========================#
        tree = KDTree(first_xyz, leaf_size=2)
        mod_bdl_ps_label = np.zeros((bdl_ps_label.shape[0]))
        mod_bdl_sem_label = np.zeros((bdl_ps_label.shape[0]))
        #0번이 둘 다 잇몸으로 세팅됨
        for bdl_cluster_label in np.unique(bdl_ps_label):
            if bdl_cluster_label==0:
                continue
            #bdl_ps_label이 모두 0이라면,,  
            #여기까지는 절대 오류 불가
            bdl_cluster_points = bdl_xyz[bdl_cluster_label == bdl_ps_label]
            cluster_near_points = tree.query(bdl_cluster_points, k=1, return_distance=False)
            #뭐 어떻게된상태든 무조건 N개는 나옴

            cluster_first_cluster_label = first_ps_label[cluster_near_points.reshape(-1)]
            #무조건 어떻게되든 ㅏㅅㅇ태든 무조건 쿼리는 될듯
            #cluster_first_cluster_label = cluster_first_cluster_label[cluster_first_cluster_label!=0]
            #if cluster_first_cluster_label.shape[0]==0:
            #    mod_bdl_ps_label[bdl_cluster_label == bdl_ps_label] = 0
            #    continue
            max_freq_first_cluster_label = np.argmax(np.bincount(cluster_first_cluster_label))
            
            ins_cluster_mask = first_ps_label == max_freq_first_cluster_label
            sem_labels = new_sem_labels[ins_cluster_mask]
            if np.unique(sem_labels).shape[0] != 1:
                raise "sem label error"
            sem_label = sem_labels[0]
            
            mod_bdl_ps_label[bdl_cluster_label == bdl_ps_label] = max_freq_first_cluster_label
            mod_bdl_sem_label[bdl_cluster_label == bdl_ps_label] = sem_label
        #=====전체 ins, sem propagation부분=================

        if True:
            #==================디버깅용=========================#
            bdl_full_xyz = bdl_results["ins"]["full_ins_labeled_points"][:,:3]
            bdl_full_ps_label = bdl_results["ins"]["full_ins_labeled_points"][:,3].astype(int)
            #================boundary 부분,full.시각화용===========================#
            tree = KDTree(first_xyz, leaf_size=2)
            mod_bdl_full_ps_label = np.zeros((bdl_full_ps_label.shape[0]))
            mod_bdl_full_sem_label = np.zeros((bdl_full_ps_label.shape[0]))
            #0번이 둘 다 잇몸으로 세팅됨
            for bdl_cluster_label in np.unique(bdl_full_ps_label):
                if bdl_cluster_label==0:
                    continue
                #bdl_ps_label이 모두 0이라면,,  
                #여기까지는 절대 오류 불가
                bdl_cluster_points = bdl_full_xyz[bdl_cluster_label == bdl_full_ps_label]
                cluster_near_points = tree.query(bdl_cluster_points, k=1, return_distance=False)
                #뭐 어떻게된상태든 무조건 N개는 나옴

                cluster_first_cluster_label = first_ps_label[cluster_near_points.reshape(-1)]
                #무조건 어떻게되든 ㅏㅅㅇ태든 무조건 쿼리는 될듯
                #cluster_first_cluster_label = cluster_first_cluster_label[cluster_first_cluster_label!=0]
                #if cluster_first_cluster_label.shape[0]==0:
                #    mod_bdl_ps_label[bdl_cluster_label == bdl_ps_label] = 0
                #    continue
                max_freq_first_cluster_label = np.argmax(np.bincount(cluster_first_cluster_label))
                
                ins_cluster_mask = first_ps_label == max_freq_first_cluster_label
                sem_labels = new_sem_labels[ins_cluster_mask]
                if np.unique(sem_labels).shape[0] != 1:
                    raise "sem label error"
                sem_label = sem_labels[0]
                
                mod_bdl_full_ps_label[bdl_cluster_label == bdl_full_ps_label] = max_freq_first_cluster_label
                mod_bdl_full_sem_label[bdl_cluster_label == bdl_full_ps_label] = sem_label
            #=====전체 ins, sem propagation부분=================



        final_ins_points = np.concatenate([first_xyz, bdl_xyz], axis=0)
        mod_bdl_ps_label = mod_bdl_ps_label.astype(int)
        final_ins_labels = np.concatenate([first_ps_label.reshape(-1,1), mod_bdl_ps_label.reshape(-1,1)], axis=0)
        final_ins_labels = final_ins_labels.astype(int)
        final_sem_labels = np.concatenate([new_sem_labels.reshape(-1,1), mod_bdl_sem_label.reshape(-1,1)], axis=0)
        final_sem_labels = final_sem_labels.astype(int)



        if DEBUG:
            for ins_label in np.unique(final_ins_labels):
                ins_cluster_mask = final_ins_labels == ins_label
                sem_labels_in_cluster = final_sem_labels[ins_cluster_mask]
                print(np.unique(sem_labels_in_cluster).shape[0])

        tree = KDTree(final_ins_points, leaf_size=2)
        near_points = tree.query(org_feats[:,:3], k=1, return_distance=False)
        result_ins_labels = final_ins_labels.reshape(-1)[near_points.reshape(-1)].reshape(-1,1)
        result_sem_labels = final_sem_labels.reshape(-1)[near_points.reshape(-1)].reshape(-1,1)
        
        if False: 
            gu.print_3d(gu.np_to_pcd_with_label(org_feats[:,:3], result_ins_labels))
            gu.print_3d(gu.np_to_pcd_with_label(org_feats[:,:3], result_sem_labels))
            
        if False:
            label_path = stl_path.replace("3D_scans_per_patient_obj_files", "ground-truth_labels_instances").replace("obj","json")
            loaded_json = gu.load_json(label_path)
            gt_labels = np.array(loaded_json['labels']).reshape(-1,1)
            if loaded_json['jaw'] == 'lower':
                gt_labels -= 20
            gt_labels[gt_labels//10==1] %= 10
            gt_labels[gt_labels//10==2] = (gt_labels[gt_labels//10==2]%10) + 8
            gt_labels[gt_labels<0] = 0
            a=1


        if False:
            
            tree = KDTree(first_xyz, leaf_size=2)
            near_points = tree.query(org_feats[:,:3], k=1, return_distance=False)
            fps_ins_labels = first_ps_label.reshape(-1)[near_points.reshape(-1)].reshape(-1,1)
            gu.print_3d(gu.np_to_pcd_with_label(org_feats[:,:3], fps_ins_labels))
            import copy
            plain_mesh = copy.deepcopy(mesh)
            plain_mesh.vertex_colors = gu.np_to_pcd_with_label(org_feats[:,:3], fps_ins_labels).colors
            gu.print_3d(plain_mesh)

            tree = KDTree(first_xyz, leaf_size=2)
            near_points = tree.query(org_feats[:,:3], k=1, return_distance=False)
            fps_sem_labels = first_sem_label.copy()
            fps_sem_labels[fps_sem_labels==9] = 1
            fps_sem_labels = fps_sem_labels.reshape(-1)[near_points.reshape(-1)].reshape(-1,1)
            gu.print_3d(gu.np_to_pcd_with_label(org_feats[:,:3], fps_sem_labels))
            import copy
            plain_mesh = copy.deepcopy(mesh)
            plain_mesh.vertex_colors = gu.np_to_pcd_with_label(org_feats[:,:3], fps_sem_labels).colors
            gu.print_3d(plain_mesh)
        if False:
            gt_xyz = org_feats[:,:3]
            gt_labels = gt_labels
            #================boundary 부분,full.시각화용===========================#
            tree = KDTree(first_xyz, leaf_size=2)
            mod_gt_labels = np.zeros((gt_labels.shape[0]))
            gt_labels = gt_labels.reshape(-1)
            #0번이 둘 다 잇몸으로 세팅됨
            for cluster_label in np.unique(gt_labels):
                if cluster_label==0:
                    continue
                #bdl_ps_label이 모두 0이라면,,  
                #여기까지는 절대 오류 불가
                bdl_cluster_points = gt_xyz[cluster_label == gt_labels]
                cluster_near_points = tree.query(bdl_cluster_points, k=1, return_distance=False)
                #뭐 어떻게된상태든 무조건 N개는 나옴

                cluster_first_cluster_label = first_ps_label[cluster_near_points.reshape(-1)]
                #무조건 어떻게되든 ㅏㅅㅇ태든 무조건 쿼리는 될듯
                #cluster_first_cluster_label = cluster_first_cluster_label[cluster_first_cluster_label!=0]
                #if cluster_first_cluster_label.shape[0]==0:
                #    mod_bdl_ps_label[bdl_cluster_label == bdl_ps_label] = 0
                #    continue
                max_freq_first_cluster_label = np.argmax(np.bincount(cluster_first_cluster_label))
                
                ins_cluster_mask = first_ps_label == max_freq_first_cluster_label
                mod_gt_labels[cluster_label == gt_labels] = max_freq_first_cluster_label
            plain_mesh = copy.deepcopy(mesh)
            plain_mesh.vertex_colors = gu.np_to_pcd_with_label(org_feats[:,:3], mod_gt_labels).colors

            #=====전체 ins, sem propagation부분=================

        if False:
            plain_mesh = copy.deepcopy(mesh)
            plain_mesh.vertex_colors = gu.np_to_pcd_with_label(org_feats[:,:3], result_ins_labels).colors

        result_sem_labels[result_sem_labels>=9] += 2
        result_sem_labels[result_sem_labels>0] += 10
        #if jaw_name == 'lower':
        #    result_pred_labels[result_pred_labels>0] += 20
        assert result_sem_labels.shape[0] == n_vertices
        assert result_ins_labels.shape[0] == n_vertices
        
        if ret_results:
            return {
                "sem":result_sem_labels.reshape(-1),
                "ins":result_ins_labels.reshape(-1),
                "sem_points": np.concatenate([org_feats[:,:3], result_sem_labels.reshape(-1,1)], axis=1),
                "ins_points": np.concatenate([org_feats[:,:3], result_ins_labels.reshape(-1,1)], axis=1)
            }
        else:
            return {
                "sem":result_sem_labels.reshape(-1),
                "ins":result_ins_labels.reshape(-1),
            }       
            
    def get_ens_module_results(self, feats):
        """

        Args:
            batch_idx (_type_): _description_

        Returns:
            labels: N
        """
        points = feats

        #module.load_state_dict(torch.load(config["load_state_path"]))
        cls_output_ls = []
        for i in range(len(self.first_module_ckpt_paths)):
            self.cls_ens_module.load_state_dict(torch.load(self.first_module_ckpt_paths[i]))
            with torch.no_grad():
                cls_output_ls.append(self.cls_ens_module([points], test=True))
        
        mask_output_ls = []

        for i in range(len(self.mask_module_ckpt_paths)):
            self.mask_ens_module.load_state_dict(torch.load(self.mask_module_ckpt_paths[i]))
            with torch.no_grad():
                mask_output_ls.append(self.mask_ens_module([points], test=True))

        offset_output_ls = []
        for i in range(len(self.offset_module_ckpt_paths)):
            self.offset_ens_module.load_state_dict(torch.load(self.offset_module_ckpt_paths[i]))
            with torch.no_grad():
                offset_output_ls.append(self.offset_ens_module([points], test=True))

        results = {}

        org_xyz_cpu = gu.torch_to_numpy(points)[0,:3,:].T

        output = {}

        output["cls_sem_1"] = None
        for i in range(len(self.first_module_ckpt_paths)):
            if output["cls_sem_1"] is None:
                output["cls_sem_1"] = gu.torch_to_numpy(cls_output_ls[i]["sem_1"])[0,:,:].T
            else:
                output["cls_sem_1"] = output["cls_sem_1"] + gu.torch_to_numpy(cls_output_ls[i]["sem_1"])[0,:,:].T
        
        output["offset_1"] = None
        for i in range(len(self.offset_module_ckpt_paths)):
            if output["offset_1"] is None:
                output["offset_1"] = gu.torch_to_numpy(offset_output_ls[i]["offset_1"])[0,:,:].T
            else:
                output["offset_1"] = output["offset_1"] + gu.torch_to_numpy(offset_output_ls[i]["offset_1"])[0,:,:].T
        output["offset_1"] /= len(self.offset_module_ckpt_paths)

        output["mask_1"] = None
        for i in range(len(self.mask_module_ckpt_paths)):
            if output["mask_1"] is None:
                output["mask_1"] = gu.torch_to_numpy(mask_output_ls[i]["sem_1"])[0,:,:].T
            else:
                output["mask_1"] = output["mask_1"] + gu.torch_to_numpy(mask_output_ls[i]["sem_1"])[0,:,:].T

        whole_mask_1 = output["mask_1"]
        whole_mask_1 = np.argmax(whole_mask_1, axis=1)

        whole_pre_cls_1 = output["cls_sem_1"][:,1:]
        whole_pre_cls_1 = np.argmax(whole_pre_cls_1, axis=1) + 1
        
        whole_cls_1 = whole_pre_cls_1.copy()
        whole_cls_1[whole_mask_1==0] = 0
        full_labeled_points_1 = np.concatenate([org_xyz_cpu, whole_cls_1.reshape(-1,1)], axis=1)

        results["sem_1"] = {}
        results["sem_1"]["full_labeled_points"] = full_labeled_points_1

        moved_points_cpu = org_xyz_cpu + output["offset_1"]
        
        if False:
            gu.print_3d(gu.np_to_pcd_with_label(org_xyz_cpu, whole_mask_1))
            gu.print_3d(gu.np_to_pcd_with_label(org_xyz_cpu, whole_pre_cls_1))
            gu.print_3d(gu.np_to_pcd_with_label(org_xyz_cpu, whole_cls_1))

            gu.print_3d(gu.np_to_pcd_with_label(full_labeled_points_1[:,:3], whole_pre_cls_1))
            for i in range(len(self.first_module_ckpt_paths)):
                whole_pd_sem = gu.torch_to_numpy(cls_output_ls[i]["sem_1"])
                whole_cls = np.argmax(whole_pd_sem[:,1:], axis=1)+1
                test_full_labeled_points = np.concatenate([org_xyz_cpu, whole_cls.reshape(-1,1)], axis=1)
                gu.print_3d(gu.np_to_pcd_with_label(test_full_labeled_points))

            gu.print_3d(gu.np_to_pcd_with_label(org_xyz_cpu, whole_mask_1))
            for i in range(len(self.first_module_ckpt_paths)):
                whole_pd_sem = gu.torch_to_numpy(mask_output_ls[i]["sem_1"])
                whole_cls = np.argmax(whole_pd_sem, axis=1)
                test_full_labeled_points = np.concatenate([org_xyz_cpu, whole_cls.reshape(-1,1)], axis=1)
                gu.print_3d(gu.np_to_pcd_with_label(test_full_labeled_points))

            gu.print_3d(moved_points_cpu[whole_cls_1!=0, :])
            for i in range(len(self.offset_module_ckpt_paths)):
                test_moved_points = org_xyz_cpu + gu.torch_to_numpy(offset_output_ls[i]["offset_1"])[0,:,:].T
                gu.print_3d(test_moved_points[whole_cls_1!=0, :])
        #num_of_clusters=[tu.get_cluster_number(moved_points_cpu, whole_cls_1.reshape(-1, 1), whole_pd_sem_1.reshape(-1,1))]
        #num_of_clusters = []
        #for b_idx in range(1):
        #    num_of_clusters.append(len(np.unique(gt_seg_label[b_idx,:]))-1)

        #cluster_centroids, cluster_centroids_labels, fg_points_labels_ls = tu.clustering_points(
        #    [fg_moved_points], 
        #    method="kmeans", 
        #    num_of_clusters=num_of_clusters
        #)
        
        fg_points_labels_ls = tu.get_clustering_labels(moved_points_cpu, results["sem_1"]["full_labeled_points"][:,3])

        points_ins_labels = np.zeros(org_xyz_cpu.shape[0])
        points_ins_labels[:] = -1
        points_ins_labels[np.where(results["sem_1"]["full_labeled_points"][:,3]!=0)] = fg_points_labels_ls
        points_ins_labels += 1

        if False:
            sem_1_ins_labeled_points = np.concatenate([org_xyz_cpu, points_ins_labels.reshape(-1,1)], axis=1)
            results["ins_sem_1"] = {}
            results["ins_sem_1"]["full_ins_labeled_points"] = sem_1_ins_labeled_points
            gu.print_3d(gu.np_to_pcd_with_label(sem_1_ins_labeled_points))
        second_fg_labels = self.get_second_module_results(feats, points_ins_labels)

        if True:
            final_fg_points_labels_ls = tu.get_clustering_labels(moved_points_cpu, second_fg_labels)
            final_points_ins_labels = np.zeros(org_xyz_cpu.shape[0])
            final_points_ins_labels[:] = -1
            final_points_ins_labels[np.where(second_fg_labels!=0)] = final_fg_points_labels_ls
            final_points_ins_labels += 1
        
        if False:
            #안잡히던 티쓰 포인트가있는데, 크롭에서 잡혀버렸어
            #하지만, 얘는 더이상 고려되지 못한다,,
            cond = (points_ins_labels==0) & (second_fg_labels==1)
            points_ins_labels[cond] = -1
            points_ins_labels[second_fg_labels==0] = 0
            final_points_ins_labels = tu.propagate_unlabeled_points(moved_points_cpu, points_ins_labels)
        
        final_ins_labels_points = np.concatenate([org_xyz_cpu, final_points_ins_labels.reshape(-1,1)], axis=1)
        results["ins"] = {}
        results["ins"]["full_ins_labeled_points"] = final_ins_labels_points

        if False:
            full_ins_labeled_points = np.concatenate([org_xyz_cpu, final_points_ins_labels.reshape(-1,1)], axis=1)
            results["ins"] = {}
            results["ins"]["full_ins_labeled_points"] = full_ins_labeled_points
            gu.print_3d(gu.np_to_pcd_with_label(results["ins"]["full_ins_labeled_points"]))
            gu.print_3d(gu.np_to_pcd_with_label(results["sem_1"]["full_labeled_points"]))

            first_ins_labeled_points = np.concatenate([org_xyz_cpu, points_ins_labels.reshape(-1,1)], axis=1)
            gu.print_3d(gu.np_to_pcd_with_label(first_ins_labeled_points))
        return results

    def get_second_module_results(self, feats, ins_labels):
        """

        Args:
            batch_idx (_type_): _description_

        Returns:
            labels: N
        """
        points = feats

        #module.load_state_dict(torch.load(config["load_state_path"]))
        output_ls = []
        ins_labels = torch.from_numpy(np.array([[ins_labels]])) - 1
        for i in range(len(self.second_module_ckpt_paths)):
            self.tf_cbl_ens_module.load_state_dict(torch.load(self.second_module_ckpt_paths[i]), strict=False)
            with torch.no_grad():
                output_ls.append(self.tf_cbl_ens_module([points, ins_labels], test=True))

        results = {}
        org_xyz_cpu = gu.torch_to_numpy(points)[0,:3,:].T

        output = {}
        output["sem_2"] = None
        for i in range(len(self.second_module_ckpt_paths)):
            if output["sem_2"] is None:
                output["sem_2"] = output_ls[i]["sem_2"]
            else:
                output["sem_2"] = output["sem_2"] + output_ls[i]["sem_2"]
        output["nn_crop_indexes"] = output_ls[0]["nn_crop_indexes"]

        org_xyz_cpu = gu.torch_to_numpy(points)[0,:3,:].T
        crop_num = output["sem_2"].shape[0]

        whole_pd_sem_2 = torch.zeros((points.shape[2], output["sem_2"].shape[1])).cuda()
        whole_pd_count_2 = torch.zeros(points.shape[2]).cuda()
        #whole_pd => 1,1, 포인트 수
        for crop_idx in range(crop_num):
            pd_mask = output["sem_2"][crop_idx, :, :].permute(1,0) # 3072,17
            #크롭내부에서의 확률
            inside_crop_idx = output["nn_crop_indexes"][0][crop_idx]
            # 안에 있는 포인트의 index들
            whole_pd_sem_2[inside_crop_idx] += pd_mask
            whole_pd_count_2[inside_crop_idx] += 1
        if False:
            for i in range(crop_num):
                xyz_cpu = gu.torch_to_numpy(output_ls[0]["cropped_feature_ls"][i,:3,:].T)
                cls_pred = np.argmax(gu.torch_to_numpy(output["sem_2"][i,:]).T, axis=1)
                gu.print_3d(gu.np_to_pcd_with_label(xyz_cpu, cls_pred))


        whole_pd_sem_2 = gu.torch_to_numpy(whole_pd_sem_2)
        whole_pd_count_2 = gu.torch_to_numpy(whole_pd_count_2)
        whole_pd_sem_2[whole_pd_count_2!=0, : ] /= whole_pd_count_2[whole_pd_count_2!=0].reshape(-1,1)
        whole_cls_2 = np.argmax(whole_pd_sem_2, axis=1)
        full_labeled_points_2 = np.concatenate([org_xyz_cpu, whole_cls_2.reshape(-1,1)], axis=1)
        if False:
            gu.print_3d(gu.np_to_pcd_with_label(full_labeled_points_2))

            for i in range(len(self.second_module_ckpt_paths)):
                whole_pd_sem_2 = torch.zeros((points.shape[2], output["sem_2"].shape[1])).cuda()
                whole_pd_count_2 = torch.zeros(points.shape[2]).cuda()
                #whole_pd => 1,1, 포인트 수
                for crop_idx in range(crop_num):
                    pd_mask = output_ls[i]["sem_2"][crop_idx, :, :].permute(1,0) # 3072,17
                    #크롭내부에서의 확률
                    inside_crop_idx = output_ls[i]["nn_crop_indexes"][0][crop_idx]
                    # 안에 있는 포인트의 index들
                    whole_pd_sem_2[inside_crop_idx] += pd_mask
                    whole_pd_count_2[inside_crop_idx] += 1
                
                whole_pd_sem_2 = gu.torch_to_numpy(whole_pd_sem_2)
                whole_pd_count_2 = gu.torch_to_numpy(whole_pd_count_2)
                whole_pd_sem_2[whole_pd_count_2!=0, : ] /= whole_pd_count_2[whole_pd_count_2!=0].reshape(-1,1)
                whole_cls_2 = np.argmax(whole_pd_sem_2, axis=1)
                full_labeled_points = np.concatenate([org_xyz_cpu, whole_cls_2.reshape(-1,1)], axis=1)
                gu.print_3d(gu.np_to_pcd_with_label(full_labeled_points))
            

        return whole_cls_2


    def get_bdl_module_results(self, feats, sampled_boundary_seg_label, base_model):
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
        #whole_pd => 1,1, 포인트 수
        for crop_idx in range(crop_num):
            pd_mask = output["sem_2"][crop_idx, :, :].permute(1,0) # 3072,17
            #크롭내부에서의 확률
            inside_crop_idx = output["nn_crop_indexes"][0][crop_idx]
            # 안에 있는 포인트의 index들
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

        #TODO 자동화
        #num_of_clusters = []
        #for b_idx in range(1):
        #    num_of_clusters.append(len(np.unique(gt_seg_label[b_idx,:]))-1)

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
        #gu.print_3d(gu.np_to_pcd_with_label(results["ins"]["full_ins_labeled_points"]))

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

        bd_labels[label_ratio < 0.7] = 1

        #labels_arr = labels_arr.astype(int)
        #for i in range(ps_labels.shape[0]):
            #ps_labels[i] = np.argmax(np.bincount(labels_arr[i]))
        #ps_labels = ps_labels.reshape(-1,1)

        k_1_near_points = tree.query(org_feats[:,:3], k=1, return_distance=False)
        ps_labels = point_labels[k_1_near_points[:,0]].reshape(-1,1)


        bd_org_feat_cpu = org_feats[bd_labels==1, :]
        bd_org_ps_label_cpu = ps_labels[bd_labels==1, :]
        bd_org_feat_cpu, bd_org_ps_label_cpu = gu.resample_pcd([bd_org_feat_cpu, bd_org_ps_label_cpu], 20000, "uniformly")
        non_bd_org_feat_cpu = org_feats[bd_labels!=1, :]
        non_bd_org_ps_label_cpu = ps_labels[bd_labels!=1, :]
        non_bd_org_feat_cpu, non_bd_org_ps_label_cpu = gu.resample_pcd([non_bd_org_feat_cpu, non_bd_org_ps_label_cpu], 24000-bd_org_feat_cpu.shape[0], "fps")
        
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
