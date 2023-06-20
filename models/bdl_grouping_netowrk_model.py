import torch
import numpy as np
from . import tgn_loss
import ops_utils as ou
import gen_utils as gu
from models.base_model import BaseModel
from sklearn.neighbors import KDTree
from loss_meter import LossMap
from .modules.grouping_network_module import GroupingNetworkModule
import os
from glob import glob

class BdlGroupingNetworkModel(BaseModel):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.base_model = GroupingNetworkModule(config["fps_model_info"])
        self.base_model.load_state_dict(torch.load(self.config["fps_model_info"]["load_ckpt_path"]+".h5"))
        self.base_model.cuda()

        self.stl_path_map = {}
        for dir_path in [
            x[0] for x in os.walk(config["boundary_sampling_info"]["orginal_data_obj_path"])
            ][1:]:
            for stl_path in glob(os.path.join(dir_path,"*.obj")):
                self.stl_path_map[os.path.basename(stl_path).split(".")[0]] = stl_path

        self.json_path_map = {}
        for dir_path in [
            x[0] for x in os.walk(config["boundary_sampling_info"]["orginal_data_json_path"])
            ][1:]:
            for json_path in glob(os.path.join(dir_path,"*.json")):
                self.json_path_map[os.path.basename(json_path).split(".")[0]] = json_path

        self.Y_AXIS_MAX = 33.15232091532151
        self.Y_AXIS_MIN = -36.9843781139949
        
    def get_loss(self, offset_1, offset_2, sem_1, sem_2, mask_1, mask_2, gt_seg_label_1, gt_seg_label_2, input_coords, cropped_coords):
        half_seg_label = gt_seg_label_1.clone()
        half_seg_label[half_seg_label>=9] -= 8

        gt_seg_label_2[gt_seg_label_2>=0] = 0
        tooth_class_loss_1 = tgn_loss.tooth_class_loss(sem_1, half_seg_label,9)
        tooth_class_loss_2 = tgn_loss.tooth_class_loss(sem_2, gt_seg_label_2,2)

        offset_1_loss, offset_1_dir_loss = tgn_loss.batch_center_offset_loss(offset_1, input_coords, gt_seg_label_1)
        
        chamf_1_loss = tgn_loss.batch_chamfer_distance_loss(offset_1, input_coords, gt_seg_label_1)
        return {
            "tooth_class_loss_1": (tooth_class_loss_1, self.config["tr_set"]["loss"]["tooth_class_loss_1"]),
            "tooth_class_loss_2": (tooth_class_loss_2, self.config["tr_set"]["loss"]["tooth_class_loss_2"]),
            "offset_1_loss": (offset_1_loss, self.config["tr_set"]["loss"]["offset_1_loss"]),
            "offset_1_dir_loss": (offset_1_dir_loss, self.config["tr_set"]["loss"]["offset_1_dir_loss"]),
            "chamf_1_loss" : (chamf_1_loss, self.config["tr_set"]["loss"]["chamf_1_loss"])
        }

    def get_points_cluster_labels(self, batch_item):
        """

        Args:
            batch_idx (_type_): _description_
            batch_item (batch_item): _description_

        Returns:
            labels: N
        """
        points = batch_item["feat"].cuda()
        seg_label = batch_item["gt_seg_label"].cuda()
        gt_seg_label = gu.torch_to_numpy(batch_item["gt_seg_label"])
        with torch.no_grad():
            output = self.base_model([points, seg_label])
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
        
        whole_pd_mask_2 = gu.torch_to_numpy(whole_pd_mask_2)
        whole_mask_2 = np.argmax(whole_pd_mask_2, axis=1)
        full_masked_points_2 = np.concatenate([org_xyz_cpu, whole_mask_2.reshape(-1,1)], axis=1)

        results["sem_2"] = {}
        results["sem_2"]["full_masked_points"] = full_masked_points_2
        results["sem_2"]["whole_pd_mask"] = whole_pd_mask_2

        moved_points_cpu = org_xyz_cpu + gu.torch_to_numpy(output["offset_1"])[0,:3,:].T
        fg_moved_points = moved_points_cpu[results["sem_2"]["full_masked_points"][:,3]==1, :]

        num_of_clusters = []
        for b_idx in range(1):
            num_of_clusters.append(len(np.unique(gt_seg_label[b_idx,:]))-1)

        cluster_centroids, cluster_centroids_labels, fg_points_labels_ls = ou.clustering_points(
            [fg_moved_points], 
            method="kmeans", 
            num_of_clusters=num_of_clusters
        )
        
        points_ins_labels = np.zeros(org_xyz_cpu.shape[0])
        points_ins_labels[:] = -1
        points_ins_labels[np.where(results["sem_2"]["full_masked_points"][:,3])] = fg_points_labels_ls[b_idx]
        
        full_ins_labeled_points = np.concatenate([org_xyz_cpu, points_ins_labels.reshape(-1,1)], axis=1)
        results["ins"] = {}
        results["ins"]["full_ins_labeled_points"] = full_ins_labeled_points

        results["first_features"] = output["first_features"]
        return results

    def load_mesh(self, base_name):
        loaded_json = gu.load_json(self.json_path_map[base_name])
        labels = np.array(loaded_json['labels']).reshape(-1,1)
        if loaded_json['jaw'] == 'lower':
            labels -= 20
        labels[labels//10==1] %= 10
        labels[labels//10==2] = (labels[labels//10==2]%10) + 8
        labels[labels<0] = 0

        vertices = gu.read_txt_obj_ls(self.stl_path_map[base_name], ret_mesh=False)[0]
        vertices[:, :3] -= np.mean(vertices[:,:3], axis=0)
        vertices[:, :3] = ((vertices[:, :3]-self.Y_AXIS_MIN)/(self.Y_AXIS_MAX-self.Y_AXIS_MIN))*2-1
        vertices = vertices.astype("float32")
        labels -= 1
        labels = labels.astype(int)
        return vertices, labels.reshape(-1,1)

    def get_boundary_sampled_points(self, batch_item):
        base_name = os.path.basename(batch_item["mesh_path"][0]).split("_")[0] + "_" + os.path.basename(batch_item["mesh_path"][0]).split("_")[1]

        cache_path = os.path.join(self.config["boundary_sampling_info"]["bdl_cache_path"], base_name+".npy")

        if not os.path.exists(cache_path):
            org_feat_cpu, org_gt_seg_label = self.load_mesh(base_name)
            if(org_feat_cpu.shape[0] < self.config["boundary_sampling_info"]["num_of_all_points"]):
                return batch_item["feat"], batch_item["gt_seg_label"]
            results = self.get_points_cluster_labels(batch_item) # N
            points_labels = results["ins"]["full_ins_labeled_points"][:,3]
            xyz_cpu = gu.torch_to_numpy(batch_item["feat"])[0,:3,:].T # N, 3

            tree = KDTree(xyz_cpu, leaf_size=2)


            if batch_item["aug_obj"][0]:
                auged_org_feat_cpu = batch_item["aug_obj"][0].run(org_feat_cpu.copy())
            else:
                auged_org_feat_cpu = org_feat_cpu.copy()
            bd_labels = np.zeros(auged_org_feat_cpu.shape[0]) # N
            near_points = tree.query(auged_org_feat_cpu[:,:3], k=40, return_distance=False)

            labels_arr = points_labels[near_points]
            label_counts = gu.count_unique_by_row(labels_arr)
            label_ratio = label_counts[:, 0] / 40.
            
            #To change
            bd_labels[label_ratio < self.config["boundary_sampling_info"]["bdl_ratio"]] = 1

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            bd_org_feat_cpu = org_feat_cpu[bd_labels==1, :]
            bd_auged_org_feat_cpu = auged_org_feat_cpu[bd_labels==1, :]
            bd_org_gt_seg_label = org_gt_seg_label[bd_labels==1, :]

            bd_auged_org_feat_cpu, bd_org_gt_seg_label, bd_org_feat_cpu = gu.resample_pcd([bd_auged_org_feat_cpu, bd_org_gt_seg_label, bd_org_feat_cpu], self.config["boundary_sampling_info"]["num_of_bdl_points"], "uniformly")

            non_bd_org_feat_cpu = org_feat_cpu[bd_labels==0, :]
            non_bd_auged_org_feat_cpu = auged_org_feat_cpu[bd_labels==0, :]
            non_bd_org_gt_seg_label = org_gt_seg_label[bd_labels==0, :]
            non_bd_auged_org_feat_cpu, non_bd_org_gt_seg_label, non_bd_org_feat_cpu = gu.resample_pcd([
                non_bd_auged_org_feat_cpu, 
                non_bd_org_gt_seg_label, 
                non_bd_org_feat_cpu], self.config["boundary_sampling_info"]["num_of_all_points"]-bd_auged_org_feat_cpu.shape[0], "fps")

            sampled_auged_org_feat_cpu = np.concatenate([bd_auged_org_feat_cpu, non_bd_auged_org_feat_cpu], axis=0)
            sampled_org_feat_cpu = np.concatenate([bd_org_feat_cpu, non_bd_org_feat_cpu], axis=0)
            sampled_org_gt_seg_label = np.concatenate([bd_org_gt_seg_label ,non_bd_org_gt_seg_label], axis=0)
            np.save(cache_path, np.concatenate([sampled_org_feat_cpu, sampled_org_gt_seg_label], axis=1))            
        else:
            cached_arr = np.load(cache_path)
            sampled_auged_org_feat_cpu, sampled_org_gt_seg_label = cached_arr[:,:6], cached_arr[:,6:]
            if batch_item["aug_obj"][0]:
                sampled_auged_org_feat_cpu = batch_item["aug_obj"][0].run(sampled_auged_org_feat_cpu.copy())
            else:
                sampled_auged_org_feat_cpu = sampled_auged_org_feat_cpu.copy()
            sampled_auged_org_feat_cpu = sampled_auged_org_feat_cpu.astype('float32')
            sampled_org_gt_seg_label = sampled_org_gt_seg_label.astype(int)

        return torch.from_numpy(sampled_auged_org_feat_cpu.T.reshape(1,sampled_auged_org_feat_cpu.shape[1],-1)), torch.from_numpy(sampled_org_gt_seg_label.T.reshape(1,1,-1))


    def step(self, batch_idx, batch_item, phase):
        self._set_model(phase)

        points, seg_label = self.get_boundary_sampled_points(batch_item)

        points = points.cuda()
        l0_xyz = points[:,:3,:].cuda()
        
        seg_label = seg_label.cuda()
        
        if phase == "train":
            output = self.module([points, seg_label])
        else:
            with torch.no_grad():
                output = self.module([points, seg_label])
        loss_meter = LossMap()
        
        loss_meter.add_loss_by_dict(self.get_loss(
            output["offset_1"], 
            output["offset_2"], 
            output["sem_1"], 
            output["sem_2"], 
            output["mask_1"], 
            output["mask_2"], 
            seg_label, 
            output["cluster_gt_seg_label"], 
            l0_xyz, 
            output["cropped_feature_ls"][:,:3,:]
            )
        )
        
        if phase == "train":
            loss_meter.add_loss("cbl_loss_1", output["cbl_loss_1"].sum(), self.config["tr_set"]["loss"]["cbl_loss_1"])
            loss_meter.add_loss("cbl_loss_2", output["cbl_loss_2"].sum(), self.config["tr_set"]["loss"]["cbl_loss_2"])
            loss_sum = loss_meter.get_sum()
            self.optimizer.zero_grad()
            loss_sum.backward()
            self.optimizer.step()

        return loss_meter

    def infer(self, batch_idx, batch_item, **options):
        pass
