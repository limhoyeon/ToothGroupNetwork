from genericpath import exists
from matplotlib import offsetbox
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import os
import sys

sys.path.append(os.getcwd())
sys.path.append("..")
import numpy as np
from . import train_loss as tsg_ins_cent_loss
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import tsg_utils as tu
import gen_utils as gu
from .modules.cbl_point_transformer.cbl_point_transformer_module import get_model
from glob import glob
from models.base_model import BaseModel
from sklearn.neighbors import KDTree
from loss_meter import LossMap
class TfCblTwoStepHalfNumOnlyOffsetModule(torch.nn.Module):
    def __init__(self, config):
        self.config = config

        super().__init__()
        class_num = 9
        self.first_ins_cent_model = get_model(**config["model_parameter"], c=config["model_parameter"]["input_feat"], k=class_num + 1)
        if self.config["model_parameter"]["concat_first_module"]:
            self.second_ins_cent_model = get_model(**config["model_parameter"], c=38, k=2).train().cuda()
        else:
            self.second_ins_cent_model = get_model(**config["model_parameter"], c=config["model_parameter"]["input_feat"], k=2).train().cuda()

    def forward(self, inputs, test=False):
        DEBUG=False
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1] => B, 1, 24000 : ground truth segmentation
        """
        B, C, N = inputs[0].shape
        outputs = {}
        if len(inputs)>=2 and not test:
            half_seg_label = inputs[1].clone()
            half_seg_label[half_seg_label>=9] -= 8
            cbl_loss_1, sem_1, offset_1, mask_1, first_features = self.first_ins_cent_model([inputs[0], half_seg_label])
            outputs.update({
                "cbl_loss_1": cbl_loss_1,
                "sem_1": sem_1,
                "offset_1":offset_1,
                "mask_1":mask_1,
                "first_features": first_features
            })
        else:
            sem_1, offset_1, mask_1, first_features = self.first_ins_cent_model([inputs[0]])
            outputs.update({
                "sem_1": sem_1,
                "offset_1":offset_1,
                "mask_1":mask_1,
                "first_features": first_features
            })
        return outputs


class TfCblTwoStepHalfNumModule(torch.nn.Module):
    def __init__(self, config):
        self.config = config

        super().__init__()
        class_num = 9
        self.first_ins_cent_model = get_model(**config["model_parameter"], c=config["model_parameter"]["input_feat"], k=class_num + 1)
        if self.config["model_parameter"]["concat_first_module"]:
            self.second_ins_cent_model = get_model(**config["model_parameter"], c=38, k=2).train().cuda()
        else:
            self.second_ins_cent_model = get_model(**config["model_parameter"], c=config["model_parameter"]["input_feat"], k=2).train().cuda()

    def forward(self, inputs, test=False):
        DEBUG=False
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1] => B, 1, 24000 : ground truth segmentation
        """
        B, C, N = inputs[0].shape
        outputs = {}
        if len(inputs)>=2 and not test:
            half_seg_label = inputs[1].clone()
            half_seg_label[half_seg_label>=9] -= 8
            cbl_loss_1, sem_1, offset_1, mask_1, first_features = self.first_ins_cent_model([inputs[0], half_seg_label])
            outputs.update({
                "cbl_loss_1": cbl_loss_1,
                "sem_1": sem_1,
                "offset_1":offset_1,
                "mask_1":mask_1,
                "first_features": first_features
            })
        else:
            sem_1, offset_1, mask_1, first_features = self.first_ins_cent_model([inputs[0]])
            outputs.update({
                "sem_1": sem_1,
                "offset_1":offset_1,
                "mask_1":mask_1,
                "first_features": first_features
            })
        
        cluster_centroids = []
        if len(inputs) >= 2:
            for b_idx in range(B):
                b_gt_seg_labels = gu.torch_to_numpy(inputs[1][b_idx,:,:].view(-1))
                b_points_coords = gu.torch_to_numpy(inputs[0][b_idx,:3,:]).T
                contained_tooth_num = np.unique(b_gt_seg_labels)
                temp_list = []
                for tooth_num in contained_tooth_num:
                    if tooth_num == -1:
                        continue
                    temp_list.append(b_points_coords[tooth_num == b_gt_seg_labels].mean(axis=0))
                cluster_centroids.append(temp_list)
        else:
            for b_idx in range(B):
                whole_pd_sem_1 = gu.torch_to_numpy(sem_1)[b_idx,:,:].T
                whole_cls_1 = np.argmax(whole_pd_sem_1, axis=1)
                whole_offset_1 = gu.torch_to_numpy(offset_1)[b_idx,:,:].T
                b_points_coords = gu.torch_to_numpy(inputs[b_idx][b_idx,:3,:]).T
                b_moved_points = b_points_coords + whole_offset_1
                b_fg_moved_points = b_moved_points[whole_cls_1.reshape(-1)!=0, :]
                fg_points_labels_ls = tu.get_clustering_labels(b_moved_points, whole_cls_1)
                temp_centroids = []
                for i in np.unique(fg_points_labels_ls):
                    temp_centroids.append(np.mean(b_fg_moved_points[fg_points_labels_ls==i, :],axis=0))

                #b_cluster_num = tu.get_cluster_number(b_points_coords, whole_cls_1, None)
                #
                #temp_centroids, cluster_centroids_labels, fg_points_labels_ls = tu.clustering_points(
                #    [b_points_coords[whole_cls_1!=0, :]], 
                #    method="kmeans", 
                #    num_of_clusters=[b_cluster_num]
                #)


                cluster_centroids.append(temp_centroids)
            
        if self.config["model_parameter"]["use_uniform_sampled_on_crop"]:
            org_xyz_cpu = gu.torch_to_numpy(inputs[2][:, :3, :].permute(0, 2, 1))
            nn_crop_indexes = tu.get_nearest_neighbor_idx(org_xyz_cpu, cluster_centroids, 8192)
            permute_idxes = np.random.permutation(8192)[:self.config["model_parameter"]["crop_sample_size"]]
            nn_crop_indexes[0] = nn_crop_indexes[0][:,permute_idxes]
            cropped_feature_ls = tu.get_indexed_features(inputs[2], nn_crop_indexes)
            cluster_gt_seg_label = tu.get_indexed_features(inputs[3], nn_crop_indexes)
        else:
            org_xyz_cpu = gu.torch_to_numpy(inputs[0][:, :3, :].permute(0, 2, 1))
            nn_crop_indexes = tu.get_nearest_neighbor_idx(org_xyz_cpu, cluster_centroids, self.config["model_parameter"]["crop_sample_size"])
            cropped_feature_ls = tu.get_indexed_features(inputs[0], nn_crop_indexes)
            #cropped_output_feature_ls = tu.get_indexed_features(first_features, nn_crop_indexes)
            if len(inputs)>=2:
                cluster_gt_seg_label = tu.get_indexed_features(inputs[1], nn_crop_indexes)

        cropped_feature_ls = tu.centering_object(cropped_feature_ls)
        if DEBUG:
            for i in range(cropped_feature_ls.shape[0]):
                xyzs = gu.torch_to_numpy(cropped_feature_ls[i,:3,:]).T
                labels = gu.torch_to_numpy(cluster_gt_seg_label[i,:1,:]).T
                gu.print_3d(gu.np_to_pcd_with_label(np.concatenate([xyzs, labels], axis=1)), gu.make_coord_frame())

        if self.config["model_parameter"]["concat_first_module"]:
            pass
            #crop_input_features = torch.cat([cropped_feature_ls, cropped_output_feature_ls], dim=1)
        else:
            crop_input_features = cropped_feature_ls

        if len(inputs) >= 2 and not test:
            cluster_gt_seg_label[cluster_gt_seg_label>=0] = 0
            outputs["cluster_gt_seg_label"] = cluster_gt_seg_label
            cbl_loss_2, sem_2, offset_2, mask_2, _ = self.second_ins_cent_model([crop_input_features, cluster_gt_seg_label])

            outputs.update({
                "cbl_loss_2": cbl_loss_2,
                "sem_2": sem_2,
                "offset_2":offset_2,
                "mask_2":mask_2,
            })

        else:
            sem_2, offset_2, mask_2, _ = self.second_ins_cent_model([crop_input_features])
            outputs.update({
                "sem_2": sem_2,
                "offset_2":offset_2,
                "mask_2":mask_2,
            })

        outputs["cropped_feature_ls"] = cropped_feature_ls
        outputs["nn_crop_indexes"] =  nn_crop_indexes

        return outputs

class TfCblTwoStepHalfNumModel(BaseModel):
    def tilted_weight_maker(self, gt_seg_label):
        """편향된 weight 만들기

        Args:
            gt_seg_label (B, 1, 24000): temp로 일단 B는 1만
        """
        unique_labels = gt_seg_label.unique()
        for i in [0,1,2,3,4,5,   8,9,10,11,12,13]:
            if i not in unique_labels:
                return 1
        return 0.5
        
    def get_loss(self, offset_1, offset_2, sem_1, sem_2, mask_1, mask_2, gt_seg_label_1, gt_seg_label_2, input_coords, cropped_coords):
        half_seg_label = gt_seg_label_1.clone()
        half_seg_label[half_seg_label>=9] -= 8

        gt_seg_label_2[gt_seg_label_2>=0] = 0
        tooth_class_loss_1 = tsg_ins_cent_loss.tooth_class_loss(sem_1, half_seg_label)
        tooth_class_loss_2 = tsg_ins_cent_loss.tooth_class_loss(sem_2, gt_seg_label_2)

        offset_1_loss, offset_1_dir_loss = tsg_ins_cent_loss.batch_center_offset_loss(offset_1, input_coords, gt_seg_label_1)
        
        chamf_1_loss = tsg_ins_cent_loss.batch_chamfer_distance_loss(offset_1, input_coords, gt_seg_label_1)
        return {
            "tooth_class_loss_1": (tooth_class_loss_1, 1),
            "tooth_class_loss_2": (tooth_class_loss_2, 1),
            "offset_1_loss": (offset_1_loss, 0.03),
            "offset_1_dir_loss": (offset_1_dir_loss, 0.03),
            "chamf_1_loss" : (chamf_1_loss, 0.15)
        }

    def step(self, batch_idx, batch_item, phase):
        self._set_model(phase)

        points = batch_item["feat"].cuda()
        l0_xyz = batch_item["feat"][:,:3,:].cuda()
        
        #centroids = batch_item[1].cuda()
        seg_label = batch_item["gt_seg_label"].cuda()
        
        inputs = [points, seg_label]

        if self.config["model_parameter"]["use_uniform_sampled_on_crop"]:
            uniform_points = batch_item[4].cuda()
            uniform_seg_label = batch_item[5].cuda()
            inputs.append(uniform_points)
            inputs.append(uniform_seg_label)
        if phase == "train":
            output = self.model(inputs)
        else:
            with torch.no_grad():
                output = self.model(inputs)
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
            loss_meter.add_loss("cbl_loss_1", output["cbl_loss_1"].sum(), 1)
            loss_meter.add_loss("cbl_loss_2", output["cbl_loss_2"].sum(), 1)
            loss_sum = loss_meter.get_sum()
            self.optimizer.zero_grad()
            loss_sum.backward()
            self.optimizer.step()

        return loss_meter
    
        #필요한 함수
        #loss를 등록
        #loss의 weight도 등록
        #loss를 반환(dictionary 형태로, total도 함께 넣어서, item 형태로 모두)
        #loss를 summation하여 반환 - backprop용도

    def save_prob_off_sem_info(self, batch_idx, batch_item, **options):
        self._set_model("val")
        anorm_list = ["019ZGSJR_lower", "019TTV1D_lower", "019TTv1D_UPPER", "017U3R3T_UPPER", "019ZKUHV_UPPER", 
                      "80RPZWJT_LOWER", "80RPZWJT_UPPER", "949XHLS5_UPPER", "ANLLPLV7_LOWER", "ANLLPLV7_UPPER",
                      "AY0DEPFN_LOWER", "BIIEY91S_LOWER", "C3TQ47Z0_UPPER", "CIGUMDD0_UPPER", "CXAJM3O9_LOWER",
                      "DG27PDD4_lower", "E23G704K_LOWER"
                      ]
        anorm_list = [x.lower() for x in anorm_list]
        
        flag=True
        for anorm_path in anorm_list:
            if anorm_path in batch_item[-1][0].lower():
                flag=False
        if flag:
            return 0

        info_save_path = os.path.join("results", "centerpoint_detection_test_info")
        basename = os.path.basename(batch_item[-1][0]).replace(".npy", "")
        print(info_save_path)
        os.makedirs(info_save_path, exist_ok = True)

        print(batch_item[-1])
        points = batch_item[0].cuda()
        seg_label = batch_item[2].cuda()
        gt_seg_label = gu.torch_to_numpy(batch_item[2])
        with torch.no_grad():
            output = self.ins_cent_model([points, seg_label])

        org_xyz_cpu = gu.torch_to_numpy(points)[0,:3,:].T
        whole_pd = gu.torch_to_numpy(output["sem_1"])[0,:,:].T
        np.save(os.path.join(info_save_path, f"{basename}_whole_pd.npy"), whole_pd)
        whole_pd = np.argmax(whole_pd, axis=1)
        full_labeled_points = np.concatenate([org_xyz_cpu, whole_pd.reshape(-1,1)], axis=1)
        org_stl_path = options["get_mesh_path"](os.path.basename(batch_item[-1][0]))
        np.save(os.path.join(info_save_path, f"{basename}_full_labeled_points.npy"), full_labeled_points)

        #gu.print_3d(gu.np_to_pcd_with_label(full_labeled_points))

        B = output["sem_2"].shape[0]
        
        for i in range(B):
            xyz_cpu = gu.torch_to_numpy(output["cropped_feature_ls"])[i,:3,:].T
            cls_pred = np.argmax(gu.torch_to_numpy(output["sem_2"][i,:]).T, axis=1)
            #gu.print_3d(gu.np_to_pcd_with_label(np.concatenate([xyz_cpu, cls_pred.reshape(-1,1)], axis=1)))

        whole_pd = torch.zeros((points.shape[2], 17)).cuda()
        whole_pd_count = torch.zeros(points.shape[2]).cuda()
        #whole_pd => 1,1, 포인트 수
        for crop_idx in range(B):
            pd_mask = output["sem_2"][crop_idx, :, :].permute(1,0) # 3072,17
            #크롭내부에서의 확률
            inside_crop_idx = output["nn_crop_indexes"][0][crop_idx]
            # 안에 있는 포인트의 index들
            whole_pd[inside_crop_idx] += pd_mask
            whole_pd_count[inside_crop_idx] += 1
        
        org_xyz_cpu = gu.torch_to_numpy(points)[0,:3,:].T
        whole_pd = gu.torch_to_numpy(whole_pd)
        #np.save(os.path.join(info_save_path, "whole_pd.npy"), whole_pd)
        whole_pd = np.argmax(whole_pd, axis=1)
        full_cls_labeled_points = np.concatenate([org_xyz_cpu, whole_pd.reshape(-1,1)], axis=1)
        org_stl_path = options["get_mesh_path"](os.path.basename(batch_item[-1][0]))
        #gu.print_3d(gu.np_to_pcd_with_label(full_labeled_points))
        full_stl_path = options["get_mesh_path"](os.path.basename(batch_item[-1][0]))
        #mu.save_result_mesh_with_label_chl(full_stl_path, full_labeled_points, "results/0609_tf_cbl_sem_results")

        """
        for i in range(B):
            xyz_cpu = gu.torch_to_numpy(output["cropped_feature_ls"])[i,:3,:].T
            cls_pred = np.argmax(gu.torch_to_numpy(output["mask_2"][i,:]).T, axis=1)
            #gu.print_3d(gu.np_to_pcd_with_label(np.concatenate([xyz_cpu, cls_pred.reshape(-1,1)], axis=1)))

        org_xyz_cpu = gu.torch_to_numpy(points)[0,:3,:].T
        whole_pd = gu.torch_to_numpy(output["mask_1"])[0,:,:].T
        whole_pd = np.argmax(whole_pd, axis=1)
        full_labeled_points = np.concatenate([org_xyz_cpu, whole_pd.reshape(-1,1)], axis=1)
        org_stl_path = options["get_mesh_path"](os.path.basename(batch_item[-1][0]))
        #gu.print_3d(gu.np_to_pcd_with_label(full_labeled_points))
        """

        whole_pd = torch.zeros((points.shape[2], 2)).cuda()
        whole_pd_count = torch.zeros(points.shape[2]).cuda()
        #whole_pd => 1,1, 포인트 수
        for crop_idx in range(B):
            pd_mask = output["mask_2"][crop_idx, :, :].permute(1,0) # 3072,17
            #크롭내부에서의 확률
            inside_crop_idx = output["nn_crop_indexes"][0][crop_idx]
            # 안에 있는 포인트의 index들
            whole_pd[inside_crop_idx] += pd_mask
            whole_pd_count[inside_crop_idx] += 1
        
        org_xyz_cpu = gu.torch_to_numpy(points)[0,:3,:].T
        whole_pd = gu.torch_to_numpy(whole_pd)
        fg_labels = np.argmax(whole_pd, axis=1)
        full_labeled_points = np.concatenate([org_xyz_cpu, fg_labels.reshape(-1,1)], axis=1)
        org_stl_path = options["get_mesh_path"](os.path.basename(batch_item[-1][0]))
        #gu.print_3d(gu.np_to_pcd_with_label(full_labeled_points))
        #mu.save_result_mesh_with_label_chl(org_stl_path, full_labeled_points, f"results/{options['checkpoint_path']}")

        
        #기본 offset
        moved_points_cpu = org_xyz_cpu + gu.torch_to_numpy(output["offset_1"])[0,:3,:].T
        fg_moved_points = moved_points_cpu[fg_labels==1, :]
        #full_cls_labeled_points_fg = full_cls_labeled_points[fg_labels==1, :]

        np.save(os.path.join(info_save_path, f"{basename}_fg_moved_points.npy"), moved_points_cpu)
        #np.save(os.path.join(info_save_path, f"{basename}_full_cls_labeled_points_fg.npy"), full_cls_labeled_points_fg)
        

        #gu.print_3d(gu.np_to_pcd(fg_moved_points))


        num_of_clusters = []
        for b_idx in range(1):
            num_of_clusters.append(len(np.unique(gt_seg_label[b_idx,:]))-1)

        cluster_centroids, cluster_centroids_labels, fg_points_labels_ls = tu.clustering_points(
            [fg_moved_points], 
            method="kmeans", 
            num_of_clusters=num_of_clusters
        )
        fg_points = np.concatenate([org_xyz_cpu[fg_labels==1, :], fg_points_labels_ls[0].reshape(-1,1)], axis=1)
        fg_points[:,3] +=1
        
        bg_points = org_xyz_cpu[fg_labels==0, :]
        bg_points = np.concatenate([bg_points, np.zeros([org_xyz_cpu.shape[0]-fg_points.shape[0],1])], axis=1)
        
        full_labeled_points = np.concatenate([fg_points, bg_points], axis=0)
        #gu.print_3d(gu.np_to_pcd_with_label(full_labeled_points))
        full_stl_path = options["get_mesh_path"](os.path.basename(batch_item[-1][0]))
        #mu.save_result_mesh_with_label_chl(full_stl_path, full_labeled_points, "results/0611_tf_cbl_offset_only_coord_mask", True)

    def infer_uniform(self, batch_idx, batch_item, **options):
        points = batch_item["feat"].cuda()
        seg_label = batch_item["gt_seg_label"].cuda()
        gt_seg_label = gu.torch_to_numpy(seg_label)
        uniform_points = batch_item["uniform_feat"].cuda()
        uniform_gt_seg_label = batch_item["uniform_gt_seg_label"].cuda()
        inputs = [points, seg_label]
        if self.config["model_parameter"]["use_uniform_sampled_on_crop"]:
            inputs += [uniform_points, uniform_gt_seg_label]

        with torch.no_grad():
            output = self.model(inputs)


        org_xyz_cpu = gu.torch_to_numpy(points)[0,:3,:].T
        crop_num = output["sem_2"].shape[0]
        
        results = {}

        whole_pd_sem_1 = gu.torch_to_numpy(output["sem_1"])[0,:,:].T
        whole_cls_1 = np.argmax(whole_pd_sem_1, axis=1)
        full_labeled_points_1 = np.concatenate([org_xyz_cpu, whole_cls_1.reshape(-1,1)], axis=1)

        results["sem_1"] = {}
        results["sem_1"]["full_labeled_points"] = full_labeled_points_1
        results["sem_1"]["whole_pd_sem"] = whole_pd_sem_1

        uni_org_xyz_cpu = gu.torch_to_numpy(uniform_points)[0,:3,:].T
        whole_pd_sem_2 = torch.zeros((uniform_points.shape[2], output["sem_2"].shape[1])).cuda()
        whole_pd_count_2 = torch.zeros(uniform_points.shape[2]).cuda()
        #whole_pd => 1,1, 포인트 수
        for crop_idx in range(crop_num):
            pd_mask = output["sem_2"][crop_idx, :, :].permute(1,0) # 3072,17
            #크롭내부에서의 확률
            inside_crop_idx = output["nn_crop_indexes"][0][crop_idx]
            # 안에 있는 포인트의 index들
            whole_pd_sem_2[inside_crop_idx] += pd_mask
            whole_pd_count_2[inside_crop_idx] += 1
        
        whole_pd_sem_2 = gu.torch_to_numpy(whole_pd_sem_2)
        whole_pd_count_2 = gu.torch_to_numpy(whole_pd_count_2)
        whole_pd_sem_2[whole_pd_count_2!=0, : ] /= whole_pd_count_2[whole_pd_count_2!=0].reshape(-1,1)
        whole_cls_2 = np.argmax(whole_pd_sem_2, axis=1)
        full_labeled_points_2 = np.concatenate([uni_org_xyz_cpu , whole_cls_2.reshape(-1,1)], axis=1)
        roi_uni_idxes =  np.unique(output["nn_crop_indexes"][0])
        full_labeled_points_2 = full_labeled_points_2[roi_uni_idxes, :]

        results["uni_sem_2"] = {}
        results["uni_sem_2"]["full_labeled_points"] = full_labeled_points_2
        results["uni_sem_2"]["whole_pd_sem"] = whole_pd_sem_2

        tree = KDTree(full_labeled_points_2[:,:3], leaf_size=2)
        nn_uni_points_idxes = tree.query(org_xyz_cpu, k=1, return_distance=False).reshape(-1)
        org_sem_2_labels = results["uni_sem_2"]["full_labeled_points"][nn_uni_points_idxes, 3]
        org_full_labeled_points_2 = np.concatenate([org_xyz_cpu, org_sem_2_labels.reshape(-1,1)],axis=1)
        results["sem_2"] = {}
        results["sem_2"]["full_labeled_points"] = org_full_labeled_points_2
        gu.print_3d(gu.np_to_pcd_with_label(org_full_labeled_points_2))


        gin_infer_miss_points = results["sem_2"]["full_labeled_points"].copy()
        gin_infer_miss_points_mask = (results["sem_2"]["full_labeled_points"][:,3] >= 1) != (gt_seg_label.reshape(-1) >= 0)
        gin_infer_miss_points[gin_infer_miss_points_mask, 3] = 1
        gin_infer_miss_points[np.invert(gin_infer_miss_points_mask), 3] = 0
        gu.print_3d(gu.np_to_pcd_with_label(gin_infer_miss_points))

        gin_infer_miss_points = results["uni_sem_2"]["full_labeled_points"].copy()
        gin_infer_miss_points_mask = (results["uni_sem_2"]["full_labeled_points"][:,3] >= 1) != (gu.torch_to_numpy(uniform_gt_seg_label).reshape(-1)[roi_uni_idxes] >= 0)
        gin_infer_miss_points[gin_infer_miss_points_mask, 3] = 1
        gin_infer_miss_points[np.invert(gin_infer_miss_points_mask), 3] = 0
        gu.print_3d(gu.np_to_pcd_with_label(gin_infer_miss_points))


        results["offset_1"] = {}
        results["offset_1"]["moved_points"] = org_xyz_cpu + gu.torch_to_numpy(output["offset_1"])[0,:3,:].T

        if True:
            #if batch_idx == 0:
                self.offset_acc = 0
                self.mask_acc = 0
                self.sem_acc = 0
                self.sem_ins_acc = 0
                self.gt_to_pred_iou_acc = 0
                self.pred_to_gt_iou_acc = 0


    def infer(self, batch_idx, batch_item, **options):
        if self.config["model_parameter"]["use_uniform_sampled_on_crop"]:
            self.infer_uniform(batch_idx, batch_item, **options)
            return
        self._set_model("val")
        print(batch_item["mesh_path"])
        #points = batch_item[0].cuda()
        #seg_label = batch_item[2].cuda()
        #points, seg_label = self.get_boundary_sampled_points(batch_item)

        points = batch_item["feat"].cuda()
        seg_label = batch_item["gt_seg_label"].cuda()
        gt_seg_label = gu.torch_to_numpy(seg_label)

        inputs = [points, seg_label]
        with torch.no_grad():
            output = self.model(inputs)


        org_xyz_cpu = gu.torch_to_numpy(points)[0,:3,:].T
        crop_num = output["sem_2"].shape[0]
        
        results = {}

        whole_pd_sem_1 = gu.torch_to_numpy(output["sem_1"])[0,:,:].T
        whole_cls_1 = np.argmax(whole_pd_sem_1, axis=1)
        full_labeled_points_1 = np.concatenate([org_xyz_cpu, whole_cls_1.reshape(-1,1)], axis=1)

        results["sem_1"] = {}
        results["sem_1"]["full_labeled_points"] = full_labeled_points_1
        results["sem_1"]["whole_pd_sem"] = whole_pd_sem_1

        for i in range(crop_num):
            xyz_cpu = gu.torch_to_numpy(output["cropped_feature_ls"])[i,:3,:].T
            cls_pred = np.argmax(gu.torch_to_numpy(output["sem_2"][i,:]).T, axis=1)
            #gu.print_3d(gu.np_to_pcd_with_label(np.concatenate([xyz_cpu, cls_pred.reshape(-1,1)], axis=1)))
        
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
        
        whole_pd_sem_2 = gu.torch_to_numpy(whole_pd_sem_2)
        whole_pd_count_2 = gu.torch_to_numpy(whole_pd_count_2)
        whole_pd_sem_2[whole_pd_count_2!=0, : ] /= whole_pd_count_2[whole_pd_count_2!=0].reshape(-1,1)
        whole_cls_2 = np.argmax(whole_pd_sem_2, axis=1)
        full_labeled_points_2 = np.concatenate([org_xyz_cpu, whole_cls_2.reshape(-1,1)], axis=1)
        
        results["sem_2"] = {}
        results["sem_2"]["full_labeled_points"] = full_labeled_points_2
        results["sem_2"]["whole_pd_sem"] = whole_pd_sem_2

        results["offset_1"] = {}
        results["offset_1"]["moved_points"] = org_xyz_cpu + gu.torch_to_numpy(output["offset_1"])[0,:3,:].T

        whole_pd_sem_3 = gu.torch_to_numpy(output["sem_1"])[0,1:,:].T
        whole_cls_3 = np.argmax(whole_pd_sem_3, axis=1)+1
        full_labeled_points_3 = np.concatenate([org_xyz_cpu, whole_cls_3.reshape(-1,1)], axis=1)
        #gu.print_3d(gu.np_to_pcd_with_label(full_labeled_points_3))


        # ================================================ #
        # centerpoint detection test part #
        #from testmodel.test_script.centerpoint_detection_test.test import clustering_moved_points
        #clustering_moved_points(whole_pd_cls_1, moved_points_cpu)
        

        fg_moved_points = results["offset_1"]["moved_points"][results["sem_2"]["full_labeled_points"][:,3]!=0, :]
        num_of_clusters = []
        for b_idx in range(1):
            num_of_clusters.append(len(np.unique(gt_seg_label[b_idx,:]))-1)

        #cluster_centroids, cluster_centroids_labels, fg_points_labels_ls = tu.clustering_points(
        #    [fg_moved_points], 
        #    method="kmeans", 
        #    num_of_clusters=num_of_clusters
        #)
        fg_points_labels_ls = [tu.get_clustering_labels(results["offset_1"]["moved_points"], results["sem_2"]["full_labeled_points"][:,3])]

        points_ins_labels = np.zeros(org_xyz_cpu.shape[0])
        points_ins_labels[:] = -1
        points_ins_labels[np.where(results["sem_2"]["full_labeled_points"][:,3]!=0)] = fg_points_labels_ls[b_idx]
        
        full_ins_labeled_points = np.concatenate([org_xyz_cpu, points_ins_labels.reshape(-1,1)], axis=1)
        full_ins_labeled_points[:,3] += 1
        results["ins"] = {}
        results["ins"]["full_ins_labeled_points"] = full_ins_labeled_points

        def get_ins_labeled_points(org_xyz_cpu, offset_points, labels):
            fg_points_labels_ls = [tu.get_clustering_labels(offset_points, labels)]

            points_ins_labels = np.zeros(org_xyz_cpu.shape[0])
            points_ins_labels[:] = -1
            points_ins_labels[np.where(labels!=0)] = fg_points_labels_ls[0]
        
            full_sem_1_ins_labeled_points = np.concatenate([org_xyz_cpu, points_ins_labels.reshape(-1,1)], axis=1)
            full_sem_1_ins_labeled_points[:,3] += 1
            return full_sem_1_ins_labeled_points
        
        results["sem_1_ins"] = {}
        results["sem_1_ins"]["full_ins_labeled_points"] = get_ins_labeled_points(org_xyz_cpu, results["offset_1"]["moved_points"], full_labeled_points_1[:,3])


        bin_gt_mask = gt_seg_label.copy().reshape(-1)
        bin_gt_mask[bin_gt_mask>=0] = 1
        bin_gt_mask[bin_gt_mask==-1] = 0
        gt_masked_moved_points = results["offset_1"]["moved_points"][bin_gt_mask==1, :]
        results["offset_1"]["gt_masked_moved_points"] = gt_masked_moved_points

        results["gt_ins"] = {}
        results["gt_ins"]["full_ins_labeled_points"] = get_ins_labeled_points(org_xyz_cpu, results["offset_1"]["moved_points"], bin_gt_mask)
        
        if True:
            if batch_idx == 0:
                self.gt_to_pred_sem_1_iou = 0
                self.pred_to_gt_sem_1_iou = 0
                self.gt_to_pred_gt_mask_iou = 0
                self.pred_to_gt_gt_mask_iou = 0
                self.sem_1_acc = 0
                self.sem_1_mask_acc = 0
        if True:
            #iou accruacy
            gt_to_pred_sem_1_iou = mu.gt_to_pred_cal_metric(gt_seg_label.reshape(-1)+1, results["sem_1_ins"]["full_ins_labeled_points"][:,3], results["sem_1_ins"]["full_ins_labeled_points"][:,3], is_half=True)        
            self.gt_to_pred_sem_1_iou += gt_to_pred_sem_1_iou[0]
            if min(gt_to_pred_sem_1_iou[4])<0.9:
                a=1
            if gt_to_pred_sem_1_iou[0] < 0.93:
                a=1

            #iou accruacy
            pred_to_gt_sem_1_iou = mu.cal_metric(gt_seg_label.reshape(-1)+1, results["sem_1_ins"]["full_ins_labeled_points"][:,3], results["sem_1_ins"]["full_ins_labeled_points"][:,3], is_half=True)        
            self.pred_to_gt_sem_1_iou += pred_to_gt_sem_1_iou[0]
            if min(pred_to_gt_sem_1_iou[4])<0.9:
                a=1
            if pred_to_gt_sem_1_iou[0] < 0.93:
                a=1

            #iou accruacy
            gt_to_pred_gt_mask_iou = mu.cal_metric(gt_seg_label.reshape(-1)+1, results["gt_ins"]["full_ins_labeled_points"][:,3], results["gt_ins"]["full_ins_labeled_points"][:,3], is_half=True)        
            self.gt_to_pred_gt_mask_iou += gt_to_pred_gt_mask_iou[0]
            if min(gt_to_pred_gt_mask_iou[4])<0.9:
                a=1
            if gt_to_pred_gt_mask_iou[0] < 0.93:
                a=1

            #iou accruacy
            pred_to_gt_gt_mask_iou = mu.cal_metric(gt_seg_label.reshape(-1)+1, results["gt_ins"]["full_ins_labeled_points"][:,3], results["gt_ins"]["full_ins_labeled_points"][:,3], is_half=True)        
            self.pred_to_gt_gt_mask_iou += pred_to_gt_gt_mask_iou[0]
            if min(pred_to_gt_gt_mask_iou[4])<0.9:
                a=1
            if pred_to_gt_gt_mask_iou[0] < 0.93:
                a=1

            #sem accuracy
            sem_1_acc = mu.cal_metric(gt_seg_label.reshape(-1)+1, results["sem_1"]["full_labeled_points"][:,3], gt_seg_label.reshape(-1), is_half=True)[3]
            self.sem_1_acc += sem_1_acc
            if sem_1_acc<0.99:
                a=1

            sem_1_mask_acc = mu.cal_metric(gt_seg_label.reshape(-1)+1, full_labeled_points_3[:,3], gt_seg_label.reshape(-1)+1, is_half=True)[3]
            self.sem_1_mask_acc += sem_1_mask_acc
            if sem_1_mask_acc<0.99:
                a=1

        if False:
            if batch_idx == 0:
                self.offset_acc = 0
                self.mask_acc = 0
                self.sem_acc = 0
                self.sem_ins_acc = 0
                self.gt_to_pred_iou_acc = 0
                self.pred_to_gt_iou_acc = 0
            #results["ins"]["full_ins_labeled_points"][:,3]
            #self.ins_results += np.array(mu.cal_metric(gt_seg_label.reshape(-1)+1, results["sem_1"]["full_labeled_points"][:,3], results["ins"]["full_ins_labeled_points"][:,3], is_half=True))
            #self.sem_results += np.array(mu.cal_metric(gt_seg_label.reshape(-1)+1, results["sem_1"]["full_labeled_points"][:,3], gt_seg_label.reshape(-1)+1, is_half=True))
        
        if False:
            #iou accruacy
            gt_to_pred_iou_acc = mu.gt_to_pred_cal_metric(gt_seg_label.reshape(-1)+1, results["ins"]["full_ins_labeled_points"][:,3], results["ins"]["full_ins_labeled_points"][:,3], is_half=True)        
            self.gt_to_pred_iou_acc += gt_to_pred_iou_acc[0]
            if min(gt_to_pred_iou_acc[4])<0.9:
                a=1
            if gt_to_pred_iou_acc[0] < 0.93:
                a=1

            #iou accruacy
            pred_to_gt_iou_acc = mu.cal_metric(gt_seg_label.reshape(-1)+1, results["ins"]["full_ins_labeled_points"][:,3], results["ins"]["full_ins_labeled_points"][:,3], is_half=True)        
            self.pred_to_gt_iou_acc += pred_to_gt_iou_acc[0]
            if min(pred_to_gt_iou_acc[4])<0.9:
                a=1
            if pred_to_gt_iou_acc[0] < 0.93:
                a=1

            #offset accuracy
            offset_acc = mu.cal_metric(gt_seg_label.reshape(-1)+1, results["gt_ins"]["full_ins_labeled_points"][:,3], results["gt_ins"]["full_ins_labeled_points"][:,3], is_half=True)[0]
            self.offset_acc += offset_acc
            if offset_acc<0.93:
                a=1

            #sem-ins accuracy
            #sem_ins_acc = mu.cal_metric(gt_seg_label.reshape(-1)+1, results["sem_1"]["full_labeled_points"][:,3], results["sem_1"]["full_labeled_points"][:,3], is_half=True)[0]
            #self.sem_ins_acc += sem_ins_acc
            #print(sem_ins_acc)
            #print(offset_acc)
            
            #sem mask accuracy
            bin_sem_1_mask = results["sem_1"]["full_labeled_points"][:,3].copy()
            bin_sem_1_mask[bin_sem_1_mask>=1] = 1
            #print("sim_mask", mu.cal_metric(bin_gt_mask, bin_sem_1_mask, bin_sem_1_mask, is_half=True)[0])

            #mask accuracy
            mask_acc = mu.cal_metric(bin_gt_mask, results["sem_2"]["full_labeled_points"][:,3], results["sem_2"]["full_labeled_points"][:,3], is_half=True)[0]
            self.mask_acc += mask_acc
            if mask_acc<0.90:
                a=1
                
            #sem accuracy
            sem_acc = mu.cal_metric(gt_seg_label.reshape(-1)+1, results["sem_1"]["full_labeled_points"][:,3], gt_seg_label.reshape(-1), is_half=True)[3]
            self.sem_acc += sem_acc
            if sem_acc<0.99:
                a=1

        #기본 offset
        if False:
            #sem1 prob vis
            whole_pd_points = results["sem_1"]["whole_pd_sem"]
            arg_max_arr = np.argmax(whole_pd_points, axis=1)
            whole_pd_points = tu.softmax(whole_pd_points)
            whole_max_pd_points_1 = whole_pd_points[np.arange(len(whole_pd_points)), arg_max_arr]
            whole_max_pd_points_1 = whole_max_pd_points_1.reshape(-1,1)
            gu.print_3d(gu.np_to_pcd_with_prob(np.concatenate([org_xyz_cpu, whole_max_pd_points_1], axis=1)))

            #sm2 prob vis
            whole_pd_points = results["sem_2"]["whole_pd_sem"]
            arg_max_arr = np.argmax(whole_pd_points, axis=1)
            whole_pd_points = tu.softmax(whole_pd_points)
            whole_max_pd_points_2 = whole_pd_points[np.arange(len(whole_pd_points)), arg_max_arr]
            whole_max_pd_points_2 = whole_max_pd_points_2.reshape(-1,1)
            gu.print_3d(gu.np_to_pcd_with_prob(np.concatenate([org_xyz_cpu, whole_max_pd_points_2], axis=1)))

            #합친거
            sem_1_pd_points = np.zeros((24000,2))
            sem_1_pd_points[:, 0] = tu.softmax(results["sem_1"]["whole_pd_sem"])[:, 0]
            sem_1_pd_points[:, 1] = np.sum(tu.softmax(results["sem_1"]["whole_pd_sem"])[:, 1:], axis=1)
            whole_pd_points = (sem_1_pd_points + tu.softmax(results["sem_2"]["whole_pd_sem"]))/2
            arg_max_arr = np.argmax(whole_pd_points, axis=1).reshape(-1,1)
            whole_max_pd_points_2 = whole_pd_points[np.arange(len(whole_pd_points)), arg_max_arr]
            whole_max_pd_points_2 = whole_max_pd_points_2.reshape(-1,1)

            gu.print_3d(gu.np_to_pcd_with_prob(np.concatenate([org_xyz_cpu, whole_max_pd_points_2], axis=1)))
            gu.print_3d(gu.np_to_pcd_with_label(np.concatenate([org_xyz_cpu,arg_max_arr], axis=1)))
            #gt
            #gu.print_3d(gu.np_to_pcd_with_label(np.concatenate([org_xyz_cpu, gt_seg_label.reshape(-1,1)],axis=1)))

            #gu.print_3d(gu.np_to_pcd_with_label(results["mask_1"]["full_masked_points"]))
            #gu.print_3d(gu.np_to_pcd_with_label(results["mask_2"]["full_masked_points"]))
            gu.print_3d(gu.np_to_pcd_with_label(results["sem_1"]["full_labeled_points"]))
            #
            #gu.print_3d(gu.np_to_pcd_with_label(results["sem_2"]["full_labeled_points"]))
            #gu.print_3d(results["offset_1"]["moved_points"])
            #gu.print_3d(results["offset_1"]["moved_points"][results["sem_2"]["full_labeled_points"][:,3]!=0, :])
            gu.print_3d(gu.np_to_pcd_with_label(results["ins"]["full_ins_labeled_points"]))            

            #gu.print_3d(gu.np_to_pcd_with_label(results["gt_ins"]["full_ins_labeled_points"]))
            #gu.print_3d(gu.np_to_pcd_with_label(np.concatenate([gt_masked_moved_points, gt_fg_points_labels_ls[0].reshape(-1,1)], axis=1)))
            #gu.print_3d(gu.np_to_pcd_with_label(np.concatenate([gt_masked_moved_points, gt_seg_label.reshape(-1,1)[bin_gt_mask==1, :]], axis=1)))

            gin_infer_miss_points = results["sem_1"]["full_labeled_points"].copy()
            gin_infer_miss_points_mask = (results["sem_1"]["full_labeled_points"][:,3] >= 1) != (gt_seg_label.reshape(-1) >= 0)
            gin_infer_miss_points[gin_infer_miss_points_mask, 3] = 1
            gin_infer_miss_points[np.invert(gin_infer_miss_points_mask), 3] = 0
            gu.print_3d(gu.np_to_pcd_with_label(gin_infer_miss_points))

            gin_infer_miss_points = results["sem_2"]["full_labeled_points"].copy()
            gin_infer_miss_points_mask = (results["sem_2"]["full_labeled_points"][:,3] >= 1) != (gt_seg_label.reshape(-1) >= 0)
            gin_infer_miss_points[gin_infer_miss_points_mask, 3] = 1
            gin_infer_miss_points[np.invert(gin_infer_miss_points_mask), 3] = 0
            gu.print_3d(gu.np_to_pcd_with_label(gin_infer_miss_points))
        return
        #mu.save_result_mesh_with_label_chl(full_stl_path, full_labeled_points, "results/0622_tf_cbl_offset_only_coord_mask_ins", True)
        #full_stl_path = options["get_mesh_path"](os.path.basename(batch_item[-1][0]))

