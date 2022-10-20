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

class PTNSecondModule(torch.nn.Module):
    def __init__(self, config):

        self.config = config
        super().__init__()
        self.second_ins_cent_model = pointnext.pointnext_imp.get_model(self.config["model_parameter"]["detail_model_config"], c=self.config["model_parameter"]["input_dim"], k=2)

    def forward(self, inputs, test=False, cent_aug=False):
        DEBUG=False
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1] => B, 1, 24000 : ground truth segmentation
            inputs[2] => B, 6, 24000 : uniform_point_features
        """
        B, C, N = inputs[0].shape
        outputs = {}

        """
        with torch.no_grad():
            cent_output = self.cent_model(inputs[0]) 
        
        offset_result = gu.torch_to_numpy(cent_output["offset_result"][0])
        dist_result = gu.torch_to_numpy(cent_output["dist_result"][0])
        l3_xyz = gu.torch_to_numpy(cent_output["l3_xyz"][0]).T
        l0_xyz = gu.torch_to_numpy(cent_output["l0_xyz"][0]).T
        centroids_coord = []
        for i in range(16):
            tooth_offset = offset_result[3*i:3*(i+1),:].T
            tooth_dist = dist_result[i,:]
            moved_points = l3_xyz + tooth_offset
            moved_points = moved_points[tooth_dist<0.2]
            if np.any(np.isnan(np.mean(moved_points,axis=0))):
                continue
            centroids_coord.append(np.mean(moved_points,axis=0))
        centroids_coord = np.array(centroids_coord)
        """

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
        cluster_centroids = np.array(cluster_centroids)

        if cent_aug:
            change_sets = np.random.permutation(cluster_centroids.shape[1])
            cluster_centroids[0, :] = cluster_centroids[0, :] + (np.random.rand(change_sets.shape[0],3)-0.5)/10
            cluster_centroids[0, change_sets[:5]] = cluster_centroids[0, change_sets[:5]] + (np.random.rand(change_sets[:5].shape[0],3)-0.5)/2


        org_xyz_cpu = gu.torch_to_numpy(inputs[0][:, :3, :].permute(0, 2, 1))
        nn_crop_indexes = tu.get_nearest_neighbor_idx(org_xyz_cpu, cluster_centroids, 3072)
        cropped_feature_ls = tu.get_indexed_features(inputs[0], nn_crop_indexes)
        if len(inputs)>=2:
            cluster_gt_seg_label = tu.get_indexed_features(inputs[1], nn_crop_indexes)

        cropped_feature_ls = tu.centering_object(cropped_feature_ls)
        if self.config["model_parameter"]["add_features_to_crop"]:
            cropped_feature_ls = cropped_feature_ls.permute(0,2,1)
            low_add_feat = torch.zeros((cropped_feature_ls.shape[0], cropped_feature_ls.shape[1], 4)).cuda()

            low_add_feat[:,:, 0] = torch.sum(cropped_feature_ls[:,:, :3] * cropped_feature_ls[:, :, 3:],dim=2)
            rho = torch.norm(cropped_feature_ls[:,:, :3],dim=2)
            theta = torch.arccos(cropped_feature_ls[:,:,2]/rho)
            phi = torch.atan2(cropped_feature_ls[:,:, 1], cropped_feature_ls[:,:, 0])

            theta[rho==0] = 0
            theta = theta/np.pi
            phi = phi/(2*np.pi) + .5
            low_add_feat[:,:,1] = rho
            low_add_feat[:,:,2] = theta
            low_add_feat[:,:,3] = phi
            cropped_feature_ls = torch.cat([cropped_feature_ls, low_add_feat],dim=2).permute(0,2,1)

        if DEBUG:
            for i in range(cropped_feature_ls.shape[0]):
                xyzs = gu.torch_to_numpy(cropped_feature_ls[i,:3,:]).T
                labels = gu.torch_to_numpy(cluster_gt_seg_label[i,:1,:]).T
                gu.print_3d(gu.np_to_pcd_with_label(np.concatenate([xyzs, labels], axis=1)), gu.make_coord_frame())

        crop_input_features = cropped_feature_ls

        cluster_gt_seg_label[cluster_gt_seg_label>=0] = 0
        outputs["cluster_gt_seg_label"] = cluster_gt_seg_label
        sem_2, offset_2, mask_2, _ = self.second_ins_cent_model(crop_input_features) 

        outputs.update({
            "sem_2": sem_2,
            "offset_2":offset_2,
            "mask_2":mask_2,
        })

        outputs["cropped_feature_ls"] = cropped_feature_ls
        outputs["nn_crop_indexes"] =  nn_crop_indexes

        return outputs

class TfCblSecondModule(torch.nn.Module):
    def __init__(self, config):

        self.config = config
        super().__init__()
        self.second_ins_cent_model = get_model(**config["model_parameter"], c=config["model_parameter"]["input_feat"], k=2).train().cuda()
        
        #self.cent_model = tsg_centroid_16_module.get_model(config)
        #self.cent_model.cuda()
        #self.cent_model.train()
        #self.cent_model.load_state_dict(torch.load("testmodel/ckpts/0713_roi_detection_val.h5"))
        #self.cent_model.load_state_dict(torch.load("testmodel/ckpts/cent/0331_chl_with_normal_val.h5"))

    def forward(self, inputs, test=False, cent_aug=False):
        DEBUG=False
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1] => B, 1, 24000 : ground truth segmentation
            inputs[2] => B, 6, 24000 : uniform_point_features
        """
        B, C, N = inputs[0].shape
        outputs = {}

        """
        with torch.no_grad():
            cent_output = self.cent_model(inputs[0]) 
        
        offset_result = gu.torch_to_numpy(cent_output["offset_result"][0])
        dist_result = gu.torch_to_numpy(cent_output["dist_result"][0])
        l3_xyz = gu.torch_to_numpy(cent_output["l3_xyz"][0]).T
        l0_xyz = gu.torch_to_numpy(cent_output["l0_xyz"][0]).T
        centroids_coord = []
        for i in range(16):
            tooth_offset = offset_result[3*i:3*(i+1),:].T
            tooth_dist = dist_result[i,:]
            moved_points = l3_xyz + tooth_offset
            moved_points = moved_points[tooth_dist<0.2]
            if np.any(np.isnan(np.mean(moved_points,axis=0))):
                continue
            centroids_coord.append(np.mean(moved_points,axis=0))
        centroids_coord = np.array(centroids_coord)
        """

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
        cluster_centroids = np.array(cluster_centroids)

        if cent_aug:
            change_sets = np.random.permutation(cluster_centroids.shape[1])
            cluster_centroids[0, :] = cluster_centroids[0, :] + (np.random.rand(change_sets.shape[0],3)-0.5)/10
            cluster_centroids[0, change_sets[:5]] = cluster_centroids[0, change_sets[:5]] + (np.random.rand(change_sets[:5].shape[0],3)-0.5)/2


        org_xyz_cpu = gu.torch_to_numpy(inputs[0][:, :3, :].permute(0, 2, 1))
        nn_crop_indexes = tu.get_nearest_neighbor_idx(org_xyz_cpu, cluster_centroids, 3072)
        cropped_feature_ls = tu.get_indexed_features(inputs[0], nn_crop_indexes)
        if len(inputs)>=2:
            cluster_gt_seg_label = tu.get_indexed_features(inputs[1], nn_crop_indexes)

        cropped_feature_ls = tu.centering_object(cropped_feature_ls)
        if DEBUG:
            for i in range(cropped_feature_ls.shape[0]):
                xyzs = gu.torch_to_numpy(cropped_feature_ls[i,:3,:]).T
                labels = gu.torch_to_numpy(cluster_gt_seg_label[i,:1,:]).T
                #gu.print_3d(gu.np_to_pcd_with_label(np.concatenate([xyzs, labels], axis=1)), gu.make_coord_frame())
                gu.print_3d(gu.np_to_pcd_with_label(np.concatenate([xyzs, labels+1], axis=1)))

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

class TfCblSecondModel(BaseModel):
    def __init__(self, config, model):
        BaseModel.__init__(self, config, model)
        self.mask_acc = 0 
        self.mask_acc_array = []


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
        
    def get_loss(self, sem_2, gt_seg_label_2):
        tooth_class_loss_2 = tsg_ins_cent_loss.tooth_class_loss(sem_2, gt_seg_label_2, label_smoothing=self.config["tr_set"]["label_smoothing"])
        return {
            "tooth_class_loss_2": (tooth_class_loss_2, 1),
        }

    def step(self, batch_idx, batch_item, phase):
        self._set_model(phase)

        points = batch_item["feat"].cuda()
        l0_xyz = batch_item["feat"][:,:3,:].cuda()
        
        #centroids = batch_item[1].cuda()
        seg_label = batch_item["gt_seg_label"].cuda()
        
        inputs = [points, seg_label]

        if phase == "train":
            output = self.model(inputs, test=False, cent_aug=True)
        else:
            with torch.no_grad():
                output = self.model(inputs, test=False, cent_aug=False)
        loss_meter = LossMap()
        
        loss_meter.add_loss_by_dict(self.get_loss(
            output["sem_2"], 
            output["cluster_gt_seg_label"], 
            )
        )
        
        if phase == "train":
            if "cbl_loss_2" in output:
                loss_meter.add_loss("cbl_loss_2", output["cbl_loss_2"].sum(), self.config["tr_set"]["loss"]["cbl"])
            loss_sum = loss_meter.get_sum()
            self.optimizer.zero_grad()
            loss_sum.backward()
            if torch.isnan(loss_sum).any():
                a=1
            self.optimizer.step()

        return loss_meter
    
        #필요한 함수
        #loss를 등록
        #loss의 weight도 등록
        #loss를 반환(dictionary 형태로, total도 함께 넣어서, item 형태로 모두)
        #loss를 summation하여 반환 - backprop용도

        

    def infer(self, batch_idx, batch_item, **options):
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
        results = {}

        org_xyz_cpu = gu.torch_to_numpy(points)[0,:3,:].T
        crop_num = output["sem_2"].shape[0]
        
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
        results["sem_2"]["whole_pd_sem"] = whole_pd_sem_2
        results["sem_2"]["full_labeled_points"] = full_labeled_points_2


        bin_gt_mask = gt_seg_label.copy().reshape(-1)
        bin_gt_mask[bin_gt_mask>=0] = 1
        bin_gt_mask[bin_gt_mask==-1] = 0

        if True:
            mask_acc = mu.cal_metric(bin_gt_mask, results["sem_2"]["full_labeled_points"][:,3], results["sem_2"]["full_labeled_points"][:,3], is_half=True)[1]
            self.mask_acc += mask_acc
            self.mask_acc_array.append(mask_acc)
            if mask_acc<0.96:
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

            gin_infer_miss_points = results["sem_2"]["full_labeled_points"].copy()
            gin_infer_miss_points_mask = (results["sem_2"]["full_labeled_points"][:,3] >= 1) != (gt_seg_label.reshape(-1) >= 0)
            gin_infer_miss_points[gin_infer_miss_points_mask, 3] = 1
            gin_infer_miss_points[np.invert(gin_infer_miss_points_mask), 3] = 0
            gu.print_3d(gu.np_to_pcd_with_label(gin_infer_miss_points))
            gu.print_3d(gu.np_to_pcd_with_label(results["sem_2"]["full_labeled_points"]))

        return
        #mu.save_result_mesh_with_label_chl(full_stl_path, full_labeled_points, "results/0622_tf_cbl_offset_only_coord_mask_ins", True)
        #full_stl_path = options["get_mesh_path"](os.path.basename(batch_item[-1][0]))
