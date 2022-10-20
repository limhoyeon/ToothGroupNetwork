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
class TfCblFirstMaskModule(torch.nn.Module):
    def __init__(self, config):
        self.config = config

        super().__init__()
        class_num = 1
        self.first_ins_cent_model = get_model(**config["model_parameter"], c=config["model_parameter"]["input_feat"], k=class_num + 1)

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
            bin_seg_label = inputs[1].clone()
            bin_seg_label[bin_seg_label>=0] = 0
            cbl_loss_1, sem_1, offset_1, mask_1, first_features = self.first_ins_cent_model([inputs[0], bin_seg_label])
            cbl_loss_2, sem_2, offset_2, mask_2, second_features =  [None]*5
            outputs.update({
                "cbl_loss_1": cbl_loss_1,
                "sem_1": sem_1,
                "offset_1":offset_1,
                "mask_1":mask_1,
                "first_features": first_features,
                "cbl_loss_2": cbl_loss_2,
                "sem_2": sem_2,
                "offset_2":offset_2,
                "mask_2":mask_2,
                "second_features": second_features
            })
        else:
            sem_1, offset_1, mask_1, first_features = self.first_ins_cent_model([inputs[0]])
            #cbl_loss_2, sem_2, offset_2, mask_2, second_features = self.second_ins_cent_model([inputs[0]])
            sem_2, offset_2, mask_2, second_features =  [None]*4
            outputs.update({
                "sem_1": sem_1,
                "offset_1":offset_1,
                "mask_1":mask_1,
                "first_features": first_features,
                "sem_2": sem_2,
                "offset_2":offset_2,
                "mask_2":mask_2,
                "second_features": second_features
            })
        return outputs

class TfCblFirstMaskModel(BaseModel):
    def __init__(self, config, model):
        BaseModel.__init__(self, config, model)
        self.mask_acc = 0 

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
        
    def get_loss(self, offset_1, sem_1, offset_2, sem_2, gt_seg_label_1, input_coords):
        bin_seg_label = gt_seg_label_1.clone()
        bin_seg_label[bin_seg_label>=0] = 0


        tooth_class_loss_1 = tsg_ins_cent_loss.tooth_class_loss(sem_1, bin_seg_label, label_smoothing=self.config["tr_set"]["label_smoothing"], weight=self.config["tr_set"]["first_cr_weight"])
        
        #tooth_class_loss_1 = tsg_ins_cent_loss.weighted_cls1_loss(weight_1, sem_1, half_seg_label, None)
        #tooth_class_loss_2 = tsg_ins_cent_loss.weighted_cls2_loss(weight_1, sem_2, half_seg_label, None)
        offset_1_loss, offset_1_dir_loss = tsg_ins_cent_loss.batch_center_offset_loss(offset_1, input_coords, gt_seg_label_1)
        #offset_2_loss, offset_2_dir_loss = tsg_ins_cent_loss.weighted_batch_center_offset_loss(offset_1, offset_2, input_coords, gt_seg_label_1)

        chamf_1_loss = tsg_ins_cent_loss.batch_chamfer_distance_loss(offset_1, input_coords, gt_seg_label_1)
        #chamf_2_loss = tsg_ins_cent_loss.batch_chamfer_distance_loss(offset_2, input_coords, gt_seg_label_1)
        return {
            #TODO
            "tooth_class_loss_1": (tooth_class_loss_1, 1),
            #"tooth_class_loss_2": (tooth_class_loss_2, 1),
            "offset_1_loss": (offset_1_loss, 0.03),
            "offset_1_dir_loss": (offset_1_dir_loss, 0.03),
            #"offset_2_loss": (offset_2_loss, 0.02),
            #"offset_2_dir_loss": (offset_2_dir_loss, 0.03),
            "chamf_1_loss" : (chamf_1_loss, 0.15),
            #"chamf_2_loss": (chamf_2_loss, 0.15),
        }

    def step(self, batch_idx, batch_item, phase):
        self._set_model(phase)

        points = batch_item["feat"].cuda()
        l0_xyz = batch_item["feat"][:,:3,:].cuda()
        
        #centroids = batch_item[1].cuda()
        seg_label = batch_item["gt_seg_label"].cuda()
        
        inputs = [points, seg_label]
        
        if phase == "train":
            output = self.model(inputs)
        else:
            with torch.no_grad():
                output = self.model(inputs)
        loss_meter = LossMap()
        
        loss_meter.add_loss_by_dict(self.get_loss(
            output["offset_1"], 
            output["sem_1"], 
            output["offset_2"], 
            output["sem_2"], 
            seg_label, 
            l0_xyz, 
            )
        )
        
        if phase == "train":
            loss_meter.add_loss("cbl_loss_1", output["cbl_loss_1"].sum(), 1)
            #TODO
            #loss_meter.add_loss("cbl_loss_2", output["cbl_loss_2"].sum(), 1)
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


        org_xyz_cpu = gu.torch_to_numpy(points)[0,:3,:].T
        
        results = {}

        whole_pd_sem_1 = gu.torch_to_numpy(output["sem_1"])[0,:,:].T
        whole_cls_1 = np.argmax(whole_pd_sem_1, axis=1)
        full_labeled_points_1 = np.concatenate([org_xyz_cpu, whole_cls_1.reshape(-1,1)], axis=1)

        results["sem_1"] = {}
        results["sem_1"]["full_labeled_points"] = full_labeled_points_1
        results["sem_1"]["whole_pd_sem"] = whole_pd_sem_1

        results["offset_1"] = {}
        results["offset_1"]["moved_points"] = org_xyz_cpu + gu.torch_to_numpy(output["offset_1"])[0,:3,:].T


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

        #gu.print_3d(gu.np_to_pcd_with_label(np.concatenate([gt_masked_moved_points, gt_seg_label.reshape(-1,1)[bin_gt_mask==1, :]], axis=1)))
        """
        whole_pd_sem_2 = gu.torch_to_numpy(output["sem_2"])[0,:,:].T
        whole_cls_2 = np.argmax(whole_pd_sem_2, axis=1)
        full_labeled_points_2 = np.concatenate([org_xyz_cpu, whole_cls_2.reshape(-1,1)], axis=1)

        results["sem_2"] = {}
        results["sem_2"]["full_labeled_points"] = full_labeled_points_2
        results["sem_2"]["whole_pd_sem"] = whole_pd_sem_2

        #TODO
        results["offset_2"] = {}
        results["offset_2"]["moved_points"] = org_xyz_cpu + gu.torch_to_numpy(output["offset_2"])[0,:3,:].T

        bin_gt_mask = gt_seg_label.copy().reshape(-1)
        bin_gt_mask[bin_gt_mask>=0] = 1
        bin_gt_mask[bin_gt_mask==-1] = 0
        gt_masked_moved_points = results["offset_2"]["moved_points"][bin_gt_mask==1, :]
        results["offset_2"]["gt_masked_moved_points"] = gt_masked_moved_points
        """
        
        bin_gt_mask = gt_seg_label.copy().reshape(-1)
        bin_gt_mask[bin_gt_mask>=0] = 1
        bin_gt_mask[bin_gt_mask==-1] = 0

        if True:
            mask_acc = mu.cal_metric(bin_gt_mask, full_labeled_points_1[:,3], full_labeled_points_1[:,3], is_half=True)[1]
            self.mask_acc += mask_acc
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

            gu.print_3d(gu.np_to_pcd_with_label(results["sem_1_ins"]["full_ins_labeled_points"]))
            
            gu.print_3d(gu.np_to_pcd_with_label(results["sem_1"]["full_labeled_points"]))
            gin_infer_miss_points = results["sem_1"]["full_labeled_points"].copy()
            gin_infer_miss_points_mask = (results["sem_1"]["full_labeled_points"][:,3] >= 1) != (gt_seg_label.reshape(-1) >= 0)
            gin_infer_miss_points[gin_infer_miss_points_mask, 3] = 1
            gin_infer_miss_points[np.invert(gin_infer_miss_points_mask), 3] = 0
            gu.print_3d(gu.np_to_pcd_with_label(gin_infer_miss_points))
        return
        #mu.save_result_mesh_with_label_chl(full_stl_path, full_labeled_points, "results/0622_tf_cbl_offset_only_coord_mask_ins", True)
        #full_stl_path = options["get_mesh_path"](os.path.basename(batch_item[-1][0]))

