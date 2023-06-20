import torch
from . import tgn_loss
from models.base_model import BaseModel
from loss_meter import LossMap

class FpsGroupingNetworkModel(BaseModel):
    def get_loss(self, offset_1, offset_2, sem_1, sem_2, mask_1, mask_2, gt_seg_label_1, gt_seg_label_2, input_coords, cropped_coords):
        half_seg_label = gt_seg_label_1.clone()
        half_seg_label[half_seg_label>=9] -= 8

        gt_seg_label_2[gt_seg_label_2>=0] = 0
        tooth_class_loss_1 = tgn_loss.tooth_class_loss(sem_1, half_seg_label, 9)
        tooth_class_loss_2 = tgn_loss.tooth_class_loss(sem_2, gt_seg_label_2, 2)

        offset_1_loss, offset_1_dir_loss = tgn_loss.batch_center_offset_loss(offset_1, input_coords, gt_seg_label_1)
        
        chamf_1_loss = tgn_loss.batch_chamfer_distance_loss(offset_1, input_coords, gt_seg_label_1)
        return {
            "tooth_class_loss_1": (tooth_class_loss_1, self.config["tr_set"]["loss"]["tooth_class_loss_1"]),
            "tooth_class_loss_2": (tooth_class_loss_2, self.config["tr_set"]["loss"]["tooth_class_loss_2"]),
            "offset_1_loss": (offset_1_loss, self.config["tr_set"]["loss"]["offset_1_loss"]),
            "offset_1_dir_loss": (offset_1_dir_loss, self.config["tr_set"]["loss"]["offset_1_dir_loss"]),
            "chamf_1_loss" : (chamf_1_loss, self.config["tr_set"]["loss"]["chamf_1_loss"])
        }

    def step(self, batch_idx, batch_item, phase):
        self._set_model(phase)

        points = batch_item["feat"].cuda()
        l0_xyz = batch_item["feat"][:,:3,:].cuda()
        
        seg_label = batch_item["gt_seg_label"].cuda()
        
        inputs = [points, seg_label]

        if phase == "train":
            output = self.module(inputs)
        else:
            with torch.no_grad():
                output = self.module(inputs)
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