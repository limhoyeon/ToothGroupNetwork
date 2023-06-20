import torch
from . import tgn_loss
from models.base_model import BaseModel
from loss_meter import LossMap

class TransformerModel(BaseModel):
    def get_loss(self, sem_1,  gt_seg_label_1):
        tooth_class_loss_1 = tgn_loss.tooth_class_loss(sem_1, gt_seg_label_1, 17)
        return {
            "tooth_class_loss_1": (tooth_class_loss_1, self.config["tr_set"]["loss"]["tooth_class_loss_1"]),
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
            output["sem_1"], 
            seg_label, 
            )
        )
        
        if phase == "train":
            loss_sum = loss_meter.get_sum()
            self.optimizer.zero_grad()
            loss_sum.backward()
            self.optimizer.step()

        return loss_meter