import torch
from . import tsg_loss
from models.base_model import BaseModel
from loss_meter import LossMap
from external_libs.pointnet2_utils.pointnet2_utils import square_distance
import ops_utils as ou

class TSegNetModel(BaseModel):
    def __init__(self, config, module):
        super().__init__(config, module)
        if self.config.get("pretrained_centroid_model_path", None) is not None:
            self.module.load_state_dict(torch.load(self.config["pretrained_centroid_model_path"] +".h5"), strict=False)

    def get_loss(self, outputs, gt):
        losses = {}

        dist_loss, cent_loss, chamf_loss = tsg_loss.centroid_loss(
            outputs["offset_result"], outputs["l3_xyz"], outputs["dist_result"], gt["centroid_coords"]
        )
        losses.update({
            "dist_loss": (dist_loss, 1),
            "cent_loss": (cent_loss, 1),
            "chamf_loss": (chamf_loss, 0.1),
        })
        if self.config["run_tooth_segmentation_module"] is False: return losses
        sqd = square_distance(torch.from_numpy(outputs["center_points"]).cuda(),gt["centroid_coords"].permute(0,2,1))  # 1, N, 3 X 1, M, 3 => 1, N, M
        sqd_argmin =  sqd.argmin(axis=2).reshape(-1)
        pred_centerpoint_gt_label_ls = gt["centroid_labels"][:, sqd_argmin] # 1, N, M => 1, N
        
        cluster_gt_seg_bin_label_ls = torch.zeros_like(outputs["cluster_gt_seg_label"]).cuda()
        for i in range(outputs["cluster_gt_seg_label"].shape[0]):
            cluster_gt_seg_bin_label_ls[i, 0, pred_centerpoint_gt_label_ls[0][i]==outputs["cluster_gt_seg_label"][i][0]+1] = 1

        seg_1_loss, seg_2_loss, id_pred_loss = tsg_loss.segmentation_loss(outputs["pd_1"], outputs["weight_1"], outputs["pd_2"], outputs["id_pred"],
        pred_centerpoint_gt_label_ls, cluster_gt_seg_bin_label_ls)
        losses.update({
            "seg_1_loss":(seg_1_loss,1), 
            "seg_2_loss":(seg_2_loss,1), 
            "id_pred_loss":(id_pred_loss,1)
        })
        
        return losses


    def step(self, batch_idx, batch_item, phase):
        self._set_model(phase)
        B, C, N = batch_item["feat"].shape

        gt_centroid_coords, gt_centroid_exists = ou.seg_label_to_cent(batch_item["feat"][:,:3,:], batch_item["gt_seg_label"])

        gt_centroids_label = torch.range(0, 15).view(1,-1).cuda() + 1
        gt_centroid_exists = gt_centroid_exists.view(1, -1)
        gt_centroid_coords = gt_centroid_coords.permute(0,2,1)
        gt_centroid_coords = gt_centroid_coords[gt_centroid_exists>0, :]
        gt_centroids_label = gt_centroids_label[gt_centroid_exists>0]
        gt_centroid_coords = gt_centroid_coords.unsqueeze(dim=0)
        gt_centroids_label = gt_centroids_label.unsqueeze(dim=0)
        gt_centroid_coords = gt_centroid_coords.permute(0,2,1)
        gt_centroid_coords = gt_centroid_coords.cuda() # B, 3, 14
        gt_centroids_label = gt_centroids_label.cuda() # B, 14



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
            output,
            {
                "seg_label": seg_label,
                "centroid_coords": gt_centroid_coords,
                "centroid_labels": gt_centroids_label,
            } 
            )
        )
        
        if phase == "train":
            loss_sum = loss_meter.get_sum()
            self.optimizer.zero_grad()
            loss_sum.backward()
            self.optimizer.step()

        return loss_meter
    

    def infer(self, batch_idx, batch_item, **options):
        pass