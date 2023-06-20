import torch
import numpy as np
import ops_utils as ou
import gen_utils as gu
from .cbl_point_transformer.cbl_point_transformer_module import get_model

class GroupingNetworkModule(torch.nn.Module):
    def __init__(self, config):
        self.config = config

        super().__init__()
        class_num = 9
        self.first_ins_cent_model = get_model(**config["model_parameter"], c=config["model_parameter"]["input_feat"], k=class_num + 1)
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
                fg_points_labels_ls = ou.get_clustering_labels(b_moved_points, whole_cls_1)
                temp_centroids = []
                for i in np.unique(fg_points_labels_ls):
                    temp_centroids.append(np.mean(b_fg_moved_points[fg_points_labels_ls==i, :],axis=0))
                cluster_centroids.append(temp_centroids)
        
        org_xyz_cpu = gu.torch_to_numpy(inputs[0][:, :3, :].permute(0, 2, 1))
        nn_crop_indexes = ou.get_nearest_neighbor_idx(org_xyz_cpu, cluster_centroids, self.config["model_parameter"]["crop_sample_size"])
        cropped_feature_ls = ou.get_indexed_features(inputs[0], nn_crop_indexes)
        if len(inputs)>=2:
            cluster_gt_seg_label = ou.get_indexed_features(inputs[1], nn_crop_indexes)

        cropped_feature_ls = ou.centering_object(cropped_feature_ls)

        if len(inputs) >= 2 and not test:
            cluster_gt_seg_label[cluster_gt_seg_label>=0] = 0
            outputs["cluster_gt_seg_label"] = cluster_gt_seg_label
            cbl_loss_2, sem_2, offset_2, mask_2, _ = self.second_ins_cent_model([cropped_feature_ls, cluster_gt_seg_label])

            outputs.update({
                "cbl_loss_2": cbl_loss_2,
                "sem_2": sem_2,
                "offset_2":offset_2,
                "mask_2":mask_2,
            })
        else:
            sem_2, offset_2, mask_2, _ = self.second_ins_cent_model([cropped_feature_ls])
            outputs.update({
                "sem_2": sem_2,
                "offset_2":offset_2,
                "mask_2":mask_2,
            })

        outputs["cropped_feature_ls"] = cropped_feature_ls
        outputs["nn_crop_indexes"] =  nn_crop_indexes

        return outputs