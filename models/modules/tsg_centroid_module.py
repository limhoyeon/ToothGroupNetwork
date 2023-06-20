import torch.nn as nn
import torch.nn.functional as F
from external_libs.pointnet2_utils.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
import torch
class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        
        input_feauture_num = 6
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.025, 0.05], [32, 64], input_feauture_num, [[32, 32], [32, 32]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.05, 0.1], [32, 64], 32+32, [[64, 128], [64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [32, 64], 128+128, [[196, 256], [196, 256]])
        

        self.fp3 = PointNetFeaturePropagation(768, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [128, 128])
        self.fp1 = PointNetFeaturePropagation(128+input_feauture_num, [64, 32])

        self.offset_conv_1 = nn.Conv1d(515,256, 1)
        self.offset_bn_1 = nn.BatchNorm1d(256)
        self.dist_conv_1 = nn.Conv1d(515,256, 1)
        self.dist_bn_1 = nn.BatchNorm1d(256)
        
        self.offset_conv_2 = nn.Conv1d(256,3, 1)
        self.dist_conv_2 = nn.Conv1d(256,1, 1)

        nn.init.zeros_(self.offset_conv_2.weight)
        nn.init.zeros_(self.dist_conv_2.weight)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        
        offset_result = F.relu(self.offset_bn_1(self.offset_conv_1(torch.cat([l3_points,l3_xyz], axis=1))))
        offset_result = self.offset_conv_2(offset_result)

        dist_result = F.relu(self.dist_bn_1(self.dist_conv_1(torch.cat([l3_points,l3_xyz], axis=1))))
        dist_result = self.dist_conv_2(dist_result)
        
        return l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import torch
    model = get_model()
    xyz = torch.rand(6, 6, 2048)
    #output is B, C, N order
    for item in model(xyz):
        print(item.shape)
    input()