import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from external_libs.pointnet2_utils.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        self.k = 17
        scale=2
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=6,scale=scale)
        self.conv1 = torch.nn.Conv1d(1088*scale, 512*scale, 1)
        self.conv2 = torch.nn.Conv1d(512*scale, 256*scale, 1)
        self.conv3 = torch.nn.Conv1d(256*scale, 128*scale, 1)
        self.conv4 = torch.nn.Conv1d(128*scale, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512*scale)
        self.bn2 = nn.BatchNorm1d(256*scale)
        self.bn3 = nn.BatchNorm1d(128*scale)

    def forward(self, x_in):
        x = x_in[0]
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k).permute(0,2,1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight = weight)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


class PointFirstModule(torch.nn.Module):
    def __init__(self, config):
        self.config = config

        super().__init__()
        self.first_sem_model = get_model()

    def forward(self, inputs, test=False):
        DEBUG=False
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1] => B, 1, 24000 : ground truth segmentation
        """
        B, C, N = inputs[0].shape
        cls_pred, feats = self.first_sem_model(inputs)
        outputs = {
            "cls_pred": cls_pred
        }
        return outputs
        
if __name__ == '__main__':
    model = get_model(13)
    xyz = torch.rand(12, 3, 2048)
    (model(xyz))