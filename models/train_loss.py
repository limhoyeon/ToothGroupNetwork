from cProfile import label
import torch
import sys
sys.path.append("./")
from external_libs.pointnet2_utils.pointnet2_utils import square_distance
DEBUG_NAN = True
def batch_center_offset_loss(pred_offset, sample_xyz, gt_seg_label):
    """offset loss

    Args:
        pred_offset (B, 3, 16000): _description_
        sample_xyz (B, 3, 16000): _description_
        gt_seg_label (B, 1, 16000): _description_
    """
    B, _, N = pred_offset.shape
    
    pred_offset = pred_offset.permute(0,2,1) #pred_distance: B, 16000, 1
    sample_xyz = sample_xyz.permute(0,2,1) #pred_offset = B, 16000, 3
    gt_seg_label = gt_seg_label.permute(0,2,1) #sample_xyz: B, 16000, 3
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[:2]) #seg_label: B, 16000
    
    centroid_losses = 0
    dir_losses = 0
    centroid_count = 0
    dir_count = 0
    for batch_idx in range(B):
        for tooth_num in range(0, 16):
            cls_cond = gt_seg_label[batch_idx, :] == tooth_num # 일치한 tooth point 개수,

            cls_sample_xyz = sample_xyz[batch_idx, cls_cond, :] # 일치한 tooth point 개수, 3 
            if cls_sample_xyz.shape[0] < 5:
                continue
            centroid_count += 1
            cls_sample_xyz = cls_sample_xyz.view(1, *cls_sample_xyz.shape) # 1, 일치한 tooth point 개수, 3

            cls_offset = pred_offset[batch_idx, cls_cond, :]
            cls_offset = cls_offset.view(1, *cls_offset.shape) #cls_offset: 1, 일치한 cls points num, 3

            centroid = torch.mean(cls_sample_xyz, dim=1).view(1, 1, 3)
            cls_moved_xyz = torch.add(cls_sample_xyz, cls_offset) #cls_sample_xyz: 1, 일치한 cls points num, 3
            moved_dists = square_distance(cls_moved_xyz, centroid) #moved_dists: 1, 일치한 cls points num, 1
            centroid_losses += torch.div(torch.sum(moved_dists), cls_sample_xyz.shape[1])

            cls_offset_norm = torch.norm(cls_offset, dim=2).view(1,-1,1)
            cls_offset_dir = torch.div(cls_offset, cls_offset_norm)

            points_to_center_dir =  centroid - cls_sample_xyz
            points_to_center_dir_norm = torch.norm(points_to_center_dir, dim=2).view(1,-1,1)
            points_to_center_dir = torch.div(points_to_center_dir, points_to_center_dir_norm)
            
            cls_offset_dir = cls_offset_dir[cls_offset_norm.view(1,-1)>0.0002]
            points_to_center_dir = points_to_center_dir[cls_offset_norm.view(1,-1)>0.0002]
            if cls_offset_dir.shape[0] != 0:
                dir_count += 1
                dot_mat = torch.sum(points_to_center_dir * cls_offset_dir, dim=1)
                dot_mat -= 1
                dot_mat = dot_mat * dot_mat
                
                dir_losses += torch.div(torch.sum(dot_mat), cls_offset_dir.shape[0])
    centroid_losses = torch.div(centroid_losses, centroid_count)
    dir_losses = torch.div(dir_losses, dir_count)
    return centroid_losses, dir_losses

def weighted_batch_center_offset_loss(pred_offset_1, pred_offset_2, sample_xyz, gt_seg_label):
    """offset loss

    Args:
        pred_offset (B, 3, 16000): _description_
        sample_xyz (B, 3, 16000): _description_
        gt_seg_label (B, 1, 16000): _description_
    """
    B, _, N = pred_offset_2.shape
    
    pred_offset_1 = pred_offset_1.permute(0,2,1)
    pred_offset_2 = pred_offset_2.permute(0,2,1) #pred_distance: B, 16000, 1
    sample_xyz = sample_xyz.permute(0,2,1) #pred_offset = B, 16000, 3
    gt_seg_label = gt_seg_label.permute(0,2,1) #sample_xyz: B, 16000, 3
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[:2]) #seg_label: B, 16000
    
    centroid_losses = 0
    dir_losses = 0
    centroid_count = 0
    dir_count = 0
    for batch_idx in range(B):
        for tooth_num in range(0, 16):
            cls_cond = gt_seg_label[batch_idx, :] == tooth_num # 일치한 tooth point 개수,

            cls_sample_xyz = sample_xyz[batch_idx, cls_cond, :] # 일치한 tooth point 개수, 3 
            if cls_sample_xyz.shape[0] < 5:
                continue
            centroid_count += 1
            cls_sample_xyz = cls_sample_xyz.view(1, *cls_sample_xyz.shape) # 1, 일치한 tooth point 개수, 3

            cls_1_offset = pred_offset_1[batch_idx, cls_cond, :]
            cls_1_offset = cls_1_offset.view(1, *cls_1_offset.shape) #cls_offset: 1, 일치한 cls points num, 3

            cls_2_offset = pred_offset_2[batch_idx, cls_cond, :]
            cls_2_offset = cls_2_offset.view(1, *cls_2_offset.shape) #cls_offset: 1, 일치한 cls points num, 3


            centroid = torch.mean(cls_sample_xyz, dim=1).view(1, 1, 3)

            cls1_moved_xyz = torch.add(cls_sample_xyz, cls_1_offset)
            moved_dists_1 = torch.sqrt(square_distance(cls1_moved_xyz, centroid)+1e-5)
            weight_1 = moved_dists_1.clone().detach()

            thr=0.1 if tooth_num in [3,4,5,6,7, 11,12,13,14,15] else 0.075
            weight_1[moved_dists_1>=thr] = (weight_1[weight_1>=thr]*10-thr*10)*2 + 1
            weight_1[weight_1>2] = 2
            weight_1[moved_dists_1<thr] = 1

            cls2_moved_xyz = torch.add(cls_sample_xyz, cls_2_offset) #cls_sample_xyz: 1, 일치한 cls points num, 3
            moved_dists_2 = square_distance(cls2_moved_xyz, centroid) #moved_dists: 1, 일치한 cls points num, 1
            centroid_losses += torch.div(torch.sum(moved_dists_2 * weight_1), cls_sample_xyz.shape[1])

            cls_offset_norm = torch.norm(cls_2_offset, dim=2).view(1,-1,1)
            cls_offset_dir = torch.div(cls_2_offset, cls_offset_norm)

            points_to_center_dir =  centroid - cls_sample_xyz
            points_to_center_dir_norm = torch.norm(points_to_center_dir, dim=2).view(1,-1,1)
            points_to_center_dir = torch.div(points_to_center_dir, points_to_center_dir_norm)

            cls_offset_dir = cls_offset_dir[cls_offset_norm.view(1,-1)>0.0002]
            points_to_center_dir = points_to_center_dir[cls_offset_norm.view(1,-1)>0.0002]
            if cls_offset_dir.shape[0] != 0:
                dir_count += 1
                dot_mat = torch.sum(points_to_center_dir * cls_offset_dir, dim=1)
                dot_mat -= 1
                dot_mat = dot_mat * dot_mat
                
                dir_losses += torch.div(torch.sum(dot_mat), cls_offset_dir.shape[0])
    if torch.isnan(dir_losses).any() or torch.isnan(centroid_losses).any():
        print(1)
    centroid_losses = torch.div(centroid_losses, centroid_count)
    dir_losses = torch.div(dir_losses, dir_count)
    return centroid_losses, dir_losses

def distance_loss(pred_offset, sample_xyz, gt_seg_label):
    #pred_distance B, 1, 16000
    #sample_xyz B, 3, 16000(sampling num)
    #seg_label B, 1, 16000(sampling num) #label 상태는 12345678까지만 되어있음.
    #centroid B, 3, 16 -> 지금 이 centroid에 순서가 있는가,,? 없다.

    #pred_distance = pred_distance.permute(0,2,1)
    pred_offset = pred_offset.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    gt_seg_label = gt_seg_label.permute(0,2,1)
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[:2])
    #pred_distance: B, 16000, 1
    #pred_offset = B, 16000, 3
    #sample_xyz: B, 16000, 3
    #seg_label: B, 16000, 1
    
    dir_losses = 0
    centroid_losses = 0
    dist_losses = 0
    for i in range(0, 16):
        cls_cond = gt_seg_label==i
        if cls_cond.shape[0]<5:
            continue
        #cls_pred_dist = pred_distance[cls_cond]
        #if cls_pred_dist.shape[0]<5:
        #    continue
        #cls_pred_dist = cls_pred_dist.view(1, *cls_pred_dist.shape) #cls_pred_dist: 1, 일치한 cls points num, 1

        cls_sample_xyz = sample_xyz[cls_cond]
        cls_sample_xyz = cls_sample_xyz.view(1, *cls_sample_xyz.shape) #cls_sample_xyz: 1, 일치한 cls points num, 3
        cls_offset = pred_offset[cls_cond]
        cls_offset = cls_offset.view(1, *cls_offset.shape) #cls_offset: 1, 일치한 cls points num, 3

        centroid = torch.mean(cls_sample_xyz, dim=1).view(1, 1, 3)
        cls_moved_xyz = torch.add(cls_sample_xyz, cls_offset) #cls_sample_xyz: 1, 일치한 cls points num, 3
        moved_dists = square_distance(cls_moved_xyz, centroid) #moved_dists: 1, 일치한 cls points num, 1
        #moved_dists = torch.sqrt(moved_dists)
        centroid_losses += torch.sum(moved_dists)

        cls_offset_norm = torch.norm(cls_offset, dim=2).view(1,-1,1)
        cls_offset_dir = torch.div(cls_offset, cls_offset_norm)

        points_to_center_dir =  centroid - cls_sample_xyz
        points_to_center_dir_norm = torch.norm(points_to_center_dir, dim=2).view(1,-1,1)
        points_to_center_dir = torch.div(points_to_center_dir, points_to_center_dir_norm)

        cls_offset_dir = cls_offset_dir[cls_offset_norm.view(1,-1)>0.0002]
        points_to_center_dir = points_to_center_dir[cls_offset_norm.view(1,-1)>0.0002]

            
        if cls_offset_dir.shape[0] != 0:
            dot_mat = torch.sum(points_to_center_dir * cls_offset_dir, dim=1)
            dot_mat -= 1
            dot_mat = dot_mat * dot_mat
            
            dir_losses += torch.sum(dot_mat)

        #if torch.isnan(dir_losses).any() or torch.isnan(centroid_losses).any():
        #    print(1)
        #cent - to - moved points
        #dists = square_distance(cls_moved_xyz, centroid)
        #dists: 1, 일치한 cls points num, 1

        #dists = torch.sqrt(dists)
        #dist_losses += torch.nn.functional.smooth_l1_loss(cls_pred_dist, dists, reduction='sum') 
    print(centroid_losses)
    print(dir_losses)
    return centroid_losses + dir_losses * 0.1

def distance_loss_with_gin(pred_offset, sample_xyz, gt_seg_label):
    #pred_distance B, 1, 16000
    #sample_xyz B, 3, 16000(sampling num)
    #seg_label B, 1, 16000(sampling num) #label 상태는 12345678까지만 되어있음.
    #centroid B, 3, 16 -> 지금 이 centroid에 순서가 있는가,,? 없다.

    #pred_distance = pred_distance.permute(0,2,1)
    pred_offset = pred_offset.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    gt_seg_label = gt_seg_label.permute(0,2,1)
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[:2])
    #pred_distance: B, 16000, 1
    #pred_offset = B, 16000, 3
    #sample_xyz: B, 16000, 3
    #seg_label: B, 16000, 1
    
    dir_losses = 0
    centroid_losses = 0
    dist_losses = 0
    
    #processing for gingiva
    cls_cond = gt_seg_label==-1
    cls_sample_xyz = sample_xyz[cls_cond]
    cls_sample_xyz = cls_sample_xyz.view(1, *cls_sample_xyz.shape)
    cls_offset = pred_offset[cls_cond]
    cls_offset = cls_offset.view(1, *cls_offset.shape)
    centroid_losses += torch.sum(cls_offset**2)*0.01

    for i in range(0, 16):
        cls_cond = gt_seg_label==i
        #cls_pred_dist = pred_distance[cls_cond]
        #if cls_pred_dist.shape[0]<5:
        #    continue
        #cls_pred_dist = cls_pred_dist.view(1, *cls_pred_dist.shape)

        #cls_pred_dist: 1, 일치한 cls points num, 1
        cls_sample_xyz = sample_xyz[cls_cond]
        cls_sample_xyz = cls_sample_xyz.view(1, *cls_sample_xyz.shape)
        #cls_sample_xyz: 1, 일치한 cls points num, 3
        cls_offset = pred_offset[cls_cond]
        cls_offset = cls_offset.view(1, *cls_offset.shape)
        #cls_offset: 1, 일치한 cls points num, 3

        centroid = torch.mean(cls_sample_xyz, dim=1).view(1, 1, 3)
        cls_moved_xyz = torch.add(cls_sample_xyz, cls_offset)
        #cls_sample_xyz: 1, 일치한 cls points num, 3
        moved_dists = square_distance(cls_moved_xyz, centroid)
       # moved_dists = torch.sqrt(moved_dists)
        #moved_dists: 1, 일치한 cls points num, 1
        centroid_losses += torch.sum(moved_dists)

        cls_offset_norm = torch.norm(cls_offset, dim=2).view(1,-1,1)
        cls_offset_dir = torch.div(cls_offset, cls_offset_norm)

        points_to_center_dir =  centroid - cls_sample_xyz
        points_to_center_dir_norm = torch.norm(points_to_center_dir, dim=2).view(1,-1,1)
        points_to_center_dir = torch.div(points_to_center_dir, points_to_center_dir_norm)

        cls_offset_dir = cls_offset_dir[cls_offset_norm.view(1,-1)>0.0002]
        points_to_center_dir = points_to_center_dir[cls_offset_norm.view(1,-1)>0.0002]

            
        if cls_offset_dir.shape[0] != 0:
            dot_mat = torch.sum(points_to_center_dir * cls_offset_dir, dim=1)
            dot_mat -= 1
            dot_mat = dot_mat * dot_mat
            
            #dir_losses += torch.sum(dot_mat)

        #if torch.isnan(dir_losses).any() or torch.isnan(centroid_losses).any():
            #print(1)
        #cent - to - moved points
        #dists = square_distance(cls_moved_xyz, centroid)
        #dists: 1, 일치한 cls points num, 1

        #dists = torch.sqrt(dists)
        #dist_losses += torch.nn.functional.smooth_l1_loss(cls_pred_dist, dists, reduction='sum') 
    print(centroid_losses)
    print(dir_losses)
    return centroid_losses + dir_losses * 0.1

def second_distance_loss(pred_distance, pred_offset, sample_xyz, gt_seg_label):
    #pred_distance B, 1, 16000
    #sample_xyz B, 3, 16000(sampling num)
    #seg_label B, 1, 16000(sampling num) #label 상태는 12345678까지만 되어있음.
    #centroid B, 3, 16 -> 지금 이 centroid에 순서가 있는가,,? 없다.

    pred_distance = pred_distance.permute(0,2,1)
    pred_offset = pred_offset.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    gt_seg_label = gt_seg_label.permute(0,2,1)
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[:2])
    #pred_distance: B, 16000, 1
    #pred_offset = B, 16000, 3
    #sample_xyz: B, 16000, 3
    #seg_label: B, 16000, 1
    
    dir_losses = 0
    centroid_losses = 0
    dist_losses = 0
    for i in range(0, 16):
        cls_cond = gt_seg_label==i
        cls_pred_dist = pred_distance[cls_cond]
        if cls_pred_dist.shape[0]==0:
            continue
        cls_pred_dist = cls_pred_dist.view(1, *cls_pred_dist.shape)
        #cls_pred_dist: 1, 일치한 cls points num, 1
        cls_sample_xyz = sample_xyz[cls_cond]
        cls_sample_xyz = cls_sample_xyz.view(1, *cls_sample_xyz.shape)
        #cls_sample_xyz: 1, 일치한 cls points num, 3
        cls_offset = pred_offset[cls_cond]
        cls_offset = cls_offset.view(1, *cls_offset.shape)
        #cls_offset: 1, 일치한 cls points num, 3

        centroid = torch.mean(cls_sample_xyz, dim=1).view(1, 1, 3)
        cls_moved_xyz = torch.add(cls_sample_xyz, cls_offset)
        #cls_sample_xyz: 1, 일치한 cls points num, 3
        moved_dists = square_distance(cls_moved_xyz, centroid)
       # moved_dists = torch.sqrt(moved_dists)
        #moved_dists: 1, 일치한 cls points num, 1
        centroid_losses += torch.sum(moved_dists)
        
        cls_offset_dir = torch.div(cls_offset, torch.norm(cls_offset))
        
        points_to_center_dir =  centroid - cls_sample_xyz
        points_to_center_dir = torch.div(points_to_center_dir, torch.norm(points_to_center_dir))

        dir_losses += -(torch.dot(cls_offset_dir.view(-1), points_to_center_dir.view(-1)))

        #cent - to - moved points
        #dists = square_distance(cls_moved_xyz, centroid)
        #dists: 1, 일치한 cls points num, 1

        #dists = torch.sqrt(dists)
        #dist_losses += torch.nn.functional.smooth_l1_loss(cls_pred_dist, dists, reduction='sum') 
    #print(centroid_losses)
    #print(dir_losses)
    return centroid_losses + dir_losses

def batch_chamfer_distance_loss(pred_offset, sample_xyz, gt_seg_label):
    """offset loss

    Args:
        pred_offset (B, 3, 16000): _description_
        sample_xyz (B, 3, 16000): _description_
        gt_seg_label (B, 1, 16000): _description_
    """
    B, _, N = pred_offset.shape
    
    pred_offset = pred_offset.permute(0,2,1) #pred_distance: B, 16000, 1
    sample_xyz = sample_xyz.permute(0,2,1) #pred_offset = B, 16000, 3
    gt_seg_label = gt_seg_label.permute(0,2,1) #sample_xyz: B, 16000, 3
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[:2]) #seg_label: B, 16000
    
    centroids = []
    for batch_idx in range(B):
        b_centroids = []
        for tooth_num in range(0, 16):
            cls_cond = gt_seg_label[batch_idx, :] == tooth_num # 일치한 tooth point 개수,

            cls_sample_xyz = sample_xyz[batch_idx, cls_cond, :] # 일치한 tooth point 개수, 3 
            if cls_sample_xyz.shape[0] < 5:
                continue
            centroid = torch.mean(cls_sample_xyz, dim=0).view(3)
            b_centroids.append(centroid)
        b_centroids = torch.stack(b_centroids)
        centroids.append(b_centroids)
    loss = 0
    for batch_idx in range(B):
        moved_points = sample_xyz[batch_idx, :] + pred_offset[batch_idx, :] # N, 3
        moved_points = moved_points[gt_seg_label[batch_idx]!=-1, :] #fg points N, 3
        b_centroids = centroids[batch_idx] # centroid 개수, 3
        pred_ct_dists = square_distance(moved_points.unsqueeze(dim=0), b_centroids.unsqueeze(dim=0))
        sorted_pred_ct_dists, _ = pred_ct_dists.sort(dim=-1)
        min_pred_ct_dists = sorted_pred_ct_dists[:, :, :2]
        ratio = torch.div(min_pred_ct_dists[:,:,0], min_pred_ct_dists[:,:,1])
        loss += torch.sum(ratio)/moved_points.shape[0]
    loss /= B
    return loss

    pred_offset = pred_offset.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    centroid = centroid.permute(0,2,1)

    pred_centroid = torch.add(pred_offset, sample_xyz)

    #source를 pred centroid로
    pred_ct_dists = square_distance(pred_centroid, centroid)
    sorted_pred_ct_dists, _ = pred_ct_dists.sort(dim=-1)
    min_pred_ct_dists = sorted_pred_ct_dists[:, :, :2]

    pred_ct_mask = min_pred_ct_dists[:,:,0].le(0.2)
    
    ratio = torch.div(min_pred_ct_dists[:,:,0], min_pred_ct_dists[:,:,1])
    ratio = torch.masked_select(ratio, pred_ct_mask)
    
    #loss = torch.div(torch.sum(ratio), torch.count_nonzero(pred_ct_mask))
    loss = torch.sum(ratio)
    return loss

def chamfer_distance_with_gin_loss(pred_offset, sample_xyz, centroid):
    pred_offset = pred_offset.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    centroid = centroid.permute(0,2,1)

    pred_centroid = torch.add(sample_xyz, pred_offset)

    #source를 pred centroid로
    pred_ct_dists = square_distance(pred_centroid, centroid)
    sorted_pred_ct_dists, _ = pred_ct_dists.sort(dim=-1)
    min_pred_ct_dists = sorted_pred_ct_dists[:, :, :2]

    
    #pred_ct_mask = min_pred_ct_dists[:,:,1].le(0.2)
    
    ratio = torch.div(min_pred_ct_dists[:,:,0], min_pred_ct_dists[:,:,1])
    #ratio = torch.masked_select(ratio, pred_ct_mask)
    
    #loss = torch.div(torch.sum(ratio), torch.count_nonzero(pred_ct_mask))
    loss = torch.sum(ratio)
    return loss


import torch.nn.functional as F

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        target = F.one_hot(target, num_classes=10)
        x = x.permute(0,2,1)
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    def forward(self, pred, target):
        #target = F.one_hot(target, num_classes=10)
        #pred = pred.permute(0,2,1)
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def tooth_class_loss(cls_pred, gt_cls, weight=None, label_smoothing=None):
    """
    Input
        cls_pred: 1, 17, 16000
        gt_cls: 1, 1, 16000 -> -1 is background, 0~15 is foreground
    """
    B, _, N = gt_cls.shape
    gt_cls = gt_cls.view(B, -1)
    gt_cls = gt_cls.type(torch.long)
    gt_cls = gt_cls + 1
    if label_smoothing is None:
        if weight is None:
            loss = torch.nn.CrossEntropyLoss().type(torch.float).cuda()(cls_pred, gt_cls)
        else:
            loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight).type(torch.float).cuda())(cls_pred, gt_cls)
    else:
    #loss = LabelSmoothingCrossEntropy()(cls_pred, gt_cls, smoothing=label_smoothing)
        loss = LabelSmoothingLoss(10, smoothing=label_smoothing)(cls_pred, gt_cls)
    return loss

def uncert_tooth_class_loss(cls_pred_1, cls_pred_2, gt_cls,weight):
    """
    Input
        cls_pred_1: 1, 17, 16000
        gt_cls: 1, 1, 16000 -> -1 is background, 0~15 is foreground
    """
    B, _, N = gt_cls.shape
    gt_cls = gt_cls.view(B, -1)
    gt_cls = gt_cls.type(torch.long)
    gt_cls = gt_cls + 1
    loss_1 = torch.nn.CrossEntropyLoss(reduction='none')(cls_pred_1, gt_cls)
    


def ins_seg_loss(l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result, cls_pred, seg_label, centroids):
    #pred_distance, pred_offset, sample_xyz, gt_seg_label, centroid
    dist_loss = distance_loss(dist_result, offset_result, l0_xyz, seg_label)
    chamf_loss = (chamfer_distance_loss(offset_result, l0_xyz, centroids) * 0.01 )
    class_loss = tooth_class_loss(cls_pred, seg_label)

    print("dist_loss", dist_loss, "chamf_loss", chamf_loss, "class_loss", class_loss)
    loss = dist_loss + chamf_loss + class_loss
    return loss

def make_teeth_mask(gt_seg_label):
    #gt_seg_label B, 1, 16000(sampling num) #label 상태는 12345678까지만 되어있음.

    #gt_bin_label B, 16000(sampling num) #label 상태는 0 혹은 1
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[0],-1)
    #gt_seg_label: B, 16000
    gt_bin_label = torch.ones_like(gt_seg_label)
    gt_bin_label[gt_seg_label == -1] = 0
    return gt_bin_label

def make_teeth_mask_binary(gt_seg_label):
    #gt_seg_label B, 1, 16000(sampling num) #label 상태는 12345678까지만 되어있음.

    #gt_bin_label B, 16000(sampling num) #label 상태는 0 혹은 1
    gt_seg_label = gt_seg_label.view(gt_seg_label.shape[0],-1)
    #gt_seg_label: B, 16000
    gt_bin_label = torch.ones_like(gt_seg_label)
    gt_bin_label[gt_seg_label == -1] = 0
    gt_bin_label = gt_bin_label.type(torch.long)
    gt_bin_label = gt_bin_label.view(gt_seg_label.shape[0], -1)
    return gt_bin_label

def teeth_mask_loss(pred_mask, gt_seg_label):
    gt_bin_label = make_teeth_mask(gt_seg_label).type(torch.float32).reshape(pred_mask.shape)
    return torch.nn.BCEWithLogitsLoss()(pred_mask, gt_bin_label)
    
def tf_onestep_ins_loss(l0_xyz, pred_offset, pred_sem, seg_label, dist_loss_weight):
    dist_loss = distance_loss(pred_offset, l0_xyz, seg_label)
    #chamf_loss = (chamfer_distance_loss(pred_offset, l0_xyz, centroids) * 0.01 )
    class_loss = teeth_mask_loss(pred_sem, seg_label)
    #weight = torch.ones(17).cuda()
    #weight[0] = 0.1
    #pred_sem = torch.nn.functional.softmax(pred_sem, dim=1)
    #class_loss = tooth_class_loss(pred_sem, seg_label, weight=weight)

    print("dist_loss", dist_loss, "class_loss", class_loss)
    #print(dist_loss)
    loss = dist_loss* dist_loss_weight  + class_loss
    return loss

def tf_ins_loss(l0_xyz, pred_offset, pred_sem, seg_label, centroids):
    dist_loss = distance_loss(pred_offset, l0_xyz, seg_label)
    chamf_loss = (chamfer_distance_loss(pred_offset, l0_xyz, centroids) * 0.01 )
    class_loss = tooth_class_loss(pred_sem, seg_label)

    print("dist_loss", dist_loss, "chamf_loss", chamf_loss, "class_loss", class_loss)
    loss = dist_loss + chamf_loss + class_loss
    return loss

def ins_seg_double_loss(l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result, offset_result_d, dist_result_d, cls_pred, seg_label, centroids):
    #pred_distance, pred_offset, sample_xyz, gt_seg_label, centroid
    dist_loss = distance_loss(dist_result, offset_result, l0_xyz, seg_label)
    dist_loss_2 = distance_loss(dist_result_d, offset_result_d, offset_result + l0_xyz, seg_label)
    chamf_loss = (chamfer_distance_loss(offset_result, l0_xyz, centroids) * 0.01 )
    class_loss = tooth_class_loss(cls_pred, seg_label)

    print("dist_loss", dist_loss, "chamf_loss", chamf_loss, "class_loss", class_loss)
    loss = dist_loss + chamf_loss + class_loss + dist_loss_2*0.2
    return loss

def weighted_cls1_loss(pred_weight_1, pred_cls_1, gt_seg_label, weight):
    #pred_weight B, 1, 16000
    #pred_cls_1 B, 10, 16000(sampling num)
    #gt_seg_label B, 1, 16000(sampling num) #label 상태는 12345678까지만 되어있음.

    B, _, N = gt_seg_label.shape
    gt_seg_label = gt_seg_label.view(B, -1)
    gt_seg_label = gt_seg_label.type(torch.long)
    gt_seg_label = gt_seg_label + 1
    bce_1 = torch.nn.CrossEntropyLoss(weight=weight, reduction="none")(pred_cls_1, gt_seg_label)

    pred_weight_1 = torch.sigmoid(pred_weight_1).view(pred_weight_1.shape[0],-1)
    #pred_weight_1: B, 16000
    loss = (bce_1 * pred_weight_1) ** 2 + (1-pred_weight_1)**2
    loss = torch.sum(loss)/loss.shape[1]
    return loss

def weighted_cls2_loss(pred_weight_1, pred_cls_2, gt_seg_label, weight):
    #pred_weight B, 1, 16000
    #pred_cls_1 B, 10, 16000(sampling num)
    #gt_seg_label B, 1, 16000(sampling num) #label 상태는 12345678까지만 되어있음.

    B, _, N = gt_seg_label.shape
    gt_seg_label = gt_seg_label.view(B, -1)
    gt_seg_label = gt_seg_label.type(torch.long)
    gt_seg_label = gt_seg_label + 1
    bce_2 = torch.nn.CrossEntropyLoss(weight=weight, reduction="none")(pred_cls_2, gt_seg_label)

    pred_weight_1 = torch.sigmoid(pred_weight_1).view(pred_weight_1.shape[0],-1)
    loss = (2.0-pred_weight_1)*bce_2
    loss = torch.sum(loss)/loss.shape[1]
    return loss