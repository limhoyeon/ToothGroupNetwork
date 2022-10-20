import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from external_libs.pointops.functions import pointops
from .utils import *
from .basic_operators import *
from .basic_operators import _eps, _inf
from .blocks import *


class MultiHead(nn.Module):
    def __init__(self, fdims, head_cfg, config, k):
        super().__init__()
        self.head_cfg = head_cfg
        self.ftype = get_ftype(head_cfg.ftype)[0]

        num_layers = config.num_layers
        infer_list = nn.ModuleList()
        ni_list = []

        for n, i in parse_stage(head_cfg.stage, num_layers):
            func = MLP(fdims[i], head_cfg, config, self.ftype)
            infer_list.append(func)
            ni_list.append((n, i))

        self.infer_list = infer_list
        self.ni_list = ni_list

        if head_cfg.combine.startswith('concat'):
            fdim = MLP.fkey_to_dims[head_cfg.ftype] * len(ni_list)
            self.comb_ops = torch.cat
        else:
            raise ValueError(f'not supported {head_cfg.combine}')
        # logits
        #k = config.num_classes
        if head_cfg.combine.endswith('mlp'):
            d = config.base_fdim
            self.cls = nn.Sequential(nn.Linear(fdim, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True), nn.Linear(d, k))
        else:
            self.cls = nn.Linear(fdim, k)

    def upsample(self, stage_n, stage_i, stage_list):
        p, x, o = fetch_pxo(stage_n, stage_i, stage_list, self.ftype)
        if stage_i == 0:
            return x

        p0, _, o0 = fetch_pxo('up', 0, stage_list, self.ftype)
        x = pointops.interpolation(p, p0, x, o, o0, k=1)
        return x

    def forward(self, stage_list):
        collect_list = []
        for (n, i), func in zip(self.ni_list, self.infer_list):
            rst = func(stage_list[n][i], 'f_out')  # process to desired fdim
            stage_list[n][i][self.ftype] = rst  # store back
            collect_list.append(self.upsample(n, i, stage_list))  # (n, c) - potentially upsampled
        x = self.comb_ops(collect_list, 1)  # combine - NCHW
        x = self.cls(x)
        return x, stage_list

class ContrastHead(nn.Module):
    """ currently used as criterion - need to be wrapped with DataParallel if params used (eg. project)
    """
    def __init__(self, head_cfg, config):
        super().__init__()
        self.nsample = torch.tensor(config.nsample)
        self.nstride = torch.tensor(config.nstride)
        self.num_classes = torch.tensor(config.num_classes)
        self.head_cfg = head_cfg
        self.config = config
        self.stages = parse_stage(head_cfg.stage, config.num_layers)
        self.vx_size = [config.voxel_size * 2 ** i for i in range(config.num_layers)]
        self.ftype = get_ftype(head_cfg.ftype)[0]
        self.dist_func = getattr(self, f'dist_{head_cfg.dist}')
        self.posmask_func = getattr(self, f'posmask_{head_cfg.pos}')
        self.contrast_func = getattr(self, f'contrast_{head_cfg.contrast}')
        assert head_cfg.sample in ['cnt', 'glb', 'sub', 'subspatial', 'pts', 'label', 'vote'], f'not support sample = {head_cfg.sample}'
        # self.sample_func = getattr(self, f'sample_{head_cfg.sample}') if 'sample' in head_cfg and head_cfg.sample else self.sample_label
        self.main_contrast = getattr(self, f'{head_cfg.main}_contrast') if 'main' in head_cfg and head_cfg.main else self.point_contrast

        self.temperature = None
        if 'temperature' in head_cfg:
            temperature = head_cfg.temperature
            self.temperature = temperature

        self.project = None
        if 'project' in head_cfg and head_cfg.project:
            self.project = nn.ModuleDict({
                f'{n}{i}': MLPbyOps(head_cfg.project, config.base_fdim * 2 ** i, d_out=config.base_fdim) for n, i in self.stages
            })

    def sample_label(self, n, i, stage_list, target):
        p, features, offset = fetch_pxo(n, i, stage_list, self.ftype)
        nsample = self.nsample[i]
        labels = get_subscene_label(n, i, stage_list, target, self.nstride, self.num_classes)  # (m, ncls) - distribution / onehot

        neighbor_idx, _ = pointops.knnquery(nsample, p, p, o, o) # (m, nsample)
        # exclude self-loop
        nsample -= 1
        neighbor_idx = neighbor_idx[..., 1:].contiguous()
        m = neighbor_idx.shape[0]

        neighbor_label = labels[neighbor_idx.view(-1).long(), :].view(m, nsample, labels.shape[1]) # (m, nsample, ncls)
        neighbor_feature = features[neighbor_idx.view(-1).long(), :].view(m, nsample, features.shape[1])
        # neighbor_label = pointops.queryandgroup(nsample, p, p, labels, neighbor_idx, o, o, use_xyz=False)  # (m, nsample, ncls)
        # neighbor_feature = pointops.queryandgroup(nsample, p, p, features, neighbor_idx, o, o, use_xyz=False)  # (m, nsample, c)

        # print('shape', p.shape, features.shape, labels.shape, 'neighbor_idx', neighbor_idx.shape, 'nsample =', nsample)
        # print('o = ', o)
        # print('shape - neighbor', neighbor_feature.shape, neighbor_label.shape)
        return nsample, labels, neighbor_label, features, neighbor_feature


    def dist_l2(self, features, neighbor_feature):
        dist = torch.unsqueeze(features, -2) - neighbor_feature
        dist = torch.sqrt(torch.sum(dist ** 2, axis=-1) + _eps) # [m, nsample]
        return dist

    def dist_kl(self, features, neighbor_feature, normalized, normalized_neighbor):
        # kl dist from featuers (gt) to neighbors (pred)
        if normalized in [False, 'softmax']:  # if still not a prob distribution - prefered
            features = F.log_softmax(features, dim=-1)
            log_target = True
        elif normalized == True:
            log_target = False
        else:
            raise ValueError(f'kl dist not support normalized = {normalized}')
        features = features.unsqueeze(-2)

        if normalized_neighbor in [False, 'softmax']:
            neighbor_feature = F.log_softmax(neighbor_feature, dim=-1)
        elif normalized_neighbor == True:
            neighbor_feature = torch.maximum(neighbor_feature, neighbor_feature.new_full([], _eps)).log()
        else:
            raise ValueError(f'kl dist not support normalized_neighbor = {normalized}')
        
        # (input, target) - i.e. (pred, gt), where input/pred should be in log space
        dist = F.kl_div(neighbor_feature, features, reduction='none', log_target=log_target)  # [m, nsample, d] - kl(pred, gt) to calculate kl = gt * [ log(gt) - log(pred) ]
        dist = dist.sum(-1)  # [m, nsample]
        return dist


    def posmask_cnt(self, labels, neighbor_label):
        labels = torch.argmax(torch.unsqueeze(labels, -2), -1)  # [m, 1]
        neighbor_label = torch.argmax(neighbor_label, -1)  # [m, nsample]
        mask = labels == neighbor_label  # [m, nsample]
        return mask

    def contrast_softnn(self, dist, posmask, invalid_mask=None):
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)
        if self.temperature is not None:
            dist = dist / self.temperature
        exp = torch.exp(dist)

        if invalid_mask is not None:
            valid_mask = 1 - invalid_mask
            exp = exp * valid_mask

        pos = torch.sum(exp * posmask, axis=-1)  # (m)
        neg = torch.sum(exp, axis=-1)  # (m)
        loss = -torch.log(pos / neg + _eps)
        return loss

    def contrast_nce(self, dist, posmask, invalid_mask=None):
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)
        if self.temperature is not None:
            dist = dist / self.temperature
        exp = torch.exp(dist)

        if invalid_mask is not None:
            valid_mask = 1 - invalid_mask
            exp = exp * valid_mask

        # each Log term an example; per-pos vs. all negs
        neg = torch.sum(exp * (1 - posmask), axis=-1)  # (m)
        under = exp + neg
        loss = (exp / (exp + neg))[posmask]  # each Log term an example
        loss = -torch.log(loss)
        return loss

    def point_contrast(self, n, i, stage_list, target):
        p, features, o = fetch_pxo(n, i, stage_list, self.ftype)
        if self.project is not None:
            features = self.project[f'{n}{i}'](features)

        nsample = self.nsample[i]
        labels = get_subscene_label(n, i, stage_list, target, self.nstride, self.config.num_classes)  # (m, ncls) - distribution / onehot
        neighbor_idx, _ = pointops.knnquery(nsample, p, p, o, o) # (m, nsample)

        # exclude self-loop
        nsample = self.nsample[i] - 1  # nsample -= 1 can only be used if nsample is py-number - results in decreasing number if is tensor, e.g. 4,3,2,1,...
        neighbor_idx = neighbor_idx[..., 1:].contiguous()
        m = neighbor_idx.shape[0]

        neighbor_label = labels[neighbor_idx.view(-1).long(), :].view(m, nsample, labels.shape[1]) # (m, nsample, ncls)

        if 'norm' in self.head_cfg.dist or self.head_cfg.dist == 'cos':
            features = F.normalize(features, dim=-1)  # p2-norm

        neighbor_feature = features[neighbor_idx.view(-1).long(), :].view(m, nsample, features.shape[1])
        # neighbor_label = pointops.queryandgroup(nsample, p, p, labels, neighbor_idx, o, o, use_xyz=False)  # (m, nsample, ncls)
        # neighbor_feature = pointops.queryandgroup(nsample, p, p, features, neighbor_idx, o, o, use_xyz=False)  # (m, nsample, c)

        # print('shape', p.shape, features.shape, labels.shape, 'neighbor_idx', neighbor_idx.shape, 'nsample =', nsample)
        # print('o = ', o)
        # print('shape - neighbor', neighbor_feature.shape, neighbor_label.shape)
        posmask = self.posmask_cnt(labels, neighbor_label)  # (m, nsample) - bool
        # select only pos-neg co-exists
        point_mask = torch.sum(posmask.int(), -1)  # (m)
        point_mask = torch.logical_and(0 < point_mask, point_mask < nsample)
        # print('mask - pos/point', posmask.shape, point_mask.shape)

        if self.head_cfg.pos != 'cnt':
            posmask = self.posmask_func(labels, neighbor_label)

        # print(point_mask.shape)
        # print(torch.any(point_mask), point_mask[:10])
        if not torch.any(point_mask):
            if i == 0:
                o = torch.cat([torch.tensor([0]).to(o.device), o])
                for bi, (start, end) in enumerate(zip(o[:-1], o[1:])):
                    print('bi / labelcnt - ', bi , ' / ', torch.unique(labels[start:end].argmax(dim=1)))
                print(point_mask.sum(0), len(point_mask))
                print(labels[0], neighbor_label[0])
                print(labels[100], neighbor_label[100])
                print(labels[900], neighbor_label[900])
                print(flush=True)
                raise
            return torch.tensor(.0)

        posmask = posmask[point_mask]
        features = features[point_mask]
        neighbor_feature = neighbor_feature[point_mask]
        # print('after masking - ', posmask.shape, features.shape, neighbor_feature.shape)

        dist = self.dist_func(features, neighbor_feature)
        loss = self.contrast_func(dist, posmask)  # (m)

        w = self.head_cfg.weight[1:]
        loss = torch.mean(loss)
        loss *= float(w)
        return loss

    def forward(self, output, target, stage_list):
        loss_list = []
        for n, i in self.stages:
            loss = self.main_contrast(n, i, stage_list, target)
            loss_list += [loss]
        return loss_list

