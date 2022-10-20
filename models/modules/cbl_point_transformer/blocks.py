import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from external_libs.pointops.functions import pointops
from .utils import *
from .basic_operators import *
from .basic_operators import _eps, _inf

from collections import defaultdict


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        # torch batch-norm should use (n, c, nsample)
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)  # v * A
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, pxo):
        # o - num of points in each batch - b batch together - using BxN, used for finding the points in each example
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            # calc the reduced size of points for the batch - n = [BxN], with each b clouds, i-th cloud containing N = b[i] points
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
            #import gen_utils as gu
            #gu.print_3d(gu.torch_to_numpy(p), gu.np_to_pcd(gu.torch_to_numpy(n_p)+0.001, color=[0,1,0]))
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                # concat each with per-cloud mean => finally x = mlp[ x, mlp[mean(x)] ]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            # upsample pxo2 to pxo1
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))  # mlp
        x = self.relu(self.bn2(self.transformer2([p, x, o])))  # - seems like trans/convolute - bn - relu
        x = self.bn3(self.linear3(x))  # linear (+bn)
        x += identity
        x = self.relu(x)  # relu(x+shortcut)
        return [p, x, o]

class PtTransBlock(nn.Module):
    # wrapper of PointTransformerBlock

    def __init__(self, stage_n, stage_i, block_cfg, config):
        super(PtTransBlock, self).__init__()
        self.config = config
        self.block_cfg = block_cfg
        self.stage = (stage_n, stage_i)
        kwargs = dict(
            in_planes=config.planes[stage_i],
            planes=config.planes[stage_i],
            share_planes=config.share_planes if config.share_planes else 8,
            nsample=config.nsample[stage_i],
        )
        if block_cfg.kwargs:
            kwargs.update(block_cfg.kwargs)

        self.blk = PointTransformerBlock(**kwargs)
    def forward(self, pxo, stage_n, stage_i, stage_list, inputs):
        return self.blk(pxo)


class MLP(nn.Module):
    """ mlp(s) to generate from f_out to the desired fkey (latent/logits)
    """
    fkey_to_dims = None
    def __init__(self, fdim, head_cfg, config, fkey, drop=None):
        super().__init__()
        infer_list = []
        fkey = get_ftype(fkey)[0]
        valid_fkey = {
            'latent': config.base_fdim,
            'logits': config.num_classes,
        }
        assert fkey in valid_fkey
        if MLP.fkey_to_dims is None:
            MLP.fkey_to_dims = valid_fkey

        if fkey in ['latent', 'logits']:
            d_out = valid_fkey['latent']
            if 'latent_ops' in head_cfg and head_cfg.latent_ops:
                ops = [MLPbyOps(head_cfg.latent_ops, fdim, d_out)]
            else:
                ops = [nn.Linear(fdim, d_out), nn.BatchNorm1d(d_out), nn.ReLU(inplace=True)]
            infer_list += ops
            fdim = d_out

        if fkey in ['logits']:
            d_out = valid_fkey['logits']
            if 'logits_ops' in head_cfg and head_cfg.logits_ops:
                ops = [MLPbyOps(head_cfg.logits_ops, fdim, d_out)]
            else:
                ops = [nn.Linear(fdim, d_out)]
            infer_list += ops
        self.infer = nn.Sequential(*infer_list)

    def forward(self, stage, k):
        return self.infer(stage[k])

class MLPbyOps(nn.Module):
    @property
    def mlp_kwargs(self):
        return {
            'activation': 'relu',
            'bias': True,
            'bn': True,
            'linear_bn': False,
        }
    def __init__(self, ops, fdim, d_mid=None, d_out=None, **kwargs):
        super().__init__()
        ops_seq = ops.split('-') if '-' in ops else [ops]
        d_mid = d_mid if d_mid else fdim
        d_out = d_out if d_out else d_mid

        ops_list = []
        for ops in ops_seq:
            assert 'mlp' in ops or ops in ['linear', 'linearbn'], f'invalid ops = {ops}'
            mlp_kwargs = self.mlp_kwargs
            mlp_kwargs.update(kwargs)

            num_mlp = re.search('\d+', ops)
            num_mlp = int(num_mlp.group()) if num_mlp else 1
            linear = 'linear' in ops or not ops.endswith('mlp')  # linear / linearbn / mlp2 to ends with linear

            def get_mlp(ops_list, din, dout, mlp_kwargs):
                ops_list += [nn.Linear(din, dout, bias=mlp_kwargs['bias'])]
                if mlp_kwargs['bn']:
                    ops_list += [nn.BatchNorm1d(dout)]
                if mlp_kwargs['activation'] == 'relu':
                    ops_list += [nn.ReLU(inplace=True)]
                elif mlp_kwargs['activation'] == '':
                    pass
                else:
                    raise ValueError(f'not support activation = ' + mlp_kwargs['activation'])
                return ops_list

            for i in range(num_mlp - 1):
                ops_list = get_mlp(ops_list, din=fdim, dout=d_mid, mlp_kwargs=mlp_kwargs)
                fdim = d_mid

            if linear:
                mlp_kwargs['activation'] = ''
                mlp_kwargs['bn'] = False
            cur_out = d_out if ops == ops_seq[-1] else d_mid
            ops_list = get_mlp(ops_list, din=fdim, dout=cur_out, mlp_kwargs=mlp_kwargs)
            fdim = cur_out
            if mlp_kwargs['linear_bn'] or 'linearbn' in ops:
                ops_list += [nn.BatchNorm1d(fdim)]

        self.ops_func = nn.Sequential(*ops_list)

    def forward(self, features):
        return self.ops_func(features)


class MLPBlock(nn.Module):
    # block-level interface of MLP / MLPbyOps

    def __init__(self, stage_n, stage_i, block_cfg, config, **kwargs):
        super(MLPBlock, self).__init__()
        self.config = config
        self.block_cfg = block_cfg
        mlp_kwargs = {'fdim': config.nsample[stage_i]}
        if block_cfg.kwargs:
            mlp_kwargs.update(block_cfg.kwargs)
        mlp_kwargs.update(**kwargs)
        self.mlp_kwargs = mlp_kwargs
        self.mlp = MLPbyOps(block_cfg.ops, **mlp_kwargs)

    def forward(self, pxo, stage_n, stage_i, stage_list, inputs):
        x = self.mlp(pxo[1])
        pxo[1] = x
        return pxo

