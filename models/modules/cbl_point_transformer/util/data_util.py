import numpy as np
import random
import SharedArray as SA

import torch

from util.voxelize import voxelize
from model.basic_operators import get_overlap


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collate_fn(batch):
    """
    Args:
        batch - [(xyz, feat, label, ...), ...] - each tuple from a sampler
    Returns: [
        xyz     : [BxN, 3]
        feat    : [BxN, d]
        label   : [BxN]
        ...
        offset  : int
    ]
    """
    batch_list = []
    for sample in batch:
        if isinstance(sample, list):
            batch_list += sample
        else:
            batch_list.append(sample)
    batch_list = list(zip(*batch_list))  # [[xyz, ...], [feat, ...], ...]

    offset, count = [], 0
    for item in batch_list[0]:
        count += item.shape[0]
        offset.append(count)
    offset = torch.IntTensor(offset)
    batch_list = [torch.cat(v) for v in batch_list]
    return [*batch_list, offset]


def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, origin='min'):
    """ coord, feat, label - an entire cloud
    """
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        # voxelize the entire cloud
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]

    if 'train' in split and voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0])
    else:
        # NOTE: not random during test
        init_idx = label.shape[0] // 2
    coord_init = coord[init_idx]

    if voxel_max and label.shape[0] > voxel_max:
        # radius crop with a random center point
        crop_idx = np.argsort(np.sum(np.square(coord - coord_init), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]
    xyz = coord

    if origin == 'min':
        coord_min = np.min(coord, 0)
        coord -= coord_min
    elif origin == 'mean':
        coord[..., :-1] -= coord[..., :-1].mean(0)
        coord[..., -1] -= coord[..., -1].min()
    elif origin == 'center':
        coord[..., :-1] -= coord_init[..., :-1]
        coord[..., -1] -= coord[..., -1].min()
    else:
        raise ValueError(f'not support origin={origin}')

    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    xyz = torch.FloatTensor(coord)
    return coord, feat, label, xyz

