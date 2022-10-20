import os

import numpy as np
import SharedArray as SA

import torch
from torch.utils.data import Dataset

from util.data_util import *


class S3DIS(Dataset):
    def __init__(self, config, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        self.data_root = data_root
        self.test_area = test_area
        self.config = config

        self.initialize_data(split)
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

        self.initialize_gen()
        self.batch_limits = self.voxel_max * config.batch_size if self.voxel_max else None

    def initialize_data(self, split):
        data_list = sorted(os.listdir(self.data_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if split == 'train':
            data_list = [item for item in data_list if not 'Area_{}'.format(self.test_area) in item]
        else:
            data_list = [item for item in data_list if 'Area_{}'.format(self.test_area) in item]

        for item in data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(self.data_root, item + '.npy')
                data = np.load(data_path)  # xyzrgbl, N*7
                sa_create("shm://{}".format(item), data)
        self.data_list = data_list
        self.data_idx = np.arange(len(self.data_list))
        # self.collect_extra_info()

    def collect_extra_info(self):
        # extra info
        split = self.split
        config = self.config
        data_list = self.data_list
        # NOTE: only 33 cloud > voxel_max = 80000 (train)
        if config.data_gen == 'mixblock' and self.voxel_max is not None:
            print(split, self.voxel_size, self.voxel_max, flush=True)
            data_idx_block = []  # those larger than voxel_max
            data_idx_vx_size = []
            from util.voxelize import voxelize
            for i, item in enumerate(data_list):
                data = np.load(os.path.join(self.data_root, item + '.npy'))[:, 0:3]
                data -= data.min(0)
                pt_idx = voxelize(data, self.voxel_size, mode={'train': 0, 'val': 1, 'test': 1}[split])
                pt_idx = pt_idx[0] if isinstance(pt_idx, (list, tuple)) else pt_idx
                data_idx_vx_size += [len(pt_idx)]
                if len(pt_idx) > self.voxel_max:
                    data_idx_block += [i]
                print(f'{str(i):4}{str(data.shape):20}', '->\t', len(pt_idx), '\t', len(pt_idx) - self.voxel_max, len(pt_idx) > self.voxel_max)
            self.data_idx_block = data_idx_block
            print('max vx num = ', max(data_idx_vx_size))
            print(f'#totol cloud {len(data_list)} : #large cloud > voxel_max {len(data_idx_block)} / #small cloud {len(data_list)-len(data_idx_block)}', flush=True)
        return

    def initialize_gen(self):
        config = self.config
        data_gen = self.data_gen_random
        data_name = ['points', 'features', 'point_labels', 'xyz', 'cloud_inds']
        collate = self.collate_default

        self.data_gen = data_gen
        self.data_name = data_name
        self.collate_fn = collate
        return

    def data_gen_random(self, idx):
        """ random choose a cloud & radius crop in the cloud
        """
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label, xyz = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return coord, feat, label, xyz, torch.IntTensor([data_idx])

    def __getitem__(self, idx):
        return self.data_gen(idx)

    def __len__(self):
        return len(self.data_idx) * self.loop

    def collate_default(self, batch, return_type='dict'):
        """
        Args:
            batch - [(pts, feat, label), ...] - each tuple from a sampler
        Returns: [
            pts     : [BxN, 3]
            feat    : [BxN, d]
            label   : [BxN]
            ...
            offset  : [B]
            c_i     : [B]
        ]
        """
        batch_list = []
        for sample in batch:
            if isinstance(sample, list):
                batch_list += sample
            else:
                batch_list.append(sample)
        batch_list = list(zip(*batch_list))  # [[xyz, ...], [feat, ...], ...]
        # for n, v in zip(self.data_name, batch_list):
        #     print('---', n, '\n', v, flush=True)

        offset, count = [], 0
        for batch_i, item in enumerate(batch_list[0]):
            count += item.shape[0]
            if count > self.batch_limits:
                batch_i -= 1  # not including cur sample (compensated later)
                break
            offset.append(count)
        batch_i += 1  # compensate to include the last-sample

        offset = torch.IntTensor(offset)
        batch_list = [torch.cat(v[:batch_i]) for v in batch_list]
        if return_type == 'dict':
            return dict(zip(self.data_name, batch_list), offset=offset)
        return [*batch_list, offset]


class ConcatDataset(Dataset):
    """Dataset as a concatenation of multiple datasets.
    This class is useful to assemble different existing datasets.
    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes
