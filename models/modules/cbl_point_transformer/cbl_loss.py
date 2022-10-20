import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .utils import *
from .blocks import *
from .heads import *
from .basic_operators import *
from .basic_operators import _eps, _inf

class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.contrast_head = ContrastHead(config.contrast, config) if 'contrast' in config else None
        self.xen = nn.CrossEntropyLoss(ignore_index=config.ignore_label)
    def forward(self, output, target, stage_list):
        loss_list = [self.xen(output, target)]
        if self.contrast_head is not None:
            loss_list += self.contrast_head(output, target, stage_list)
        return torch.stack(loss_list)