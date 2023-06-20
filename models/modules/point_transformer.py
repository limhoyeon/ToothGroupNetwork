import torch
from .cbl_point_transformer.cbl_point_transformer_module import get_model

class PointTransformerModule(torch.nn.Module):
    def __init__(self, config):
        self.config = config

        super().__init__()
        self.first_ins_cent_model = get_model(**config["model_parameter"], c=config["model_parameter"]["input_feat"], k=16 + 1)

    def forward(self, inputs, test=False):
        DEBUG=False
        """
        inputs
            inputs[0] => B, 6, 24000 : point features
            inputs[1] => B, 1, 24000 : ground truth segmentation
        """
        B, C, N = inputs[0].shape
        outputs = {}
        sem_1, offset_1, mask_1, first_features = self.first_ins_cent_model([inputs[0]])
        outputs.update({
            "sem_1": sem_1,
            "offset_1":offset_1,
            "mask_1":mask_1,
            "first_features": first_features,
            "cls_pred": sem_1
        })
        return outputs