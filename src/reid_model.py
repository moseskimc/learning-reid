
import os
import torch
import torchreid

from torchreid import models, utils, engine


class ReID_Model:

    def __init__(
        self,
        model_name="osnet_ain_x1_0",
        num_classes=1501 + 1467 + 1404,
        use_gpu=False
    ):

        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=num_classes,
            loss='softmax',
            pretrained=True,
            use_gpu=use_gpu
        )

    def load_weights(self, weight_path="models/osnet_ain_ms_d_c.pth.tar"):
        utils.load_pretrained_weights(self.model, weight_path)
        self.model.eval()
