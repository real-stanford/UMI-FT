import torch
from torch import nn
import numpy as np
from typing import Dict
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder

class ProgressionNet(ModuleAttrMixin):
    def __init__(self,
                 hidden_dims,
                 obs_encoder: TimmObsEncoder,
                 **kwargs):
        super().__init__()
        self.obs_encoder = obs_encoder
        
        self.normalizer = LinearNormalizer()

        # create the model
        obs_feature_dim = obs_encoder.output_shape()[1]
        output_dim = 1

        hidden_dims.append(output_dim)

        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            prev_dim = obs_feature_dim if i == 0 else hidden_dims[i - 1]
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if i < len(hidden_dims) - 1:
                layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch: Dict):
        # normalize the observation
        nobs = self.normalizer.normalize(batch['obs']) 

        # encode the observation
        global_cond = self.obs_encoder(nobs)

        # forward pass through the model
        model_out = self.model(global_cond)

        return model_out
