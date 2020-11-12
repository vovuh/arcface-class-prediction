import pretrainedmodels as pm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pretrainedmodels.models.senet import SEResNeXtBottleneck


class ReSimpleModel(nn.Module):
    def __init__(self, bottleneck_size=256):
        super().__init__()
        self.se_resnext50_32x4d = pm.se_resnext50_32x4d()
        self.se_resnext50_32x4d.dropout = nn.Dropout(0.5)
        self.se_resnext50_32x4d.last_linear = nn.Linear(512 * SEResNeXtBottleneck.expansion, bottleneck_size)
        self.bn512 = nn.BatchNorm1d(bottleneck_size)

        model_settings = pm.pretrained_settings['se_resnext50_32x4d']['imagenet']
        _, self.input_height, self.input_width = model_settings['input_size']

    def set_gr(self, rg):
        for l in [self.se_resnext50_32x4d.layer0,
                  self.se_resnext50_32x4d.layer1,
                  self.se_resnext50_32x4d.layer2,
                  self.se_resnext50_32x4d.layer3,
                  self.se_resnext50_32x4d.layer4]:
            for param in l.parameters():
                param.requires_grad = rg

    def vectorize(self, x):
        x = self.forward(x)
        return F.normalize(x)

    def forward(self, x):
        x = self.se_resnext50_32x4d.features(x)
        x = self.se_resnext50_32x4d.logits(x)
        return self.bn512(x)

    def save(self, model_path):
        weights = self.state_dict()
        torch.save(weights, model_path)
