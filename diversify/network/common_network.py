# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        C, H, W = args.input_shape  # Get shape from args
        input_dim = C * H * W       # Compute flat input size

        self.fc = nn.Linear(input_dim, args.domain_num)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
