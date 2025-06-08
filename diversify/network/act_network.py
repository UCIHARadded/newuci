# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import torch


var_size = {
    'emg': {
        'in_size': 8,
        'ker_size': 9
    },
    'uci_har': {
        'in_size': 570,  # 561 (features) + 9 (inertial)
        'ker_size': 9
    }
}


class ActNetwork(nn.Module):
    def __init__(self, taskname):
        super(ActNetwork, self).__init__()
        self.taskname = taskname
        in_ch = var_size[taskname]['in_size']
        kernel = var_size[taskname]['ker_size']

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=16, kernel_size=(1, kernel)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, kernel)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )

        # Init with dummy input to get correct flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 1, 128)
            out = self.conv2(self.conv1(dummy))
            self.in_features = out.view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.in_features)
        return x
