# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn

var_size = {
    'uci_har': {
        'in_size': 570,  # 561 + 9 channels (fused input)
        'ker_size': 9,
    }
}

class ActNetwork(nn.Module):
    def __init__(self, taskname):
        super(ActNetwork, self).__init__()
        self.taskname = taskname
        in_ch = var_size[taskname]['in_size']
        ker_size = var_size[taskname]['ker_size']

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=16, kernel_size=(1, ker_size)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, ker_size)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )

        # Dynamically infer final feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_ch, 1, 128)  # (N, C, H=1, W=128)
            out = self.conv2(self.conv1(dummy_input))
            self.in_features = out.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return x
