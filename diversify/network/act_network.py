# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

import torch.nn as nn

var_size = {
    'emg': {
        'in_size': (8,),   # 8 channels
        'ker_size': 9,
        'input_len': 200
    },
    'uci_har': {
        'in_size': (1,),
        'ker_size': 9,
        'input_len': 561
    }
}


class ActNetwork(nn.Module):
    def __init__(self, taskname):
        super(ActNetwork, self).__init__()
        self.taskname = taskname
        in_ch = var_size[taskname]['in_size'][0]
        ker_size = var_size[taskname]['ker_size']

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=16, kernel_size=(1, ker_size)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, ker_size)),  # Changed here
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_ch, 1, var_size[taskname]['input_len'])
            dummy_out = self.conv2(self.conv1(dummy_input))
            self.in_features = dummy_out.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = x.view(-1, self.in_features)
        return x
